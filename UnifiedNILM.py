from abc import ABC, abstractmethod
import pandas as pd
import os
import pickle
from appliance_labels import UNIVERSAL_LABEL_LIST, LABEL_KEYWORDS_MAP
import json
import h5py
import numpy as np


class Channel:
    def __init__(self, id, raw_label, unit="watts", data_type="power", data=None):
        self.id = id
        self.raw_label = raw_label
        self.unit = unit
        self.data_type = data_type
        self.data = data
        self.universal_label = self.map_to_universal_label(raw_label)

    def map_to_universal_label(self, label):
        label_clean = label.lower().replace("_", " ").replace(",", " ")
        for universal, keywords in LABEL_KEYWORDS_MAP.items():
            for kw in keywords:
                if kw in label_clean:
                    return universal
        for universal in UNIVERSAL_LABEL_LIST:
            if universal.replace("_", " ") in label_clean:
                return universal
        return "other"


class BaseNILMDataset(ABC):
    def __init__(self, dataset_name, path, format="CSV", preload_metadata=True):
        self.dataset_name = dataset_name
        self.path = path
        self.format = format

        self.houses = []
        self.channels = {}
        self.appliances = []
        self.sensor_data = {}
        self.metadata = {}

        self.sample_rates = {
            "aggregate": None,
            "appliance": None,
            "event": None,
            "waveform": None
        }

        self.timezone = "UTC"
        self.access_type = "public"
        self.data_type = "continuous"

        if preload_metadata:
            self.load_metadata()

    @abstractmethod
    def load_metadata(self):
        pass

    def list_houses(self):
        return self.houses

    def list_appliances(self, house_id):
        return self.appliances

    def resample(self, df, target_rate):
        return df.resample(target_rate).mean()

    def supports(self, feature):
        return feature in self.metadata.get("features", [])

    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {file_path}")

    @staticmethod
    def load_from_pickle(file_path):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Dataset loaded from {file_path}")
        return obj

    def save_to_h5(self, output_path):
        with h5py.File(output_path, 'w') as h5f:
            h5f.attrs["dataset_name"] = self.dataset_name
            h5f.attrs["format"] = self.format
            h5f.attrs["timezone"] = self.timezone
            h5f.attrs["access_type"] = self.access_type
            h5f.attrs["data_type"] = self.data_type

            for house_id in self.houses:
                house_group = h5f.create_group(f"house_{house_id}")
                channels = self.channels.get(house_id, {})

                # Store timestamps once per house
                first_channel = next(iter(channels.values()))
                timestamps = first_channel.data.index.astype("int64") // 10**9
                house_group.create_dataset("timestamps", data=timestamps)

                for col_name, channel in channels.items():
                    group = house_group.create_group(str(col_name))
                    data = channel.data
                    group.create_dataset("power", data=data.values.astype(np.float32))
                    group.attrs["raw_label"] = channel.raw_label
                    group.attrs["universal_label"] = channel.universal_label
                    group.attrs["unit"] = channel.unit
                    group.attrs["data_type"] = channel.data_type


    def load_from_h5(self, h5_path):
        with h5py.File(h5_path, 'r') as h5f:
            self.dataset_name = h5f.attrs.get("dataset_name", "unknown")
            self.format = h5f.attrs.get("format", "unknown")
            self.timezone = h5f.attrs.get("timezone", "UTC")
            self.access_type = h5f.attrs.get("access_type", "public")
            self.data_type = h5f.attrs.get("data_type", "continuous")

            self.channels = {}
            self.houses = []
            self.appliances = []

            for house_key in h5f:
                house_id = int(house_key.replace("house_", ""))
                self.houses.append(house_id)
                house_group = h5f[house_key]

                timestamps = pd.to_datetime(house_group["timestamps"][:], unit='s')
                self.channels[house_id] = {}

                for ch_key in house_group:
                    if ch_key == "timestamps":
                        continue

                    ch_group = house_group[ch_key]
                    power = ch_group["power"][:]
                    df = pd.DataFrame(power, index=timestamps, columns=["power"])

                    raw_label = ch_group.attrs["raw_label"]
                    unit = ch_group.attrs.get("unit", "watts")
                    data_type = ch_group.attrs.get("data_type", "power")

                    channel = Channel(id=ch_key, raw_label=raw_label, unit=unit, data_type=data_type, data=df)
                    self.channels[house_id][ch_key] = channel

                    if raw_label.lower() != "aggregate":
                        if channel.universal_label not in self.appliances:
                            self.appliances.append(channel.universal_label)

            self.metadata["features"] = ["aggregate", "appliance"]

class TimeSeriesNILMDataset(BaseNILMDataset):
    @abstractmethod
    def get_aggregate(self, house_id, start=None, end=None):
        pass

    @abstractmethod
    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        pass


class REFITCSVLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        self.houses = []
        self.channels = {}
        self.appliances = []
        self.sample_rates = {}
        self.metadata = {}

        json_path = os.path.join(self.path, "refit_appliance_metadata.json")
        with open(json_path, "r") as f:
            appliance_metadata = json.load(f)

        csv_files = [f for f in os.listdir(self.path) if f.startswith("CLEAN_House") and f.endswith(".csv")]

        for file in csv_files:
            house_id = int(file.replace("CLEAN_House", "").replace(".csv", ""))
            if house_id != 1:
                continue

            print(f"Processing House: {house_id}")
            self.houses.append(house_id)
            file_path = os.path.join(self.path, file)

            df = pd.read_csv(file_path)
            for drop_col in ['Time', 'Issues']:
                if drop_col in df.columns:
                    df.drop(columns=[drop_col], inplace=True)

            df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
            df.set_index('Unix', inplace=True)

            if house_id not in self.channels:
                self.channels[house_id] = {}

            house_key = f"House {house_id}"
            house_appliance_map = {
                entry["channel"]: entry["appliance_raw_label"]
                for entry in appliance_metadata.get(house_key, [])
            }

            for i, col in enumerate(df.columns):
                if col.lower().startswith("aggregate"):
                    label = "aggregate"
                else:
                    label = house_appliance_map.get(i + 1, col.strip())

                channel = Channel(id=col, raw_label=label, unit="watts", data_type="power", data=df[[col]])

                if label != "aggregate":
                    if channel.universal_label not in self.appliances:
                        self.appliances.append(channel.universal_label)

                self.channels[house_id][col] = channel

        self.sample_rates["aggregate"] = 8
        self.sample_rates["appliance"] = 8
        self.metadata["features"] = ["aggregate", "appliance"]

    def get_aggregate(self, house_id, start=None, end=None):
        for ch in self.channels.get(house_id, {}).values():
            if ch.raw_label.lower() == "aggregate":
                df = ch.data
                return df.loc[start:end] if start and end else df
        return pd.DataFrame()

    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        for ch in self.channels.get(house_id, {}).values():
            if ch.universal_label == appliance and ch.data_type == "power":
                df = ch.data
                return df.loc[start:end] if start and end else df
        return pd.DataFrame()



class UKDaleRawCSVLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        house_dirs = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and d.startswith('house')]
        self.houses = [int(d.replace('house_', '').replace('house', '')) for d in house_dirs]

        for house_id in self.houses:
            print(f"Processing House: {house_id}")
            
            house_path = os.path.join(self.path, f"house_{house_id}")
            labels_path = os.path.join(house_path, "labels.dat")

            channel_info = {}
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            channel = int(parts[0])
                            label = " ".join(parts[1:])
                            unit = "watts"
                            channel_info[channel] = Channel(channel, label, unit)

            for file in os.listdir(house_path):
                if file.startswith("channel_") and file.endswith(".dat"):
                    channel_name = file.replace("channel_", "").replace(".dat", "")
                    if not channel_name.isdigit():
                        continue

                    channel_id = int(channel_name)
                    file_path = os.path.join(house_path, file)
                    df = pd.read_csv(file_path, sep=' ', names=['timestamp', 'power'], index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index, unit='s')

                    if house_id not in self.channels:
                        self.channels[house_id] = {}

                    raw_label = channel_info[channel_id].label if channel_id in channel_info else f"unknown_{channel_id}"
                    channel = Channel(channel_id, raw_label)
                    channel.data = df
                    self.channels[house_id][channel_id] = channel

                    if channel_id == 1:
                        self.aggregate_data[house_id] = df
                    else:
                        universal_label = channel.universal_label
                        if universal_label not in self.appliances:
                            self.appliances.append(universal_label)

        self.sample_rates["aggregate"] = 6
        self.sample_rates["appliance"] = 6
        self.metadata["features"] = ["aggregate", "appliance"]

    def get_aggregate(self, house_id, start=None, end=None):
        df = self.aggregate_data.get(house_id, pd.DataFrame())
        return df.loc[start:end] if start and end else df

    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        for ch in self.channels.get(house_id, {}).values():
            if ch.universal_label == appliance and ch.data_type == "power":
                df = ch.data
                return df.loc[start:end] if start and end else df
        return pd.DataFrame()



