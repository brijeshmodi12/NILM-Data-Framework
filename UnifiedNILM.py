from abc import ABC, abstractmethod
import pandas as pd
import os
import pickle
from appliance_labels import UNIVERSAL_LABEL_LIST, LABEL_KEYWORDS_MAP
import json
import h5py
import numpy as np

class Channel:
    def __init__(self, id, raw_label, unit="watts", data_type="power", data=None, sample_rate=None):
        self.id = id
        self.raw_label = raw_label
        self.unit = unit
        self.data_type = data_type
        self.data = data
        self.sample_rate = sample_rate 
        self.universal_label = self.map_to_universal_label(raw_label)
        'Consider adding acquisition device, and manufacturer details?'

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

    def resample(self, new_rate):
        """
        Resample the channel's time series data to a new sampling rate.

        This function safely down-samples the channel's data to a specified rate,
        while explicitly disallowing upsampling to prevent the introduction of
        artificial or interpolated values.

        If the current sampling rate is unknown:
        - Attempts to infer it using pandas' `infer_freq()`.
        - Falls back to estimating the median time difference between samples.

        Parameters:
            new_rate (str): Target sampling interval (e.g., '8S' for 8 seconds).

        Notes:
        - If upsampling is detected (i.e., target rate < current rate), resampling is skipped.
        - The function updates `self.data` and `self.sample_rate` if resampling is performed.
        """
        if self.data is None or self.data.empty:
            print(f"[Info] No data to resample for channel {self.id}.")
            return

        # Get current sampling interval in seconds
        if self.sample_rate is not None:
            try:
                current_secs = pd.to_timedelta(self.sample_rate).total_seconds()
            except Exception as e:
                print(f"[Error] Invalid sample_rate '{self.sample_rate}' for channel {self.id}: {e}")
                return
        else:
            inferred_freq = pd.infer_freq(self.data.index)
            if inferred_freq is not None:
                current_secs = pd.to_timedelta(inferred_freq).total_seconds()
                self.sample_rate = inferred_freq
            else:
                # Estimate from median time difference
                delta = self.data.index.to_series().diff().dropna()
                if delta.empty:
                    print(f"[Warning] Not enough data to estimate sampling rate for channel {self.id}.")
                    return
                current_secs = delta.median().total_seconds()
                self.sample_rate = f"{int(current_secs)}S"
                print(f"[Info] Estimated sample_rate for channel {self.id} as {self.sample_rate} (non-uniform timestamps).")

        # Convert target rate to seconds
        try:
            target_secs = pd.to_timedelta(new_rate).total_seconds()
        except Exception as e:
            print(f"[Error] Invalid target rate '{new_rate}' for channel {self.id}: {e}")
            return

        if target_secs < current_secs:
            print(f"[Skipping] Upsampling not allowed: {self.sample_rate} → {new_rate} (channel {self.id})")
            return

        self.data = self.data.resample(new_rate).mean()
        self.sample_rate = new_rate
        print(f"[Resampled] Channel {self.id} to {new_rate}")

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

    def resample_all_channels(self, target_rate):
        """
        Resample all channels (appliance and aggregate) across all houses
        in the dataset to a specified sampling rate.

        This method calls the `resample()` method of each `Channel` object.
        It performs only downsampling — channels will be skipped if the target
        rate is finer than the current one (to avoid upsampling).

        Parameters:
            target_rate (str): Target sampling interval (e.g., '8S' for 8 seconds).
        """
        print(f"[Info] Starting dataset-wide resampling to {target_rate}...")

        for house_id in self.houses:
            if house_id not in self.channels:
                print(f"[Warning] No channels found for house {house_id}, skipping.")
                continue

            for ch_id, channel in self.channels[house_id].items():
                print(f"[Info] Resampling house {house_id}, channel {ch_id} ({channel.raw_label})...")
                channel.resample(target_rate)

        print(f"[Success] All channels resampled to {target_rate}.")


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
        """
        Save the entire dataset (metadata, appliances, channels, sensor data)
        to an HDF5 file in a structured and portable format.
        """
        with h5py.File(output_path, 'w') as h5f:
            # Save basic dataset attributes
            h5f.attrs["dataset_name"] = self.dataset_name
            h5f.attrs["format"] = self.format
            h5f.attrs["timezone"] = self.timezone
            h5f.attrs["access_type"] = self.access_type
            h5f.attrs["data_type"] = self.data_type
            h5f.attrs["path"] = self.path

            # Save appliance list
            h5f.create_dataset("appliances", data=np.array(self.appliances, dtype='S'))

            # Save global sample rates
            sample_rate_grp = h5f.create_group("sample_rates")
            for key, val in self.sample_rates.items():
                if val is not None:
                    sample_rate_grp.attrs[key] = val

            # Save metadata (as JSON strings)
            metadata_grp = h5f.create_group("metadata")
            for key, val in self.metadata.items():
                metadata_grp.attrs[key] = json.dumps(val)

            # Save sensor data (optional DataFrames)
            sensor_grp = h5f.create_group("sensor_data")
            for key, df in self.sensor_data.items():
                g = sensor_grp.create_group(str(key))
                g.create_dataset("values", data=df.values.astype(np.float32))
                g.create_dataset("index", data=df.index.astype("int64") // 10**9)
                g.attrs["columns"] = json.dumps(df.columns.tolist())

            # Save house-wise channel data
            for house_id in self.houses:
                house_group = h5f.create_group(f"house_{house_id}")
                channels = self.channels.get(house_id, {})

                if not channels:
                    continue

                # Store timestamps once per house
                first_channel = next(iter(channels.values()))
                timestamps = first_channel.data.index.astype("int64") // 10**9
                house_group.create_dataset("timestamps", data=timestamps)

                # Save each channel
                for ch_id, channel in channels.items():
                    ch_group = house_group.create_group(str(ch_id))
                    ch_group.create_dataset("power", data=channel.data.values.astype(np.float32))

                    # Save channel metadata as attributes
                    ch_group.attrs["raw_label"] = channel.raw_label
                    ch_group.attrs["universal_label"] = channel.universal_label
                    ch_group.attrs["unit"] = channel.unit
                    ch_group.attrs["data_type"] = channel.data_type
                    ch_group.attrs["sample_rate"] = channel.sample_rate if channel.sample_rate else "unknown"


    def load_from_h5(self, h5_path):
        """
        Load the dataset from an HDF5 file, restoring metadata,
        appliances, sensor data, and per-house channel structures.
        """
        with h5py.File(h5_path, 'r') as h5f:
            # Load dataset-level attributes
            self.dataset_name = h5f.attrs.get("dataset_name", "unknown")
            self.format = h5f.attrs.get("format", "unknown")
            self.timezone = h5f.attrs.get("timezone", "UTC")
            self.access_type = h5f.attrs.get("access_type", "public")
            self.data_type = h5f.attrs.get("data_type", "continuous")
            self.path = h5f.attrs.get("path", "")

            self.channels = {}
            self.houses = []
            self.appliances = []
            self.sensor_data = {}
            self.metadata = {}
            self.sample_rates = {}

            # Load appliances list
            if "appliances" in h5f:
                self.appliances = [a.decode() for a in h5f["appliances"][:]]

            # Load sample_rates
            if "sample_rates" in h5f:
                sr_grp = h5f["sample_rates"]
                for key in sr_grp.attrs:
                    self.sample_rates[key] = sr_grp.attrs[key]

            # Load metadata
            if "metadata" in h5f:
                md_grp = h5f["metadata"]
                for key in md_grp.attrs:
                    try:
                        self.metadata[key] = json.loads(md_grp.attrs[key])
                    except json.JSONDecodeError:
                        self.metadata[key] = md_grp.attrs[key]

            # Load sensor data
            if "sensor_data" in h5f:
                sensor_grp = h5f["sensor_data"]
                for key in sensor_grp:
                    g = sensor_grp[key]
                    values = g["values"][:]
                    index = pd.to_datetime(g["index"][:], unit="s")
                    columns = json.loads(g.attrs["columns"])
                    df = pd.DataFrame(values, index=index, columns=columns)
                    self.sensor_data[key] = df

            # Load house and channel data
            for house_key in h5f:
                if not house_key.startswith("house_"):
                    continue

                house_id = int(house_key.replace("house_", ""))
                self.houses.append(house_id)
                house_group = h5f[house_key]

                # Shared timestamps per house
                timestamps = pd.to_datetime(house_group["timestamps"][:], unit="s")
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
                    sample_rate = ch_group.attrs.get("sample_rate", None)
                    universal_label = ch_group.attrs.get("universal_label", raw_label.lower())

                    channel = Channel(
                        id=ch_key,
                        raw_label=raw_label,
                        unit=unit,
                        data_type=data_type,
                        data=df,
                        sample_rate=sample_rate
                    )

                    # Force override universal_label if provided
                    channel.universal_label = universal_label
                    self.channels[house_id][ch_key] = channel

                    if raw_label.lower() != "aggregate":
                        if channel.universal_label not in self.appliances:
                            self.appliances.append(channel.universal_label)

            # Fallback feature declaration
            self.metadata.setdefault("features", ["aggregate", "appliance"])

class TimeSeriesNILMDataset(BaseNILMDataset):
    @abstractmethod
    def get_aggregate(self, house_id, start=None, end=None):
        pass

    @abstractmethod
    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        pass


class REFITCSVLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        """
        Loads metadata and channel data from the REFIT dataset.
        Initializes the dataset structure including:
        - list of houses
        - appliance-to-channel mapping
        - appliance and aggregate power time series (sampled at 8 seconds)
        - channel objects with metadata and power data
        """
        # Initialize key dataset-level structures
        self.houses = []
        self.channels = {}
        self.appliances = []

        # REFIT uses fixed 8-second sampling
        self.sample_rates = {
            "aggregate": "8S",
            "appliance": "8S"
        }

        # Core metadata describing the dataset
        self.metadata = {
            "features": ["aggregate", "appliance"],
            "source": "REFIT",
            "sampling_unit": "seconds"
        }

        # Load appliance metadata mapping from JSON file
        json_path = os.path.join(self.path, "refit_appliance_metadata.json")
        with open(json_path, "r") as f:
            appliance_metadata = json.load(f)

        # Locate all house CSV files in the dataset path
        csv_files = [f for f in os.listdir(self.path)
                     if f.startswith("CLEAN_House") and f.endswith(".csv")]

        for file in sorted(csv_files):
            house_id = int(file.replace("CLEAN_House", "").replace(".csv", ""))
            if house_id != 1:
                continue
            print(f"Processing House: {house_id}")
            self.houses.append(house_id)

            file_path = os.path.join(self.path, file)

            # Load and clean the CSV file
            df = pd.read_csv(file_path)

            # Drop non-sensor columns if they exist
            for drop_col in ['Time', 'Issues']:
                if drop_col in df.columns:
                    df.drop(columns=[drop_col], inplace=True)

            # Convert 'Unix' column to datetime and set as index
            df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
            df.set_index('Unix', inplace=True)

            # Initialize channel dictionary for the current house
            self.channels[house_id] = {}

            # Get appliance mapping for this house from metadata
            house_key = f"House {house_id}"
            house_appliance_map = {
                entry["channel"]: entry["appliance_raw_label"]
                for entry in appliance_metadata.get(house_key, [])
            }

            # Iterate through each column (channel) in the CSV
            for i, col in enumerate(df.columns):
                # Identify if the column is aggregate or appliance
                if col.lower().startswith("aggregate"):
                    raw_label = "aggregate"
                else:
                    # Map based on channel number (1-indexed)
                    raw_label = house_appliance_map.get(i + 1, col.strip())

                # Create Channel object
                channel = Channel(
                    id=col,
                    raw_label=raw_label,
                    unit="watts",
                    data_type="power",
                    data=df[[col]],
                    sample_rate="8S"
                )

                # Track unique appliances
                if raw_label.lower() != "aggregate":
                    if channel.universal_label not in self.appliances:
                        self.appliances.append(channel.universal_label)

                # Add channel to house dictionary
                # Consider changing this appliance key 'col' to universal name
                self.channels[house_id][channel.universal_label] = channel

    def get_aggregate(self, house_id, start=None, end=None):
        """
        Retrieve the aggregate power time series for a given house.

        Parameters:
            house_id (int): The ID of the house.
            start (str/datetime, optional): Start of time window.
            end (str/datetime, optional): End of time window.

        Returns:
            pd.DataFrame: Time series of aggregate power.
        """
        for ch in self.channels.get(house_id, {}).values():
            if ch.raw_label.lower() == "aggregate":
                df = ch.data
                # Apply time range filter if provided
                if start or end:
                    return df.loc[start:end]
                return df
        # Return empty DataFrame if no aggregate channel found
        return pd.DataFrame()

    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        """
        Retrieve the power time series of a specific appliance in a house.

        Parameters:
            house_id (int): The ID of the house.
            appliance (str): Universal label of the appliance (e.g., 'fridge').
            start (str/datetime, optional): Start of time window.
            end (str/datetime, optional): End of time window.

        Returns:
            pd.DataFrame: Time series of appliance power.
        """
        for ch in self.channels.get(house_id, {}).values():
            if ch.universal_label == appliance and ch.data_type == "power":
                df = ch.data
                # Apply time range filter if provided
                if start or end:
                    return df.loc[start:end]
                return df
        # Return empty DataFrame if appliance not found
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



