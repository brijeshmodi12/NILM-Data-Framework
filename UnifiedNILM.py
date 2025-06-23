from abc import ABC, abstractmethod
import pandas as pd
import os
import pickle

class Channel:
    def __init__(self, id, label, unit="watts", data_type="power", data=None):
        self.id = id                  # channel number or name
        self.label = label            # appliance or sensor name
        self.unit = unit              # e.g., watts, volts, state
        self.data_type = data_type    # 'power', 'sensor', 'event', 'waveform'
        self.data = data              # pandas DataFrame (or path if lazy-loaded)

class BaseNILMDataset(ABC):
    def __init__(self, dataset_name, path, format="CSV", preload_metadata=True):
        self.dataset_name = dataset_name
        self.path = path
        self.format = format

        self.houses = []  # List of house IDs (e.g., [1, 2, 3]) found in the dataset path
        self.channels = {}  # {house_id: {channel_id: Channel}}
        self.appliances = []  # Flat list of unique appliance labels across all houses
        self.aggregate_data = {}  # Mains power data per house, {house_id: DataFrame}
        self.sensor_data = {}  # Non-power sensor data (e.g., button press), {house_id: {sensor_name: DataFrame}}
        self.metadata = {}  # Metadata like dataset description, timezone, features

        self.sample_rates = {
            "aggregate": None,
            "appliance": None,
            "event": None,
            "waveform": None
        }  # Sampling rates for each data type in Hz or seconds, depending on dataset

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
        """Serialize and save the entire dataset object to disk."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {file_path}")

    @staticmethod
    def load_from_pickle(file_path):
        """Load and return a dataset object from a saved pickle file."""
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Dataset loaded from {file_path}")
        return obj

class TimeSeriesNILMDataset(BaseNILMDataset):
    @abstractmethod
    def get_aggregate(self, house_id, start=None, end=None):
        pass

    @abstractmethod
    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        pass

class UKDaleRawCSVLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        house_dirs = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and d.startswith('house')]
        self.houses = [int(d.replace('house_', '').replace('house', '')) for d in house_dirs]

        for house_id in self.houses:
            if house_id != 3:
                continue
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
                            label = parts[1]
                            unit = parts[2] if len(parts) == 3 else "watts"
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

                    channel = channel_info.get(channel_id, Channel(channel_id, f"unknown_{channel_id}"))
                    channel.data = df
                    self.channels[house_id][channel_id] = channel

                    if channel_id == 1:
                        self.aggregate_data[house_id] = df
                    else:
                        appliance = channel.label
                        if appliance not in self.appliances:
                            self.appliances.append(appliance)

        self.sample_rates["aggregate"] = 6
        self.sample_rates["appliance"] = 6
        self.metadata["features"] = ["aggregate", "appliance"]

    def get_aggregate(self, house_id, start=None, end=None):
        df = self.aggregate_data.get(house_id, pd.DataFrame())
        if not df.empty:
            return df.loc[start:end] if start and end else df
        return df

    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        for ch in self.channels.get(house_id, {}).values():
            if ch.label == appliance and ch.data_type == "power":
                df = ch.data
                return df.loc[start:end] if start and end else df
        return pd.DataFrame()

class REFITCSVLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        self.houses = []
        csv_files = [f for f in os.listdir(self.path) if f.startswith("CLEAN_House") and f.endswith(".csv")]

        for file in csv_files:
            house_id = int(file.replace("CLEAN_House", "").replace(".csv", ""))
            if house_id != 3:
                continue

            print(f"Processing House: {house_id}")
            
            self.houses.append(house_id)
            file_path = os.path.join(self.path, file)

            df = pd.read_csv(file_path)
            df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
            df.set_index('Unix', inplace=True)

            if house_id not in self.channels:
                self.channels[house_id] = {}

            for col in df.columns:
                if col.lower().startswith("aggregate"):
                    channel = Channel(id=col, label="aggregate", unit="watts", data_type="power", data=df[[col]])
                    self.channels[house_id][col] = channel
                    self.aggregate_data[house_id] = df[[col]]
                elif col.lower().startswith("appliance"):
                    label = f"{col.strip()}"
                    channel = Channel(id=col, label=label, unit="watts", data_type="power", data=df[[col]])
                    self.channels[house_id][col] = channel
                    if label not in self.appliances:
                        self.appliances.append(label)
                elif col.lower() == "issues":
                    self.sensor_data.setdefault(house_id, {})
                    self.sensor_data[house_id]['issues'] = df[[col]]

        self.sample_rates["aggregate"] = 8
        self.sample_rates["appliance"] = 8
        self.metadata["features"] = ["aggregate", "appliance", "issues"]

    def get_aggregate(self, house_id, start=None, end=None):
        df = self.aggregate_data.get(house_id, pd.DataFrame())
        if not df.empty:
            return df.loc[start:end] if start and end else df
        return df

    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        for ch in self.channels.get(house_id, {}).values():
            if ch.label == appliance and ch.data_type == "power":
                df = ch.data
                return df.loc[start:end] if start and end else df
        return pd.DataFrame()


# For Future Expansion
class WaveformNILMDataset(BaseNILMDataset):
    @abstractmethod
    def get_waveform_segment(self, appliance, event_id):
        pass

    @abstractmethod
    def list_waveform_appliances(self):
        pass

class EventBasedNILMDataset(BaseNILMDataset):
    @abstractmethod
    def get_event_list(self, house_id):
        pass

    @abstractmethod
    def get_event_window(self, event_id):
        pass

class DEREnergyAwareDataset(TimeSeriesNILMDataset):
    @abstractmethod
    def get_solar_export(self, house_id, start=None, end=None):
        pass

    @abstractmethod
    def get_ev_charging_data(self, house_id, start=None, end=None):
        pass

    def is_battery_present(self, house_id):
        return "battery" in self.list_appliances(house_id)

class SensorEnhancedDataset(TimeSeriesNILMDataset):
    @abstractmethod
    def get_occupancy_data(self, house_id):
        pass

    @abstractmethod
    def get_environmental_data(self, house_id):
        pass
