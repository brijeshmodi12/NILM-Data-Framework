# Unified Framework for NILM Datasets
from abc import ABC, abstractmethod
import pandas as pd
import pickle
from UnifiedNILM.UniversalLabels import UNIVERSAL_LABEL_LIST, LABEL_KEYWORDS_MAP
import json
import h5py
import numpy as np

# Pending confirmation of power units for UK dale
# testing 
# adding support class for Ola house
# acquistion method, manufacturer etc for REFIT
# bring refit to same standards as ukdale in terms of computing sampling interval
# assessing power units etc.
# add feature list and version of class

class Channel:
    def __init__(
        self,
        id,
        raw_label,
        unit="watts",
        data_type="power",
        data=None,
        sample_rate=None,
        manufacturer="Unknown",
        model="Unknown",
        acquisition_device="Unknown"
    ):
        self.id = id
        self.raw_label = raw_label
        self.unit = unit
        self.data_type = data_type
        self.data = data
        self.sample_rate = sample_rate
        self.manufacturer = manufacturer
        self.model = model
        self.acquisition_device = acquisition_device
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
    def __init__(self, dataset_name, path, format="", preload_metadata=True):
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

    def get_channels_by_label(self, house_id, label):
        """
        Return all Channel objects in a house that match a given universal label.
        """
        label = label.lower()
        return [
            ch for ch in self.channels.get(house_id, {}).values()
            if ch.universal_label == label
        ]

    def supports(self, feature):
        return feature in self.metadata.get("features", [])

    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {file_path}")

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

            # Load sample rates
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

            # Load house/channel data
            for house_key in h5f:
                if not house_key.startswith("house_"):
                    continue

                house_id = int(house_key.replace("house_", ""))
                self.houses.append(house_id)
                house_group = h5f[house_key]
                self.channels[house_id] = {}

                shared_timestamps = None
                if "timestamps" in house_group:
                    shared_timestamps = pd.to_datetime(house_group["timestamps"][:], unit="s")

                for ch_key in house_group:
                    if ch_key == "timestamps":
                        continue

                    ch_group = house_group[ch_key]
                    power = ch_group["power"][:]

                    if "timestamps" in ch_group:
                        timestamps = pd.to_datetime(ch_group["timestamps"][:], unit="s")
                    else:
                        timestamps = shared_timestamps

                    df = pd.DataFrame(power, index=timestamps, columns=["power"])

                    raw_label = ch_group.attrs["raw_label"]
                    unit = ch_group.attrs.get("unit", "watts")
                    data_type = ch_group.attrs.get("data_type", "power")
                    sample_rate = ch_group.attrs.get("sample_rate", None)
                    universal_label = ch_group.attrs.get("universal_label", raw_label.lower())
                    manufacturer = ch_group.attrs.get("manufacturer", "Unknown")
                    model = ch_group.attrs.get("model", "Unknown")
                    acquisition_device = ch_group.attrs.get("acquisition_device", "Unknown")

                    channel = Channel(
                        id=ch_key,
                        raw_label=raw_label,
                        unit=unit,
                        data_type=data_type,
                        data=df,
                        sample_rate=sample_rate,
                        manufacturer=manufacturer,
                        model=model,
                        acquisition_device=acquisition_device
                    )

                    channel.universal_label = universal_label
                    self.channels[house_id][ch_key] = channel

                    if raw_label.lower() != "aggregate":
                        if channel.universal_label not in self.appliances:
                            self.appliances.append(channel.universal_label)

            self.metadata.setdefault("features", ["aggregate", "appliance"])

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

            # Save metadata
            metadata_grp = h5f.create_group("metadata")
            for key, val in self.metadata.items():
                metadata_grp.attrs[key] = json.dumps(val)

            # Save sensor data
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

                # Check if all channels share the same index
                timestamps_list = [ch.data.index for ch in channels.values()]
                aligned = all(t.equals(timestamps_list[0]) for t in timestamps_list)

                if aligned:
                    house_group.create_dataset("timestamps", data=timestamps_list[0].astype("int64") // 10**9)

                for ch_id, channel in channels.items():
                    ch_group = house_group.create_group(str(ch_id))
                    ch_group.create_dataset("power", data=channel.data.values.astype(np.float32))

                    # Save per-channel timestamps if unaligned
                    if not aligned:
                        ch_group.create_dataset("timestamps", data=channel.data.index.astype("int64") // 10**9)

                    # Save channel metadata
                    ch_group.attrs["raw_label"] = channel.raw_label
                    ch_group.attrs["universal_label"] = channel.universal_label
                    ch_group.attrs["unit"] = channel.unit
                    ch_group.attrs["data_type"] = channel.data_type
                    ch_group.attrs["sample_rate"] = channel.sample_rate or "unknown"
                    ch_group.attrs["manufacturer"] = getattr(channel, "manufacturer", "Unknown")
                    ch_group.attrs["model"] = getattr(channel, "model", "Unknown")
                    ch_group.attrs["acquisition_device"] = getattr(channel, "acquisition_device", "Unknown")

class TimeSeriesNILMDataset(BaseNILMDataset):

    def get_appliance_power(self, house_id, appliance, start=None, end=None):
        """
        Retrieve power time series of all appliance channels with a matching universal label.

        Returns:
            List[pd.DataFrame]: List of DataFrames (one per matching channel),
                                or empty list if not found.
        """
        appliance = appliance.lower()
        matched = [
            ch for ch in self.channels.get(house_id, {}).values()
            if ch.universal_label == appliance and ch.data_type == "power"
        ]

        results = []
        for ch in matched:
            df = ch.data.loc[start:end] if start or end else ch.data
            results.append(df)

        return results


    def get_aggregate(self, house_id, start=None, end=None):
        """
        Return the aggregate power time series for a given house.
        """
        for ch in self.channels.get(house_id, {}).values():
            if ch.raw_label.lower() == "aggregate":
                df = ch.data
                return df.loc[start:end] if start or end else df
        return pd.DataFrame()

