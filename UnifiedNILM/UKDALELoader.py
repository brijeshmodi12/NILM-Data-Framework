# Wrapper for UKDALE Dataset
import pandas as pd
import os
import json
from UnifiedNILM import TimeSeriesNILMDataset, Channel


# For UKDALE
ACQUISITION_DEVICE_POWER_TYPES = {
    "EcoManagerWholeHouseTx": {"data_type": "apparent", "unit": "VA"},
    "EcoManagerTxPlug": {"data_type": "active", "unit": "watts"},
    "CurrentCostTx": {"data_type": "apparent", "unit": "VA"},
    "SoundCardPowerMeter": {"data_type": "active", "unit": "watts"},  # or "apparent" if preferred
}

class UKDALELoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        """
        Load UK-DALE dataset from house folders and channel_X.dat files.
        Metadata is loaded from ukdale_combined_metadata.json.
        Sampling rate, data type, and unit are inferred per channel.
        """
        self.houses = []
        self.channels = {}
        self.appliances = []

        self.sample_rates = {
            "aggregate": None,
            "appliance": None
        }

        self.metadata = {
            "features": ["aggregate", "appliance"],
            "source": "UK-DALE",
            "sampling_unit": "seconds"
        }

        # Load appliance metadata
        metadata_path = os.path.join(self.path, "metadata", "ukdale_combined_metadata.json")
        with open(metadata_path, "r") as f:
            appliance_metadata = json.load(f)

        # Process each house folder
        house_dirs = [d for d in os.listdir(self.path)
                      if d.startswith("house_") and os.path.isdir(os.path.join(self.path, d))]

        for house_dir in sorted(house_dirs):
            house_id = int(house_dir.replace("house_", ""))
            if house_id !=4:
                continue
            print(f"[INFO] Loading House {house_id}")
            self.houses.append(house_id)
            self.channels[house_id] = {}

            house_path = os.path.join(self.path, house_dir)
            house_key = f"House {house_id}"
            house_metadata = appliance_metadata.get(house_key, [])
            channel_map = {entry["channel"]: entry for entry in house_metadata}

            for file in os.listdir(house_path):
                if not file.startswith("channel_") or not file.endswith(".dat"):
                    continue

                if "button_press" in file:
                    continue

                channel_id = int(file.replace("channel_", "").replace(".dat", ""))
                file_path = os.path.join(house_path, file)
                print(f'Reading Channel: {channel_id}')

                try:
                    df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=None, names=["timestamp", "power"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.set_index("timestamp", inplace=True)

                    # Infer sampling rate
                    deltas = df.index.to_series().diff().dropna()
                    if deltas.empty:
                        sampling_rate = "unknown"
                    else:
                        seconds = int(deltas.median().total_seconds())
                        sampling_rate = f"{seconds}S"

                    # Load metadata for this channel
                    meta = channel_map.get(channel_id, {})
                    raw_label = meta.get("appliance_raw_label", f"channel_{channel_id}")
                    acquisition_device = meta.get("acquisition_device", "Unknown")
                    manufacturer = meta.get("manufacturer", "Unknown")
                    model = meta.get("model", "Unknown")

                    # Determine data_type and unit from acquisition device
                    device_info = ACQUISITION_DEVICE_POWER_TYPES.get(acquisition_device, {})
                    data_type = device_info.get("data_type", "unknown")
                    unit = device_info.get("unit", "unknown")

                    # Create Channel object with full metadata
                    ch = Channel(
                        id=channel_id,
                        raw_label=raw_label,
                        unit=unit,
                        data_type=data_type,
                        data=df[["power"]],
                        sample_rate=sampling_rate,
                        manufacturer=manufacturer,
                        model=model,
                        acquisition_device=acquisition_device
                    )

                    # Track unique appliance types
                    if ch.universal_label != "aggregate" and ch.universal_label not in self.appliances:
                        self.appliances.append(ch.universal_label)

                    # Store channel by channel ID
                    self.channels[house_id][channel_id] = ch

                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {e}")





