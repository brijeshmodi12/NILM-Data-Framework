import pandas as pd
import os
import json
from UnifiedNILM import TimeSeriesNILMDataset, Channel


class REFITLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        """
        Loads metadata and channel data from the REFIT dataset.
        - Reads appliance metadata
        - Creates Channel objects with full metadata
        - Computes sample rate dynamically per channel
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
            "source": "REFIT",
            "sampling_unit": "seconds"
        }

        # Load appliance metadata mapping
        json_path = os.path.join(self.path, "refit_appliance_metadata.json")
        with open(json_path, "r") as f:
            appliance_metadata = json.load(f)

        # List CSV files
        csv_files = [f for f in os.listdir(self.path)
                     if f.startswith("CLEAN_House") and f.endswith(".csv")]

        for file in sorted(csv_files):
            house_id = int(file.replace("CLEAN_House", "").replace(".csv", ""))
            # if house_id != 1:
            #     continue
            print(f"Processing House: {house_id}")
            self.houses.append(house_id)

            file_path = os.path.join(self.path, file)
            df = pd.read_csv(file_path)

            # Remove non-sensor columns
            for drop_col in ['Time', 'Issues']:
                if drop_col in df.columns:
                    df.drop(columns=[drop_col], inplace=True)

            # Convert Unix to datetime index
            df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
            df.set_index('Unix', inplace=True)

            self.channels[house_id] = {}

            # Get house-specific appliance mapping
            house_key = f"House {house_id}"
            house_appliance_map = {
                entry["channel"]: entry
                for entry in appliance_metadata.get(house_key, [])
            }

            # Iterate over channels (columns)
            for i, col in enumerate(df.columns):
                channel_id = i + 1

                # Determine raw_label
                if col.lower().startswith("aggregate"):
                    raw_label = "aggregate"
                else:
                    raw_label = house_appliance_map.get(channel_id, {}).get("appliance_raw_label", col.strip())

                # Retrieve manufacturer/model from metadata if available
                meta_entry = house_appliance_map.get(channel_id, {})
                manufacturer = meta_entry.get("manufacturer", "")
                model = meta_entry.get("model", "")

                # ---- Compute sample rate dynamically ----
                deltas = df.index.to_series().diff().dropna()
                deltas = deltas[deltas > pd.Timedelta(0)]
                if not deltas.empty:
                    median_delta = deltas.median()
                    sample_rate = f"{int(median_delta.total_seconds())}S"
                else:
                    sample_rate = "8S"  # fallback

                # ---- Create Channel ----
                channel = Channel(
                    id=channel_id,
                    raw_label=raw_label,
                    unit="watts",
                    data_type="active",
                    data=df[[col]],
                    sample_rate=sample_rate,
                    manufacturer=manufacturer,
                    model=model,
                    acquisition_device="IAM"
                )

                # Track appliance list
                if raw_label.lower() != "aggregate":
                    if channel.universal_label not in self.appliances:
                        self.appliances.append(channel.universal_label)

                self.channels[house_id][channel_id] = channel

            # Optional: set dataset-wide sample rates (median)
            all_rates = [int(ch.sample_rate.replace("S", "")) for ch in self.channels[house_id].values() if ch.sample_rate != "unknown"]
            if all_rates:
                median_rate = f"{int(pd.Series(all_rates).median())}S"
                self.sample_rates["appliance"] = median_rate
                self.sample_rates["aggregate"] = median_rate
