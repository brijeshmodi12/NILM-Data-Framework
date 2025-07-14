# Wrapper for REFIT Dataset
import pandas as pd
import os
import json
from UnifiedNILM import TimeSeriesNILMDataset, Channel

class REFITLoader(TimeSeriesNILMDataset):
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
            # if house_id != 4:
            #     continue
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

                channel_id = i + 1
                # Create Channel object
                channel = Channel(
                    id=channel_id,
                    raw_label=raw_label,
                    unit="watts",
                    data_type="active",
                    data=df[[col]],
                    sample_rate="8S"
                )

                # Track unique appliances
                if raw_label.lower() != "aggregate":
                    if channel.universal_label not in self.appliances:
                        self.appliances.append(channel.universal_label)

                # Add channel to house dictionary
                # Consider changing this appliance key 'col' to universal name
                self.channels[house_id][channel_id] = channel
