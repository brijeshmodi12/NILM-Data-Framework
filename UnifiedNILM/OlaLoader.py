import pandas as pd
import os
import json
from UnifiedNILM import TimeSeriesNILMDataset, Channel

class OlaLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        self.houses = [1]
        self.channels = {1: {}}
        self.appliances = []
        self.sample_rates = {
            "aggregate": None,
            "appliance": None
        }
        self.metadata = {
            "features": ["appliance", "aggregate"],
            "source": "OLA",
            "sampling_unit": "seconds"
        }

        def infer_sample_rate(index) -> str:
            index = pd.to_datetime(index).sort_values()
            deltas = index.to_series().diff().dropna()
            deltas = deltas[deltas > pd.Timedelta(0)]

            if deltas.empty:
                return "unknown"

            median_delta = deltas.median()
            return f"{int(median_delta.total_seconds())}S"

        shelly_path = os.path.join(self.path, 'shelly_data.h5')
        emonesp_path = os.path.join(self.path, 'emonesp_data.h5')

        aggregate_frames = []

        # --- Load Shelly ---
        if os.path.exists(shelly_path):
            print("[INFO] Loading Shelly data...")
            df_shelly = pd.read_hdf(shelly_path, key="power")
            df_shelly.index = pd.to_datetime(df_shelly.index, unit='s')
            

            sampling_rate_shelly = infer_sample_rate(df_shelly.index)

            for col in df_shelly.columns:
                ch = Channel(
                    id=f"shelly_{col}",
                    raw_label=col,
                    unit="watts",
                    data_type="active",
                    data=df_shelly[[col]].rename(columns={col: "power"}),
                    sample_rate=sampling_rate_shelly,
                    manufacturer="Unknown",
                    model="Unknown",
                    acquisition_device="Shelly"
                )
                self.channels[1][ch.id] = ch
                if ch.universal_label not in self.appliances:
                    self.appliances.append(ch.universal_label)

            aggregate_frames.append(df_shelly)

        # --- Load EMONESP ---
        if os.path.exists(emonesp_path):
            print("[INFO] Loading EMONESP data...")
            df_emon = pd.read_hdf(emonesp_path, key="power")
            df_emon.index = pd.to_datetime(df_emon.index, unit='s')

            # if "timestamp" in df_emon.columns:
            #     df_emon["timestamp"] = pd.to_datetime(df_emon["timestamp"], unit="s")
            #     df_emon.set_index("timestamp", inplace=True)

            # df_emon.index = df_emon.index.astype("int64") // 10**9
            sampling_rate_emon = infer_sample_rate(df_emon.index)

            suffix_map = {
                "_W": ("active", "watts"),
                "_VA": ("apparent", "VA"),
                "_Q": ("reactive", "VA"),
                "_PF": ("pf", "unitless")
            }

            emon_active_columns = []

            for col in df_emon.columns:
                for suffix, (dtype, unit) in suffix_map.items():
                    if col.endswith(suffix):
                        base_label = col.replace(suffix, "")
                        ch = Channel(
                            id=f"emonesp_{col}",
                            raw_label=base_label,
                            unit=unit,
                            data_type=dtype,
                            data=df_emon[[col]].rename(columns={col: "power"}),
                            sample_rate=sampling_rate_emon,
                            manufacturer="Unknown",
                            model="Unknown",
                            acquisition_device="EMONESP"
                        )
                        self.channels[1][ch.id] = ch
                        if ch.universal_label not in self.appliances:
                            self.appliances.append(ch.universal_label)
                        if dtype == "active":
                            emon_active_columns.append(col)
                        break

            if emon_active_columns:
                aggregate_frames.append(df_emon[emon_active_columns])

        # --- Compute Aggregate ---
        if aggregate_frames:
            print("[INFO] Computing house-wide aggregate channel...")

            cleaned_frames = []
            for df in aggregate_frames:
                df_clean = df[~df.index.duplicated(keep='first')].sort_index()
                cleaned_frames.append(df_clean)

            aligned = pd.concat(cleaned_frames, axis=1).fillna(0)
            aggregate_series = aligned.sum(axis=1)
            aggregate_df = pd.DataFrame(aggregate_series, columns=["power"])

            agg_sample_rate = infer_sample_rate(aggregate_df.index)

            aggregate_channel = Channel(
                id="aggregate",
                raw_label="aggregate",
                unit="watts",
                data_type="active",
                data=aggregate_df,
                sample_rate=agg_sample_rate,
                manufacturer="Computed",
                model="Sum",
                acquisition_device="Composite"
            )

            self.channels[1]["aggregate"] = aggregate_channel
            self.sample_rates["aggregate"] = agg_sample_rate

            print("[INFO] Aggregate channel successfully added.")
