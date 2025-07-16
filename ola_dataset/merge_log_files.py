# Merges all log files from Ola's Datasets
# Aggregate computed as sum of individual devices

import os
import re
import json 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import pandas as pd
from IPython.display import display, HTML
import pickle
import numpy as np
# --- Config ---
log_folder = r'C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ola'
log_pattern = re.compile(r"mqtt\.log\.\d{8}")

# --- Parse one log file ---
def parse_log_file(filepath):
    records = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("ts")
                payload = entry.get("payload")

                if not isinstance(payload, dict):
                    continue

                raw_name = payload.get("dst")
                if not raw_name or "/events" not in raw_name:
                    continue
                device_name = raw_name.replace("/events", "")

                params = payload.get("params", {})
                switch_data = params.get("switch:0", {})
                apower = switch_data.get("apower")

                if device_name and apower is not None:
                    records.append((ts, device_name, apower))
            except Exception:
                continue
    return records

# --- Read all logs ---
log_files = sorted([f for f in os.listdir(log_folder) if log_pattern.fullmatch(f)])
all_records = []
for filename in log_files:
    filepath = os.path.join(log_folder, filename)
    all_records.extend(parse_log_file(filepath))

if not all_records:
    print("No usable records found.")
    exit()

# --- Create and pivot DataFrame ---
df = pd.DataFrame(all_records, columns=["timestamp", "device_name", "apower"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.groupby(["timestamp", "device_name"]).mean().reset_index()
df_pivot = df.pivot(index="timestamp", columns="device_name", values="apower").fillna(0.0).sort_index()

# --- Add aggregate power ---
df_pivot["Aggregate"] = df_pivot.sum(axis=1)

# --- Sort devices by total power ---
device_totals = df_pivot.drop(columns=["Aggregate"]).sum().sort_values(ascending=False)
ordered_devices = ["Aggregate"] + device_totals.index.tolist()

# --- Save everything to a pickle file ---
with open("power_data.pkl", "wb") as f:
    pickle.dump({
        "df_pivot": df_pivot,
        "ordered_devices": ordered_devices
    }, f)

print("Data saved to power_data.pkl")
