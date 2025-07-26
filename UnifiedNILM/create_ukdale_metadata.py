import os
import json
import yaml

# Define root paths
metadata_dir = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ukdale\metadata"
labels_base_dir = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ukdale"

# Combined metadata output lines
output_lines = ["{"]

# Loop over houses
for house_num in range(1, 6):
    house_key = f"House {house_num}"
    building_file = os.path.join(metadata_dir, f"building{house_num}.yaml")
    labels_file = os.path.join(labels_base_dir, f"house_{house_num}", "labels.dat")

    if not os.path.exists(building_file) or not os.path.exists(labels_file):
        print(f"[Warning] Missing files for House {house_num}, skipping.")
        continue

    # Load building YAML
    with open(building_file, "r") as f:
        building_data = yaml.safe_load(f)

    # Load labels.dat
    with open(labels_file, "r") as f:
        label_lines = f.readlines()

    # Parse labels
    label_lookup = {}
    for line in label_lines:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            channel = int(parts[0])
            label_lookup[channel] = parts[1]

    # Map channel to acquisition device
    meter_device_map = {
        ch: meter.get("device_model", "Unknown")
        for ch, meter in building_data.get("elec_meters", {}).items()
    }

    # Map original_name to manufacturer/model
    original_name_map = {}
    channel_to_original_name = {}
    for appliance in building_data.get("appliances", []):
        original_name = appliance.get("original_name", "").strip().lower()
        manufacturer = appliance.get("manufacturer", "Unknown")
        model = appliance.get("model", "Unknown")
        for channel in appliance.get("meters", []):
            channel_to_original_name[channel] = original_name
            if original_name:
                original_name_map[original_name] = {
                    "manufacturer": manufacturer,
                    "model": model
                }

    # Construct per-house block
    output_lines.append(f'  "{house_key}": [')
    house_channels = sorted(label_lookup)
    for idx, channel in enumerate(house_channels):
        appliance_raw_label = label_lookup[channel]
        original_name = channel_to_original_name.get(channel, "").lower()
        manufacturer = original_name_map.get(original_name, {}).get("manufacturer", "Unknown")
        model = original_name_map.get(original_name, {}).get("model", "Unknown")
        acquisition_device = meter_device_map.get(channel, "Unknown")

        entry = {
            "channel": channel,
            "appliance_raw_label": appliance_raw_label,
            "manufacturer": manufacturer,
            "model": model,
            "acquisition_device": acquisition_device
        }
        comma = "," if idx < len(house_channels) - 1 else ""
        output_lines.append(f"    {json.dumps(entry, separators=(', ', ': '))}{comma}")
    output_lines.append("  ],")

# Finalize and save
# Replace last comma with closing brace
if output_lines[-1].strip() == "],":
    output_lines[-1] = "  ]"
output_lines.append("}")

# Save to JSON file
output_path = os.path.join(metadata_dir, "ukdale_combined_metadata_oneline.json")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"[✓] Metadata file written with one channel per line:\n{output_path}")
