import os, sys
from UnifiedNILM import UKDALELoader, REFITLoader, OlaLoader
from UnifiedNILM.utils.channel_utils import get_common_channels
import matplotlib.pyplot as plt

dataset_root = r'C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets'
dataset_list = [ 'refit', 'ola']
preload = False
dataset_type = 'preprocessed'  # example: choose the correct h5 type

loader_map = {
    'ukdale': UKDALELoader,
    'refit': REFITLoader,
    'ola': OlaLoader
}

datasets = {}

for name in dataset_list:
    print(f'Loading {name}...')
    # Initialize dataset instance
    dataset_path = os.path.join(dataset_root, name)
    loader_class = loader_map[name]
    ds = loader_class(dataset_name=name, path=dataset_path, preload_metadata=preload)

    # Load corresponding H5 file
    h5_path = os.path.join(ds.path, f"{name}.h5")
    ds.load_from_h5(h5_path)

    # Store dataset object
    datasets[name] = ds

# Example usage
# ukdale = datasets['ukdale']
refit = datasets['refit']
ola = datasets['ola']

chlist = get_common_channels([refit, ola],
                             required_labels=['dishwasher', 'television', 'tumble_dryer', 'aggregate'],
                             required_data_types=['active'])





# for (ds_name, house_id), channels in chlist.items():
#     for ch_id, ch in channels.items():
#         print(f"Dataset: {ds_name}, House: {house_id}, Type: {ch.data_type}, Label: {ch.universal_label}")



for (ds_name, house_id), channels in chlist.items():
    n_channels = len(channels)
    
    # Create one figure per house
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]  # Make axes iterable if only one channel
    
    # Sort so aggregate channel (if exists) is plotted first
    sorted_channels = sorted(channels.items(), key=lambda kv: (kv[1].raw_label.lower() != "aggregate", kv[0]))

    # Use matplotlib default color cycle (sequential colors)
    for ax, (ch_id, ch) in zip(axes, sorted_channels):
        df = ch.data
        
        ax.plot(df.index, df['power'], label=f"{ch.universal_label} ({ch.data_type})")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{ds_name} - House {house_id} - {ch.universal_label}")

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()