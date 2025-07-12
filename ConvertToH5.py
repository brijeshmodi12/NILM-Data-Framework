# Reads raw data and stores it as in a standard format
# defined in UnifiedNILM
from UnifiedNILM import REFITLoader, UKDALELoader
import sys, os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import pandas as pd

# Choose dataset type 
dataset_type = "ukdale"  # or "ukdale"
print(f'Processing Dataset: {dataset_type}')

# Set preload = true if you nead to read data from raw files during instance initiation (need to do this only once).
# Set preload = false if you nead to read data h5 files for visualization or basic testing.
preload = True 

# Load dataset
if dataset_type == "ukdale":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ukdale"
    dataset = UKDALELoader(dataset_name="ukdale", path=dataset_path, preload_metadata=preload)
elif dataset_type == "refit":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\refit_clean"
    dataset = REFITLoader(dataset_name="refit", path=dataset_path, preload_metadata=preload)
else:
    raise ValueError("Unsupported dataset type")

if not preload:
    print('loading from H5...')
    h5path = os.path.join(dataset.path, f'{dataset_type}.h5')
    dataset.load_from_h5(h5path)
    print(dataset.houses)
    
    label_set = sorted(set(dataset.appliances))
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
else:
    output = os.path.join(dataset_path, f'{dataset_type}.h5')
    dataset.save_to_h5(output)
    print(f'Dataset {dataset_type} saved to h5')
    # sys.exit()

# Choose a house to inspect
house_id = 4

# Print all channel info
if house_id in dataset.channels:
    print(f"\nChannels in House {house_id}:")
    for ch_id, channel in dataset.channels[house_id].items():
        print(f" - Channel ID: {ch_id}")
        print(f"   Raw Label : {channel.raw_label}")
        print(f"   Unit      : {channel.unit}")
        print(f"   Data Type : {channel.data_type}")
        print(f"   Data Shape: {channel.data.shape}")
        print(f"   Universal Label: {channel.universal_label}")
        print(f"   Sample Rate: {channel.sample_rate}")
        
else:
    print(f"House {house_id} not found.")



def plot_house_channels(dataset, house_id, start=None, days=None):
    """
    Generalized plot function for any TimeSeriesNILMDataset.
    
    Args:
        dataset: Instance of UKDaleRawCSVLoader, REFITCSVLoader, etc.
        house_id: Integer house ID.
        start: Optional start datetime string (e.g., "2014-05-01").
        days: Number of days from start to plot. Ignored if start is None.
    """
    if house_id not in dataset.channels:
        print(f"House {house_id} not found.")
        return

    # Handle time window
    if start:
        start = pd.to_datetime(start)
        end = start + timedelta(days=days) if days else None
    else:
        start = end = None

    channels = dataset.channels[house_id]
    n = len(channels)
    fig, axs = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)

    if n == 1:
        axs = [axs]

    for i, (channel_id, channel) in enumerate(channels.items()):
        ax = axs[i]
        df = channel.data

        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Skipping channel {channel.universal_label} (no datetime index)")
            continue

        if start and end:
            df = df.loc[start:end]

        ax.plot(df.index, df.iloc[:, 0], label=f"{channel.universal_label} ({channel.unit})")
        ax.set_ylabel(channel.unit)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i != n - 1:
            ax.set_xticklabels([])

    # X-axis formatting for the last plot
    axs[-1].set_xlabel("Time")

    if start and days:  # Short time range
        axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=10))
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    else:  # Full timeline
        axs[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

    fig.suptitle(f"{dataset.dataset_name.upper()} House {house_id} - Power Channels", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_house_channels(dataset, house_id=house_id)
