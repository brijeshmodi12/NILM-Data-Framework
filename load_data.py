from UnifiedNILM import UKDaleRawCSVLoader, REFITCSVLoader
import sys, os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import pandas as pd



# Choose dataset type 
dataset_type = "refit"  # or "ukdale"

# Load dataset
if dataset_type == "ukdale":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ukdale"
    dataset = UKDaleRawCSVLoader(dataset_name="ukdale", path=dataset_path)
elif dataset_type == "refit":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\refit_clean"
    dataset = REFITCSVLoader(dataset_name="refit", path=dataset_path)
else:
    raise ValueError("Unsupported dataset type")


# sys.exit()

# Choose a house to inspect
house_id = 3

# Print all channel info
if house_id in dataset.channels:
    print(f"\nChannels in House {house_id}:")
    for ch_id, channel in dataset.channels[house_id].items():
        print(f" - Channel ID: {ch_id}")
        print(f"   Label     : {channel.label}")
        print(f"   Unit      : {channel.unit}")
        print(f"   Data Type : {channel.data_type}")
        print(f"   Data Shape: {channel.data.shape}")
        print()
else:
    print(f"House {house_id} not found.")


# sys.exit()

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
            print(f"Skipping channel {channel.label} (no datetime index)")
            continue

        if start and end:
            df = df.loc[start:end]

        ax.plot(df.index, df.iloc[:, 0], label=f"{channel.label} ({channel.unit})")
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

plot_house_channels(dataset, house_id=3)
