from typing import List, Dict, Tuple, Optional, Union
from UnifiedNILM.UnifiedNILM import BaseNILMDataset, Channel
import copy

def get_common_channels(
    datasets: Union[BaseNILMDataset, List[BaseNILMDataset]],
    required_labels: Optional[List[str]] = None,
    required_data_types: Optional[List[str]] = None
) -> Dict[Tuple[str, int], Dict[str, Channel]]:
    """
    Get channels that match specified criteria across houses in the given datasets.

    - If `required_labels` is provided, only houses that have ALL those labels are included.
      If a house has multiple channels for the same label, the one with highest variance 
      (and if tied, longest time series) is selected.
    - If `required_data_types` is provided, only channels whose `data_type` matches one of them are considered.
    - If neither is provided, all channels common across all houses and datasets are returned.

    Args:
        datasets (BaseNILMDataset | List[BaseNILMDataset]): Dataset(s) to search.
        required_labels (Optional[List[str]]): Universal labels to filter channels.
        required_data_types (Optional[List[str]]): List of allowed data types (e.g., ['active', 'reactive']).

    Returns:
        Dict[(dataset_name, house_id), {channel_id: Channel}]
    """
    if isinstance(datasets, BaseNILMDataset):
        datasets = [datasets]

    if not datasets:
        return {}

    required_labels_set = {lbl.lower() for lbl in required_labels} if required_labels else None
    required_types_set = {t.lower() for t in required_data_types} if required_data_types else None

    result: Dict[Tuple[str, int], Dict[str, Channel]] = {}

    for ds in datasets:
        for house_id, ch_dict in ds.channels.items():
            house_labels = {ch.universal_label.lower() for ch in ch_dict.values()}

            # House must contain all required labels
            if required_labels_set and not required_labels_set.issubset(house_labels):
                continue

            # Group channels by label
            grouped_channels: Dict[str, List[Tuple[str, Channel]]] = {}
            for ch_id, ch in ch_dict.items():
                if required_labels_set and ch.universal_label.lower() not in required_labels_set:
                    continue
                if required_types_set and ch.data_type.lower() not in required_types_set:
                    continue
                grouped_channels.setdefault(ch.universal_label.lower(), []).append((ch_id, ch))

            # Select best channel per label
            filtered: Dict[str, Channel] = {}
            for label, candidates in grouped_channels.items():
                if not candidates:
                    continue

                # Sort by variance (descending), then by time series length (descending)
                def channel_score(item):
                    _, ch = item
                    df = ch.data
                    variance = df['power'].var() if df is not None and not df.empty else 0
                    length = len(df) if df is not None else 0
                    return (variance if variance is not None else 0, length)

                best_ch_id, best_ch = max(candidates, key=channel_score)
                filtered[best_ch_id] = best_ch

            # Keep only if house has channels for ALL required labels (if specified)
            if required_labels_set and len(filtered) != len(required_labels_set):
                continue

            if filtered:
                result[(ds.dataset_name, house_id)] = filtered

    return result


def resample_all_channels(channels_input, new_rate):
    """
    Resample all given channels to a specified sampling rate and return a new 
    structure with resampled channels (does not modify the input).

    Supports:
    - Dictionary input: {(dataset_name, house_id): {channel_id: Channel}}
    - List input: [Channel, Channel, ...]

    Parameters:
        channels_input (dict | list): Channels to be resampled.
        new_rate (str): Target sampling interval (e.g., '8S' for 8 seconds).

    Returns:
        dict | list: A new structure with resampled channels.
    
    Potential optimization: Allow users to choose whether to resample in place or create a copy
    """

    def _clone_and_resample(channel_obj, channel_id=None, dataset_name=None, house_id=None):
        """Helper to create a copy of a channel and safely resample it."""
        new_channel = copy.deepcopy(channel_obj)
        try:
            new_channel.resample(new_rate)
        except Exception as e:
            ident = f"channel {channel_id}" if channel_id else "channel"
            if dataset_name and house_id:
                ident += f" in dataset '{dataset_name}', house '{house_id}'"
            print(f"[Error] Failed to resample {ident}: {e}")
        return new_channel

    # --- Case 1: Input is a dictionary ---
    if isinstance(channels_input, dict):
        new_dict = {}
        for (dataset_name, house_id), channels in channels_input.items():
            if not isinstance(channels, dict):
                print(f"[Warning] Expected dict of channels for {(dataset_name, house_id)}, got {type(channels)}. Skipping.")
                continue

            print(f"[Info] Resampling channels for dataset '{dataset_name}', house '{house_id}' to {new_rate}...")
            new_dict[(dataset_name, house_id)] = {
                channel_id: _clone_and_resample(channel_obj, channel_id, dataset_name, house_id)
                for channel_id, channel_obj in channels.items()
                if hasattr(channel_obj, "resample") and callable(channel_obj.resample)
            }
        return new_dict  # ✅ new dict with resampled copies

    # --- Case 2: Input is a plain list ---
    elif isinstance(channels_input, list):
        print(f"[Info] Resampling a list of {len(channels_input)} channels to {new_rate}...")
        return [
            _clone_and_resample(channel_obj, channel_id=f"list_index_{idx}")
            for idx, channel_obj in enumerate(channels_input)
            if hasattr(channel_obj, "resample") and callable(channel_obj.resample)
        ]

    else:
        raise TypeError("channels_input must be either a dict {(dataset_name, house_id): {channel_id: Channel}} or a list of Channel objects.")