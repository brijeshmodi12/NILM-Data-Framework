from typing import List, Dict, Tuple, Optional, Union
from UnifiedNILM.UnifiedNILM import BaseNILMDataset, Channel

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
