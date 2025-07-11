# Loads H5 file and converts it into Tensors for Training.
import torch
from torch.utils.data import TensorDataset
import numpy as np
from UnifiedNILM import REFITCSVLoader, UKDaleRawCSVLoader
import os

def loadh5(h5_path, houses=None, seq_len=200, overlap=0):
    """
    Creates X: aggregate power sequence (N, seq_len),
            Y: appliance power output (N, seq_len, Z) with Z universal labels.

    Args:
        h5_path (str): Path to saved HDF5 file.
        houses (list[int]): List of house IDs to include (default: all).
        seq_len (int): Sequence length for input/output.
        overlap (int): Number of overlapping timesteps between sequences.

    Returns:
        TensorDataset: (X, Y), where
            - X shape: (N, seq_len)
            - Y shape: (N, seq_len, Z)
        List[str]: universal appliance labels (length Z)
    """
    assert seq_len > 0, "Sequence length must be positive"
    assert 0 <= overlap < seq_len, "Overlap must be between 0 and seq_len - 1"

    dataset = REFITCSVLoader("refit", path=h5_path, preload_metadata=False)
    dataset.load_from_h5(h5_path)
    

    if houses is None:
        houses = dataset.list_houses()

    # Identify all universal appliance labels across selected houses
    all_appliances = set()
    for house_id in houses:
        for ch in dataset.channels.get(house_id, {}).values():
            if ch.raw_label.lower() != "aggregate":
                all_appliances.add(ch.universal_label)

    all_appliances = sorted(list(all_appliances))
    appliance_index = {label: i for i, label in enumerate(all_appliances)}
    Z = len(all_appliances)

    X_list = []
    Y_list = []

    for house_id in houses:
        # Find aggregate channel
        agg_series = None
        for ch in dataset.channels.get(house_id, {}).values():
            if ch.raw_label.lower() == "aggregate":
                agg_series = ch.data.squeeze().astype(np.float32)
                break

        if agg_series is None:
            continue

        num_samples = (len(agg_series) - seq_len) // (seq_len - overlap) + 1
        if num_samples <= 0:
            continue

        # Get appliance data for this house
        appliance_data = {}
        for ch in dataset.channels.get(house_id, {}).values():
            if ch.raw_label.lower() != "aggregate":
                label = ch.universal_label
                appliance_data[label] = ch.data.squeeze().astype(np.float32)

        # Align all appliance series to aggregate length
        for i in range(num_samples):
            start = i * (seq_len - overlap)
            end = start + seq_len
            if end > len(agg_series):
                break

            X_seq = agg_series[start:end]
            Y_seq = np.zeros((seq_len, Z), dtype=np.float32)

            for label, series in appliance_data.items():
                if len(series) < end:
                    continue
                Y_seq[:, appliance_index[label]] = series[start:end]

            X_list.append(X_seq)
            Y_list.append(Y_seq)

    X_tensor = torch.tensor(X_list)               # shape: (N, seq_len)
    Y_tensor = torch.tensor(Y_list)               # shape: (N, seq_len, Z)

    return TensorDataset(X_tensor, Y_tensor), all_appliances

# Choose dataset type 
dataset_type = "refit"  # or "ukdale"
preload = False

# Load dataset
if dataset_type == "ukdale":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ukdale"
    # dataset = UKDaleRawCSVLoader(dataset_name="ukdale", path=dataset_path, preload_metadata=preload)
elif dataset_type == "refit":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\refit_clean"
    # dataset = REFITCSVLoader(dataset_name="refit", path=dataset_path, preload_metadata=preload)
else:
    raise ValueError("Unsupported dataset type")

house_id = 1
seq_len = 512 # number of samples
overlap = 0 # number of samples

for house_id in range(25):
    h5path = os.path.join(dataset_path, 'refit.h5')
    dataset, appliance_labels = loadh5(h5_path=h5path, houses=[house_id], seq_len=512,  overlap=0)

    print("Input shape (X):", dataset.tensors[0].shape)
    print("Target shape (Y):", dataset.tensors[1].shape)
    print("Appliance labels:", appliance_labels)

    X = dataset.tensors[0]
    Y = dataset.tensors[1]

    fname = f"h{house_id}_refit_tensor_dataset.pt"
    ptfile = os.path.join(dataset_path, fname)
    save_dict = {
        'X': X,
        'Y': Y,
        'appliance_labels': appliance_labels,
        'seq_len':seq_len,
        'overlap':overlap
    }

    if dataset.tensors[0].shape != torch.tensor([0]):
        torch.save(save_dict, ptfile)
        print(f'{fname} saved')
    else:
        print(f'{fname} skipped')
