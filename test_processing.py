# For testing the quality of conversion
# Raw data => H5 => Tensors.
# Assessing integrity of datasets during preprocessing

from UnifiedNILM.UnifiedNILM import UKDaleRawCSVLoader, REFITCSVLoader
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os, sys


# Choose dataset type 
dataset_type = "refit"  # or "ukdale"

# Set preload = true if you nead to read data from raw files during instance initiation (need to do this only once).
# Set preload = false if you nead to read data h5 files for visualization or basic testing.
preload = False 

# Load dataset
if dataset_type == "ukdale":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\ukdale"
    dataset = UKDaleRawCSVLoader(dataset_name="ukdale", path=dataset_path, preload_metadata=preload)
elif dataset_type == "refit":
    dataset_path = r"C:\Users\brind\OneDrive - Universitetet i Oslo\Codes\Alva\datasets\refit_clean"
    dataset = REFITCSVLoader(dataset_name="refit", path=dataset_path, preload_metadata=preload)
    house_ids = range(22)
else:
    raise ValueError("Unsupported dataset type")

# For Tensors (Loaded from .pt file)
for house_id in house_ids:
    ptfile = os.path.join(dataset_path, f'h{house_id}_{dataset_type}_tensor_dataset.pt')
    if not os.path.exists(ptfile):
        print(f"House {house_id} data not found.")
        continue

    data = torch.load(ptfile)
    X = data['X']  # shape: [N, seq_len]
    Y = data['Y']  # shape: [N, seq_len, num_appliances]
    appliance_labels = data['appliance_labels']
    print(f'House:{house_id}')
    print(X.shape)
    print(Y.shape)
    print('-----------------')

# For h5 (Loaded from .h5 file)


    
    if not preload:
        print('loading from H5...')
        h5path = os.path.join(dataset.path, 'refit.h5')
        dataset.load_from_h5(h5path)

        
        print(f'House Ids in H5 file {dataset.houses}')
    
    sys.exit()



    





    






