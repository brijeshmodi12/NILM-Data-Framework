import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
import os, sys

# Load Dataset from Multiple Houses
def load_multiple_houses(house_ids, dataset_path, target_appliance, threshold_watts=20):
    all_X = []
    all_Y_single = []
    max_power = 0.0
    
    for house_id in house_ids:
        ptfile = os.path.join(dataset_path, f'h{house_id}_refit_tensor_dataset.pt')
        if not os.path.exists(ptfile):
            print(f"House {house_id} data not found.")
            continue

        print(f'Processing House: {house_id}')

        data = torch.load(ptfile)
        X = data['X']  # shape: [N, seq_len]
        Y_all = data['Y']  # shape: [N, seq_len, num_appliances]
        appliance_labels = data['appliance_labels']

        print(X.shape, Y_all.shape)

        if target_appliance not in appliance_labels:
            print(f"{target_appliance} not in appliance labels for house {house_id}")
            continue

        
        idx = appliance_labels.index(target_appliance)
        Y_single = Y_all[:, :, idx]     
           
        aggregate_max = torch.quantile(X.flatten(), 0.99)
        appliance_max = torch.quantile(Y_single.flatten(), 0.99)
        
        X = torch.clamp(X, 0.0, aggregate_max)
        Y_single = torch.clamp(Y_single, 0.0, appliance_max)

        all_X.append(X)
        all_Y_single.append(Y_single)
       
        print('--------------------')

    if not all_X:
        raise ValueError("No valid data found.")

    # === Concatenate across houses
    X_combined = torch.cat(all_X, dim=0)
    Y_combined = torch.cat(all_Y_single, dim=0)

    return X_combined, Y_combined


def normalize_using_train_stats(train_dataset, val_dataset, test_dataset, method="minmax"):
    # Extract features (x) and targets (y)
    x_train = torch.stack([x for x, y in train_dataset])
    y_train = torch.stack([y for x, y in train_dataset])

    x_val = torch.stack([x for x, y in val_dataset])
    y_val = torch.stack([y for x, y in val_dataset])

    x_test = torch.stack([x for x, y in test_dataset])
    y_test = torch.stack([y for x, y in test_dataset])

    # Normalize features
    if method == "zscore":
        x_mean = x_train.mean(dim=0)
        x_std = x_train.std(dim=0) + 1e-8
        x_norm_fn = lambda x: (x - x_mean) / x_std
    elif method == "minmax":
        x_min = x_train.min(dim=0).values
        x_max = x_train.max(dim=0).values
        x_norm_fn = lambda x: (x - x_min) / (x_max - x_min + 1e-8)
    else:
        raise ValueError("Unknown normalization method for X")

    # Normalize targets
    if method == "zscore":
        y_mean = y_train.mean(dim=0)
        y_std = y_train.std(dim=0) + 1e-8
        y_norm_fn = lambda y: (y - y_mean) / y_std
    elif method == "minmax":
        y_min = y_train.min(dim=0).values
        y_max = y_train.max(dim=0).values
        y_norm_fn = lambda y: (y - y_min) / (y_max - y_min + 1e-8)
    else:
        raise ValueError("Unknown normalization method for Y")

    # Apply normalization
    x_train = x_norm_fn(x_train)
    x_val = x_norm_fn(x_val)
    x_test = x_norm_fn(x_test)

    y_train = y_norm_fn(y_train)
    y_val = y_norm_fn(y_val)
    y_test = y_norm_fn(y_test)

    # Optionally, return normalization stats for inverse transform
    stats = {
        "x": {"mean": x_mean, "std": x_std} if method == "zscore" else {"min": x_min, "max": x_max},
        "y": {"mean": y_mean, "std": y_std} if method == "zscore" else {"min": y_min, "max": y_max},
    }

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), stats


def inverse_normalize(y_normalized, stats, method="minmax"):
    if method == "zscore":
        y_mean = stats["y"]["mean"]
        y_std = stats["y"]["std"]
        return y_normalized * y_std + y_mean
    elif method == "minmax":
        y_min = stats["y"]["min"]
        y_max = stats["y"]["max"]
        return y_normalized * (y_max - y_min + 1e-8) + y_min
    else:
        raise ValueError("Unknown normalization method")


def bert4nilm_loss_continuous(x_hat, x, temperature=0.1, lambda_=0.5):
    """
    x_hat: Predicted power sequence [batch, T]
    x: Ground truth power sequence [batch, T]
    """
    tau = temperature
    
    # 1. MSE loss
    mse = F.mse_loss(x_hat, x)

    # 2. KL divergence between softmaxed sequences (with temperature)
    x_hat_soft = F.softmax(x_hat / tau, dim=1)
    x_soft = F.softmax(x / tau, dim=1)
    kl = F.kl_div(x_hat_soft.log(), x_soft, reduction='batchmean')

    # 3. L1 loss across the full sequence (not masked)
    l1 = lambda_ * F.l1_loss(x_hat, x)

    # Final loss (continuous signal only)
    return mse + kl + l1