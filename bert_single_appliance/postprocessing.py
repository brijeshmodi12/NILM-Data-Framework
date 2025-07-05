import numpy as np
from scipy.ndimage import median_filter
import torch

def filter_short_ons(preds, min_duration_seconds=80, sample_interval=8):
    """Removes ON segments shorter than the minimum duration in seconds."""
    min_duration = min_duration_seconds // sample_interval
    preds = np.array(preds).flatten() 
    preds = preds.copy()
    on_start = None
    for i, val in enumerate(preds):
        # print(type(val))
        v = int(val)  # ensure it's a scalar
        
        if v == 1 and on_start is None:
            on_start = i
        elif v == 0 and on_start is not None:
            duration = i - on_start
            if duration < min_duration:
                preds[on_start:i] = 0
            on_start = None
    if on_start is not None and len(preds) - on_start < min_duration:
        preds[on_start:] = 0
    return preds


def smooth_predictions(preds, window_seconds=40, sample_interval=8):
    """Applies a median filter to smooth flickering predictions."""
    window_size = max(1, window_seconds // sample_interval)
    return median_filter(preds, size=window_size)

def get_binary_predictions(power_preds, threshold_watts=20):
    """
    Converts continuous power predictions to binary ON/OFF based on threshold.
    Args:
        power_preds: numpy array or tensor of shape [N] or [B, L]
        threshold_watts: threshold in watts to consider ON (1) or OFF (0)
    """
    if isinstance(power_preds, torch.Tensor):
        power_preds = power_preds.detach().cpu().numpy()

    return (power_preds >= threshold_watts).astype(int)