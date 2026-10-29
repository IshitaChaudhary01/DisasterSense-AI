import numpy as np
import torch

# Standard normalization parameters from Sen1Floods11 training
S1_MEAN = [0.6851, 0.5235]
S1_STD = [0.0820, 0.1102]

def preprocess_s1(s1_array):
    """
    Preprocesses Sentinel-1 SAR array (2, H, W) for inference.
    1. Replaces NaN / Inf values with 0
    2. Clips dB values to [-50, 1]
    3. Rescales linearly from [-50, 1] dB into [0, 1]
    4. Applies channel-wise Z-score normalization using Sen1Floods11 training stats.
    Returns:
        scaled: numpy array [0, 1] of shape (2, H, W)
        tensor: torch.FloatTensor of shape (1, 2, H, W) ready for PyTorch model
    """
    s1_array = np.asarray(s1_array, dtype=np.float32)
    if s1_array.ndim == 2:
        # single channel expanded
        s1_array = np.stack([s1_array, s1_array], axis=0)

    # Handle shape (H, W, C) vs (C, H, W)
    if s1_array.shape[-1] == 2 and s1_array.shape[0] != 2:
        s1_array = np.transpose(s1_array, (2, 0, 1))

    # Replace NaNs
    s1_clean = np.nan_to_num(s1_array, nan=-50.0, posinf=1.0, neginf=-50.0)

    # Clip and scale to [0, 1]
    s1_clipped = np.clip(s1_clean, -50.0, 1.0)
    s1_scaled = (s1_clipped + 50.0) / 51.0

    # Tensor normalization
    tensor = torch.from_numpy(s1_scaled.copy()).float()
    for c in range(min(2, tensor.shape[0])):
        tensor[c] = (tensor[c] - S1_MEAN[c]) / S1_STD[c]

    # Add batch dimension: (1, 2, H, W)
    tensor = tensor.unsqueeze(0)

    return s1_scaled, tensor


def preprocess_s2(s2_array):
    """
    Preprocesses Sentinel-2 optical imagery (13, H, W) or (C, H, W).
    Returns reflectance scaled to [0, 1] and computes common spectral water indices.
    """
    s2_array = np.asarray(s2_array, dtype=np.float32)
    if s2_array.shape[-1] == 13 and s2_array.shape[0] != 13:
        s2_array = np.transpose(s2_array, (2, 0, 1))

    s2_clean = np.nan_to_num(s2_array, nan=0.0)
    
    # Sentinel-2 L1C TOA reflectance is scaled by 10,000
    reflectance = np.clip(s2_clean / 10000.0, 0.0, 1.0)

    return reflectance


def compute_spectral_indices(s2_array):
    """
    Computes NDWI (Normalized Difference Water Index) and MNDWI (Modified NDWI).
    Band mapping:
      Band 1: Blue (B2)
      Band 2: Green (B3)
      Band 3: Red (B4)
      Band 7: NIR (B8)
      Band 11: SWIR-1 (B11)
    """
    s2_array = np.asarray(s2_array, dtype=np.float32)
    if s2_array.shape[-1] >= 12 and s2_array.shape[0] < 12:
        s2_array = np.transpose(s2_array, (2, 0, 1))

    green = s2_array[2] if s2_array.shape[0] > 2 else s2_array[0]
    nir = s2_array[7] if s2_array.shape[0] > 7 else green
    swir = s2_array[11] if s2_array.shape[0] > 11 else nir

    eps = 1e-7
    ndwi = (green - nir) / (green + nir + eps)
    mndwi = (green - swir) / (green + swir + eps)

    return {
        "ndwi": np.clip(ndwi, -1.0, 1.0),
        "mndwi": np.clip(mndwi, -1.0, 1.0)
    }
