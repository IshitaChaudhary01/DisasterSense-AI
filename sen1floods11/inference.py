import numpy as np
import torch
from .preprocessing import preprocess_s1, compute_spectral_indices

def compute_otsu_threshold(image_u8):
    """
    Computes Otsu's optimal threshold on a uint8 image using NumPy.
    Maximizes between-class variance.
    """
    hist, _ = np.histogram(image_u8.flatten(), bins=256, range=(0, 256))
    total = image_u8.size
    sum_total = np.dot(np.arange(256), hist)
    
    current_max = 0.0
    threshold = 128
    sum_b = 0.0
    w_b = 0.0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * ((m_b - m_f) ** 2)
        if between_var > current_max:
            current_max = between_var
            threshold = t

    return threshold


def predict_otsu(s1_array):
    """
    Applies adaptive Otsu thresholding on Sentinel-1 SAR VH band.
    Water has low radar backscatter; pixels <= threshold are classified as water.
    Returns:
        binary_mask: (H, W) uint8 (1 for water, 0 for land)
        threshold_db: float value in dB
    """
    s1_array = np.asarray(s1_array, dtype=np.float32)
    vh = s1_array[1] if s1_array.shape[0] >= 2 else s1_array[0]
    
    vh_clipped = np.clip(vh, -50.0, 1.0)
    vh_u8 = ((vh_clipped + 50.0) / 51.0 * 255.0).astype(np.uint8)

    t_idx = compute_otsu_threshold(vh_u8)
    threshold_db = (t_idx / 255.0) * 51.0 - 50.0

    # Low backscatter = water
    water_mask = (vh_clipped <= threshold_db).astype(np.uint8)

    return water_mask, float(round(threshold_db, 2))


def predict_vh_threshold(s1_array, threshold_db=-25.13):
    """
    Thresholds the Sentinel-1 VH band at a fixed dB value
    (such as the country-calibrated threshold from Sen1Floods11 metadata).
    """
    s1_array = np.asarray(s1_array, dtype=np.float32)
    vh = s1_array[1] if s1_array.shape[0] >= 2 else s1_array[0]
    
    vh_clean = np.nan_to_num(vh, nan=0.0)
    water_mask = (vh_clean <= threshold_db).astype(np.uint8)

    return water_mask, float(threshold_db)


def predict_optical_ndwi(s2_array, index_name="ndwi", threshold=0.0):
    """
    Detects water from Sentinel-2 optical imagery using spectral indices:
    NDWI = (Green - NIR) / (Green + NIR)
    MNDWI = (Green - SWIR) / (Green + SWIR)
    Water typically has positive values (> 0.0).
    """
    indices = compute_spectral_indices(s2_array)
    selected_idx = indices.get(index_name.lower(), indices["ndwi"])
    
    water_mask = (selected_idx > threshold).astype(np.uint8)
    return water_mask, selected_idx


def predict_fcn(model, s1_array, device="cpu"):
    """
    Runs inference using the FCN-ResNet50 deep learning model on Sentinel-1 SAR.
    Returns:
        water_mask: (H, W) uint8 binary mask
        confidence: (H, W) float32 probability of water
    """
    model.eval()
    model.to(device)

    _, tensor = preprocess_s1(s1_array)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)["out"]
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        water_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy().astype(np.float32)

    return pred, water_prob
