import io
import base64
import numpy as np
from PIL import Image

def percentile_stretch(band, p_min=2, p_max=98):
    """
    Applies percentile contrast stretching to satellite bands for optimal visualization.
    """
    band = np.nan_to_num(band, nan=0.0)
    valid_vals = band[band > 0]
    if len(valid_vals) > 0:
        b_min = np.percentile(valid_vals, p_min)
        b_max = np.percentile(valid_vals, p_max)
    else:
        b_min, b_max = np.min(band), np.max(band)

    if b_max > b_min:
        stretched = np.clip((band - b_min) / (b_max - b_min), 0.0, 1.0)
    else:
        stretched = np.zeros_like(band, dtype=np.float32)

    return (stretched * 255.0).astype(np.uint8)


def array_to_base64_png(img_array):
    """
    Converts a NumPy image array (uint8 HxW or HxWxC) to a Base64 data URL.
    """
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def create_s1_composite(s1_array):
    """
    Generates SAR visual representations:
    1. False Color Dual-Pol Composite (VV / VH / VV-VH ratio)
    2. VV Grayscale
    3. VH Grayscale
    """
    s1_array = np.asarray(s1_array, dtype=np.float32)
    vv = s1_array[0]
    vh = s1_array[1] if s1_array.shape[0] >= 2 else s1_array[0]

    # False color mapping:
    # R: VV in [-25, 0] dB
    # G: VH in [-32, -5] dB
    # B: (VV - VH) ratio in [0, 15] dB
    r = np.clip((vv + 25.0) / 25.0, 0.0, 1.0) * 255.0
    g = np.clip((vh + 32.0) / 27.0, 0.0, 1.0) * 255.0
    ratio = vv - vh
    b = np.clip(ratio / 15.0, 0.0, 1.0) * 255.0

    false_color = np.stack([r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)], axis=-1)

    vv_gray = percentile_stretch(vv, 2, 98)
    vh_gray = percentile_stretch(vh, 2, 98)

    return {
        "false_color": false_color,
        "vv": np.stack([vv_gray, vv_gray, vv_gray], axis=-1),
        "vh": np.stack([vh_gray, vh_gray, vh_gray], axis=-1)
    }


def create_s2_composite(s2_array):
    """
    Generates Sentinel-2 optical composites:
    1. True Color RGB (B4, B3, B2)
    2. False Color Infrared (B8, B4, B3)
    3. SWIR Agriculture/Water (B12, B8, B4)
    """
    s2_array = np.asarray(s2_array, dtype=np.float32)

    # Band indices in Sen1Floods11 13-band stack:
    # 1: B2 (Blue), 2: B3 (Green), 3: B4 (Red), 7: B8 (NIR), 12: B12 (SWIR-2)
    b_blue = s2_array[1]
    b_green = s2_array[2]
    b_red = s2_array[3]
    b_nir = s2_array[7]
    b_swir = s2_array[12] if s2_array.shape[0] > 12 else s2_array[11]

    # True Color RGB
    rgb = np.stack([
        percentile_stretch(b_red),
        percentile_stretch(b_green),
        percentile_stretch(b_blue)
    ], axis=-1)

    # False Color Infrared (Standard CIR - Color Infrared)
    cir = np.stack([
        percentile_stretch(b_nir),
        percentile_stretch(b_red),
        percentile_stretch(b_green)
    ], axis=-1)

    # SWIR Agriculture & Water
    swir = np.stack([
        percentile_stretch(b_swir),
        percentile_stretch(b_nir),
        percentile_stretch(b_red)
    ], axis=-1)

    return {
        "true_color": rgb,
        "false_color_cir": cir,
        "swir": swir
    }


def render_ground_truth(gt_mask):
    """
    Renders hand-labeled ground truth mask:
    - 1: Water (Bright Cyan: 0, 220, 255)
    - 0: Non-Water / Land (Dark Slate: 30, 41, 59)
    - -1: Invalid / No Data (Muted Gray: 100, 116, 139)
    """
    gt = np.asarray(gt_mask, dtype=np.int32)
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Non-water: dark slate background
    out[gt == 0] = [30, 41, 59]
    # Water: electric cyan
    out[gt == 1] = [0, 220, 255]
    # No data / ignore: muted slate
    out[gt < 0] = [100, 116, 139]
    out[gt == 255] = [100, 116, 139]

    return out


def render_prediction_mask(pred_mask):
    """
    Renders model prediction mask matching ground-truth styling:
    - 1: Predicted Water (Electric Cyan: [0, 220, 255])
    - 0: Predicted Land / Non-Water (Dark Slate: [30, 41, 59])
    """
    pred = np.asarray(pred_mask, dtype=np.uint8)
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[pred == 0] = [30, 41, 59]
    out[pred == 1] = [0, 220, 255]
    return out


def render_mask_overlay(base_rgb, mask, color=(0, 220, 255), alpha=0.55):
    """
    Overlays predicted water mask on top of base RGB satellite image.
    """
    base = base_rgb.astype(np.float32).copy()
    water_indices = (mask == 1)

    overlay = base.copy()
    overlay[water_indices] = (
        (1.0 - alpha) * base[water_indices] + alpha * np.array(color, dtype=np.float32)
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_error_map(pred_mask, gt_mask):
    """
    Renders classification error breakdown:
    - True Positive (Correct Water): Emerald Green [34, 197, 94]
    - False Positive (False Alarm): Red [239, 68, 68]
    - False Negative (Missed Flood): Amber/Orange [245, 158, 11]
    - True Negative (Correct Land): Dark Navy [15, 23, 42]
    - Ignored / No-Data: Slate [71, 85, 105]
    """
    pred = np.asarray(pred_mask, dtype=np.int32)
    gt = np.asarray(gt_mask, dtype=np.int32)
    h, w = pred.shape
    err_img = np.zeros((h, w, 3), dtype=np.uint8)

    valid = (gt >= 0) & (gt != 255)
    
    # Ignored pixels
    err_img[~valid] = [71, 85, 105]
    
    # True Negatives (Land detected as land)
    err_img[valid & (gt == 0) & (pred == 0)] = [15, 23, 42]
    
    # True Positives (Water detected as water)
    err_img[valid & (gt == 1) & (pred == 1)] = [34, 197, 94]
    
    # False Positives (Land mistakenly detected as water)
    err_img[valid & (gt == 0) & (pred == 1)] = [239, 68, 68]
    
    # False Negatives (Water missed as land)
    err_img[valid & (gt == 1) & (pred == 0)] = [245, 158, 11]

    return err_img
