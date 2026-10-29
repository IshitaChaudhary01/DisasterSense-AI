import numpy as np

def evaluate_segmentation(pred_mask, gt_mask, ignore_index=-1):
    """
    Computes comprehensive flood segmentation metrics:
    - Water IoU (Jaccard Index)
    - Precision, Recall, F1-Score
    - Overall Accuracy
    - Confusion Matrix (TP, FP, TN, FN, Ignored pixels)
    """
    pred = np.asarray(pred_mask, dtype=np.int32).flatten()
    gt = np.asarray(gt_mask, dtype=np.int32).flatten()

    # Valid mask excludes no-data / ignore pixels (-1 or 255)
    valid = (gt != ignore_index) & (gt != 255) & (gt >= 0)
    ignored_count = int((~valid).sum())

    if valid.sum() == 0:
        return {
            "iou": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "ignored": ignored_count,
            "total_evaluated": 0
        }

    y_true = (gt[valid] == 1).astype(np.int32)
    y_pred = (pred[valid] == 1).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0

    total_valid = tp + fp + tn + fn
    accuracy = float((tp + tn) / total_valid) if total_valid > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "ignored": ignored_count,
        "total_evaluated": total_valid
    }


def compute_flood_area(water_mask, pixel_size_m=10.0):
    """
    Computes flood surface area given a binary water mask.
    Default pixel resolution: 10m x 10m (Sentinel-1 / Sentinel-2 standard).
    Returns area in square meters, hectares, and square kilometers.
    """
    water_pixels = int(np.sum(water_mask == 1))
    total_pixels = int(water_mask.size)

    area_per_pixel_m2 = pixel_size_m * pixel_size_m
    total_area_m2 = water_pixels * area_per_pixel_m2
    hectares = total_area_m2 / 10000.0
    sq_km = total_area_m2 / 1000000.0
    water_percentage = (water_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    return {
        "water_pixels": water_pixels,
        "total_pixels": total_pixels,
        "water_percentage": round(water_percentage, 2),
        "area_m2": round(total_area_m2, 2),
        "area_hectares": round(hectares, 2),
        "area_sq_km": round(sq_km, 3)
    }
