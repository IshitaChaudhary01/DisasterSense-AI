import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import load_split_list, FloodSegmentationDataset
from src.ssl_models import SAR_UNet
import sen1floods11 as sf


def evaluate_model_on_test_set(model, test_loader, device="cpu"):
    """
    Evaluates segmentation model on test dataset.
    Accumulates true confusion matrix across all test chips.
    """
    model.eval()
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_ignored = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            valid = (targets != 255)
            total_ignored += (~valid).sum().item()

            p = preds[valid]
            t = targets[valid]

            tp = ((p == 1) & (t == 1)).sum().item()
            fp = ((p == 1) & (t == 0)).sum().item()
            tn = ((p == 0) & (t == 0)).sum().item()
            fn = ((p == 0) & (t == 1)).sum().item()

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

    union = total_tp + total_fp + total_fn
    iou = total_tp / union if union > 0 else 0.0

    total_valid = total_tp + total_fp + total_tn + total_fn
    accuracy = (total_tp + total_tn) / total_valid if total_valid > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": round(float(iou), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_dice": round(float(f1), 4),
        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
        "ignored_pixels": total_ignored,
        "total_evaluated_pixels": total_valid
    }


def evaluate_otsu_baseline(test_pairs):
    """
    Evaluates Otsu radar thresholding baseline on the exact same test chips.
    """
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_ignored = 0

    import tifffile
    for s1_path, lbl_path in test_pairs:
        s1 = tifffile.imread(s1_path)
        lbl = tifffile.imread(lbl_path)

        mask, _ = sf.predict_otsu(s1)
        res = sf.evaluate_segmentation(mask, lbl)

        total_tp += res["tp"]
        total_fp += res["fp"]
        total_tn += res["tn"]
        total_fn += res["fn"]
        total_ignored += res["ignored"]

    union = total_tp + total_fp + total_fn
    iou = total_tp / union if union > 0 else 0.0
    total_valid = total_tp + total_fp + total_tn + total_fn
    accuracy = (total_tp + total_tn) / total_valid if total_valid > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": round(float(iou), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_dice": round(float(f1), 4),
        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
        "ignored_pixels": total_ignored,
        "total_evaluated_pixels": total_valid
    }


def plot_comparison_chart(metrics_dict, output_png):
    """
    Renders bar chart comparison of all evaluated models on held-out test set.
    """
    models = list(metrics_dict.keys())
    ious = [metrics_dict[m]["iou"] * 100 for m in models]
    f1s = [metrics_dict[m]["f1_dice"] * 100 for m in models]
    precisions = [metrics_dict[m]["precision"] * 100 for m in models]
    recalls = [metrics_dict[m]["recall"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#111827")

    rects1 = ax.bar(x - 1.5 * width, ious, width, label="Water IoU (%)", color="#00d2ff")
    rects2 = ax.bar(x - 0.5 * width, f1s, width, label="F1 / Dice (%)", color="#10b981")
    rects3 = ax.bar(x + 0.5 * width, precisions, width, label="Precision (%)", color="#3b82f6")
    rects4 = ax.bar(x + 1.5 * width, recalls, width, label="Recall (%)", color="#f59e0b")

    ax.set_ylabel("Score (%)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set Performance: Supervised vs Self-Supervised Models", color="#f8fafc", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#cbd5e1")
    ax.set_ylim(0, 100)

    # Label values on top of bars
    for rect in [rects1, rects2]:
        for bar in rect:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", color="#e2e8f0", fontsize=8, fontweight="bold")

    legend = ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {output_png}")


def run_evaluation(data_root="datasets/Sen1Floods11/v1.1", results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "sample_predictions"), exist_ok=True)

    test_csv = os.path.join(data_root, "splits/flood_handlabeled/flood_test_data.csv")
    s1_dir = os.path.join(data_root, "data/flood_events/HandLabeled/S1Hand")
    lbl_dir = os.path.join(data_root, "data/flood_events/HandLabeled/LabelHand")

    test_pairs = load_split_list(test_csv, s1_dir, lbl_dir)
    print(f"Loaded {len(test_pairs)} held-out test chips for rigorous evaluation.")

    test_dataset = FloodSegmentationDataset(test_pairs, crop_size=256, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    # 1. Classical Otsu Threshold Baseline
    print("\n--- Evaluating Otsu Radar Thresholding Baseline ---")
    otsu_res = evaluate_otsu_baseline(test_pairs)
    results["Otsu Baseline (SAR)"] = otsu_res
    print(f"Otsu Water IoU: {otsu_res['iou']:.4f} | F1: {otsu_res['f1_dice']:.4f} | Accuracy: {otsu_res['accuracy']:.4f}")

    # 2. Supervised Baseline Model
    sup_ckpt = "checkpoints/supervised_baseline.pt"
    if os.path.exists(sup_ckpt):
        print("\n--- Evaluating Supervised Baseline Model ---")
        sup_model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
        sup_model.load_state_dict(torch.load(sup_ckpt, map_location=device))
        sup_model.to(device)
        sup_res = evaluate_model_on_test_set(sup_model, test_loader, device=device)
        results["Supervised Baseline"] = sup_res
        print(f"Supervised Water IoU: {sup_res['iou']:.4f} | F1: {sup_res['f1_dice']:.4f} | Accuracy: {sup_res['accuracy']:.4f}")

    # 3. SSL Fine-Tuned Model
    ssl_ckpt = "checkpoints/ssl_finetuned_model.pt"
    if os.path.exists(ssl_ckpt):
        print("\n--- Evaluating Self-Supervised Fine-Tuned Model ---")
        ssl_model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
        ssl_model.load_state_dict(torch.load(ssl_ckpt, map_location=device))
        ssl_model.to(device)
        ssl_res = evaluate_model_on_test_set(ssl_model, test_loader, device=device)
        results["Self-Supervised (SimCLR)"] = ssl_res
        print(f"SSL Fine-Tuned Water IoU: {ssl_res['iou']:.4f} | F1: {ssl_res['f1_dice']:.4f} | Accuracy: {ssl_res['accuracy']:.4f}")

    # 4. Limited Label Experiment (20% Data)
    sup_20_ckpt = "checkpoints/supervised_baseline_20pct.pt"
    if os.path.exists(sup_20_ckpt):
        print("\n--- Evaluating Supervised Baseline (20% Labels) ---")
        sup_20_model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
        sup_20_model.load_state_dict(torch.load(sup_20_ckpt, map_location=device))
        sup_20_model.to(device)
        sup_20_res = evaluate_model_on_test_set(sup_20_model, test_loader, device=device)
        results["Supervised (20% Labels)"] = sup_20_res
        print(f"Supervised (20% labels) Water IoU: {sup_20_res['iou']:.4f} | F1: {sup_20_res['f1_dice']:.4f}")

    lim_ckpt = "checkpoints/ssl_finetuned_20pct.pt"
    if os.path.exists(lim_ckpt):
        print("\n--- Evaluating SSL Fine-Tuned (20% Labels) ---")
        lim_model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
        lim_model.load_state_dict(torch.load(lim_ckpt, map_location=device))
        lim_model.to(device)
        lim_res = evaluate_model_on_test_set(lim_model, test_loader, device=device)
        results["SSL (20% Labels)"] = lim_res
        print(f"SSL (20% labels) Water IoU: {lim_res['iou']:.4f} | F1: {lim_res['f1_dice']:.4f}")

    # Save JSON results
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll test evaluation metrics exported to: {metrics_path}")

    # Render comparison plot
    plot_path = os.path.join(results_dir, "benchmark_comparison.png")
    plot_comparison_chart(results, plot_path)

    return results


if __name__ == "__main__":
    run_evaluation()
