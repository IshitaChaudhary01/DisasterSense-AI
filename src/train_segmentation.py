import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import load_split_list, FloodSegmentationDataset
from src.ssl_models import SAR_UNet, CombinedFloodLoss


def compute_batch_iou(logits, targets, ignore_index=255):
    """
    Computes Water class (class 1) IoU.
    """
    preds = torch.argmax(logits, dim=1)
    valid = (targets != ignore_index)

    p = preds[valid]
    t = targets[valid]

    intersection = ((p == 1) & (t == 1)).sum().item()
    union = ((p == 1) | (t == 1)).sum().item()

    return intersection, union


def train_model(
    split_dir,
    data_root,
    save_path,
    ssl_weights=None,
    fraction=1.0,
    epochs=5,
    batch_size=8,
    lr=5e-4,
    device="cpu"
):
    print("=" * 65)
    mode_title = "Self-Supervised Fine-Tuning" if ssl_weights else "Supervised Training from Scratch"
    print(f"=== {mode_title} (Label Fraction: {int(fraction * 100)}%) ===")
    print("=" * 65)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    s1_dir = os.path.join(data_root, "data/flood_events/HandLabeled/S1Hand")
    lbl_dir = os.path.join(data_root, "data/flood_events/HandLabeled/LabelHand")

    train_csv = os.path.join(split_dir, "flood_train_data.csv")
    valid_csv = os.path.join(split_dir, "flood_valid_data.csv")

    train_pairs = load_split_list(train_csv, s1_dir, lbl_dir)
    valid_pairs = load_split_list(valid_csv, s1_dir, lbl_dir)

    train_dataset = FloodSegmentationDataset(train_pairs, crop_size=256, is_train=True, fraction=fraction)
    valid_dataset = FloodSegmentationDataset(valid_pairs, crop_size=256, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training chips: {len(train_dataset)} | Validation chips: {len(valid_dataset)}")

    # Instantiate SAR U-Net
    model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
    if ssl_weights and os.path.exists(ssl_weights):
        model.load_ssl_encoder(ssl_weights, device=device)
        print(f"Initialized encoder with SSL pretrained weights: {ssl_weights}")
    else:
        print("Initialized encoder from scratch (random initialization).")

    model.to(device)

    criterion = CombinedFloodLoss(ce_weight=(1.0, 8.0), ignore_index=255, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_val_iou = -1.0
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_inter = 0
        train_union = 0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            inter, union = compute_batch_iou(logits, masks)
            train_inter += inter
            train_union += union

            if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                sys.stdout.write(f"\rEpoch [{epoch}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                sys.stdout.flush()

        train_iou = (train_inter / train_union) if train_union > 0 else 0.0
        train_loss /= len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_inter = 0
        val_union = 0

        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                inter, union = compute_batch_iou(logits, masks)
                val_inter += inter
                val_union += union

        val_iou = (val_inter / val_union) if val_union > 0 else 0.0
        val_loss /= len(valid_loader)
        epoch_dur = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou
        })

        print(f"\n---> Epoch {epoch}: Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} ({epoch_dur:.1f}s)")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"     [*] Model checkpoint saved with new best Val IoU: {val_iou:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val IoU: {best_val_iou:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flood Segmentation Model")
    parser.add_argument("--split_dir", default="datasets/Sen1Floods11/v1.1/splits/flood_handlabeled")
    parser.add_argument("--data_root", default="datasets/Sen1Floods11/v1.1")
    parser.add_argument("--save_path", default="checkpoints/ssl_finetuned_model.pt")
    parser.add_argument("--ssl_weights", default=None, help="Path to SSL pretrained encoder checkpoint")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(
        split_dir=args.split_dir,
        data_root=args.data_root,
        save_path=args.save_path,
        ssl_weights=args.ssl_weights,
        fraction=args.fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=dev
    )
