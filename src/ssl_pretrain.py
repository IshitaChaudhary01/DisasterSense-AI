import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader

from src.dataset import SimCLRSARDataset
from src.ssl_models import SimCLRModel, nt_xent_loss


def train_ssl(data_dir, output_checkpoint, epochs=5, batch_size=8, lr=1e-3, device="cpu"):
    print("=" * 65)
    print("=== Self-Supervised SimCLR Pretraining on Sentinel-1 SAR ===")
    print("=" * 65)
    print(f"Data directory:    {os.path.abspath(data_dir)}")
    print(f"Target checkpoint: {os.path.abspath(output_checkpoint)}")
    print(f"Device:            {device}")
    print(f"Epochs:            {epochs} | Batch size: {batch_size} | Learning rate: {lr}")

    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)

    # Collect all available Sentinel-1 GeoTIFF files
    s1_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".tif") and ("S1Hand" in f or "S1" in f):
                s1_files.append(os.path.join(root, f))

    if not s1_files:
        raise FileNotFoundError(f"No Sentinel-1 GeoTIFFs found in {data_dir}")

    print(f"Loaded {len(s1_files)} Sentinel-1 SAR chips for self-supervised pretraining.")

    dataset = SimCLRSARDataset(s1_files, crop_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = SimCLRModel(in_channels=2, backbone="resnet18")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    training_log = []
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        steps = 0

        for i, (view1, view2) in enumerate(dataloader):
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=0.1)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            steps += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(dataloader):
                sys.stdout.write(f"\rEpoch [{epoch}/{epochs}] Step [{i+1}/{len(dataloader)}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                sys.stdout.flush()

        epoch_loss = running_loss / max(1, steps)
        epoch_time = time.time() - epoch_start
        training_log.append({"epoch": epoch, "loss": epoch_loss, "time_sec": epoch_time})
        print(f"\n---> Epoch {epoch} complete in {epoch_time:.1f}s | Average NT-Xent Loss: {epoch_loss:.4f}")

    total_time = time.time() - start_total_time
    print(f"\nSimCLR pretraining completed in {total_time:.1f}s.")

    # Save pretrained state dict
    torch.save(model.state_dict(), output_checkpoint)
    print(f"Pretrained SSL weights saved to: {output_checkpoint}")
    return training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain SimCLR on SAR imagery")
    parser.add_argument("--data_dir", default="datasets/Sen1Floods11/v1.1/data/flood_events/HandLabeled/S1Hand")
    parser.add_argument("--output", default="checkpoints/ssl_pretrained_encoder.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_ssl(data_dir=args.data_dir, output_checkpoint=args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=dev)
