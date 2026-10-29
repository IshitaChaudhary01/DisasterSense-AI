import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile

# Sen1Floods11 empirical normalization statistics for Sentinel-1 (VV, VH)
S1_MEAN = [0.6851, 0.5235]
S1_STD = [0.0820, 0.1102]


def load_split_list(split_csv_path, s1_dir, label_dir=None):
    """
    Reads split CSV and returns list of file paths.
    """
    pairs = []
    with open(split_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 1:
                continue
            s1_path = os.path.join(s1_dir, row[0])
            lbl_path = os.path.join(label_dir, row[1]) if (label_dir and len(row) > 1) else None
            if os.path.exists(s1_path) and (lbl_path is None or os.path.exists(lbl_path)):
                pairs.append((s1_path, lbl_path))
    return pairs


def normalize_s1(s1_arr):
    """
    Standard Sen1Floods11 preprocessing:
    1. Replace NaNs with -50 dB
    2. Clip to [-50, 1] dB
    3. Scale linearly to [0, 1]
    4. Normalize by channel mean and std
    """
    s1_clean = np.nan_to_num(s1_arr, nan=-50.0, posinf=1.0, neginf=-50.0)
    s1_clipped = np.clip(s1_clean, -50.0, 1.0)
    s1_scaled = (s1_clipped + 50.0) / 51.0  # Range: [0, 1]

    # Z-score normalize
    c0 = (s1_scaled[0] - S1_MEAN[0]) / S1_STD[0]
    c1 = (s1_scaled[1] - S1_MEAN[1]) / S1_STD[1]
    return np.stack([c0, c1], axis=0).astype(np.float32)


class SimCLRSARAugmentation:
    """
    SimCLR data augmentations tailored for 2-channel SAR radar physics.
    Unlike RGB color jitter, SAR requires backscatter-preserving transformations:
    - Random spatial crops
    - Random geometric reflections (H-flip, V-flip)
    - Radar speckle / Gaussian noise perturbation
    - Intensity scaling / calibration jitter
    """
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, s1_arr):
        # s1_arr is (2, 512, 512) raw in dB
        h, w = s1_arr.shape[1], s1_arr.shape[2]

        # 1. Random crop
        if h > self.crop_size and w > self.crop_size:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            crop = s1_arr[:, top:top + self.crop_size, left:left + self.crop_size].copy()
        else:
            crop = s1_arr.copy()

        # 2. Geometric flips
        if random.random() > 0.5:
            crop = np.flip(crop, axis=2).copy()  # Horizontal flip
        if random.random() > 0.5:
            crop = np.flip(crop, axis=1).copy()  # Vertical flip

        # 3. Radar intensity jitter (+- 1.5 dB)
        if random.random() > 0.5:
            shift = random.uniform(-1.5, 1.5)
            crop = crop + shift

        # 4. Speckle noise injection (SAR specific)
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.5, size=crop.shape)
            crop = crop + noise

        # Normalize to PyTorch tensor
        normalized = normalize_s1(crop)
        return torch.from_numpy(normalized).float()


class SimCLRSARDataset(Dataset):
    """
    Unlabeled Dataset for Self-Supervised SimCLR Pretraining.
    Returns two independently augmented views (view1, view2) for each SAR chip.
    """
    def __init__(self, s1_files, crop_size=256):
        self.s1_files = s1_files
        self.augment = SimCLRSARAugmentation(crop_size=crop_size)

    def __len__(self):
        return len(self.s1_files)

    def __getitem__(self, idx):
        path = self.s1_files[idx]
        arr = tifffile.imread(path)  # Shape (2, 512, 512)
        if arr.shape[0] > 2:
            arr = arr[:2]

        view1 = self.augment(arr)
        view2 = self.augment(arr)
        return view1, view2


class FloodSegmentationDataset(Dataset):
    """
    Supervised Dataset for Flood Water Segmentation & Fine-Tuning.
    Returns:
        image: (2, H, W) normalized float32 tensor
        mask:  (H, W) int64 tensor (0: land, 1: water, 255: ignore)
    """
    def __init__(self, file_pairs, crop_size=256, is_train=True, fraction=1.0, seed=42):
        if fraction < 1.0:
            random.seed(seed)
            sample_size = max(1, int(len(file_pairs) * fraction))
            self.file_pairs = random.sample(file_pairs, sample_size)
        else:
            self.file_pairs = file_pairs

        self.crop_size = crop_size
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        s1_path, lbl_path = self.file_pairs[idx]
        s1 = tifffile.imread(s1_path)  # (2, 512, 512)
        lbl = tifffile.imread(lbl_path)  # (512, 512)

        if s1.shape[0] > 2:
            s1 = s1[:2]

        # Convert invalid pixels (-1) to ignore_index=255
        lbl = np.asarray(lbl, dtype=np.int32)
        lbl[lbl == -1] = 255
        lbl[lbl < 0] = 255

        h, w = s1.shape[1], s1.shape[2]

        if self.is_train:
            # Random crop
            if h > self.crop_size and w > self.crop_size:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                s1 = s1[:, top:top + self.crop_size, left:left + self.crop_size]
                lbl = lbl[top:top + self.crop_size, left:left + self.crop_size]

            # Random Flips
            if random.random() > 0.5:
                s1 = np.flip(s1, axis=2).copy()
                lbl = np.flip(lbl, axis=1).copy()
            if random.random() > 0.5:
                s1 = np.flip(s1, axis=1).copy()
                lbl = np.flip(lbl, axis=0).copy()
        else:
            # Deterministic center crop or full image
            if self.crop_size and h > self.crop_size and w > self.crop_size:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
                s1 = s1[:, top:top + self.crop_size, left:left + self.crop_size]
                lbl = lbl[top:top + self.crop_size, left:left + self.crop_size]

        norm_s1 = normalize_s1(s1)

        img_tensor = torch.from_numpy(norm_s1).float()
        mask_tensor = torch.from_numpy(lbl).long()

        return img_tensor, mask_tensor
