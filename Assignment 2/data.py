# Dataset utilities: train/val split, transforms, data loaders, and corruptions.

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    SEED, seed_everything, DEVICE, DATA_DIR, TRAIN_DIR, VAL_DIR, SPLIT_DIR,
    IMG_SIZE, INCEPTION_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD,
)

_PIN_MEMORY = DEVICE.type == "cuda"

# Train / Val split

def create_train_val_split(data_dir: str = DATA_DIR, val_frac: float = 0.20):
    """Create stratified 80/20 split from image-folder dataset."""
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print(f"[data] Split already exists at {SPLIT_DIR}")
        return
    print(f"[data] Creating {1-val_frac:.0%}/{val_frac:.0%} stratified split ...")
    seed_everything(SEED)
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = sorted(os.listdir(cls_path))
        np.random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_frac))
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        for img in imgs[n_val:]:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, cls, img))
        for img in imgs[:n_val]:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(VAL_DIR, cls, img))
    print(f"[data] Split done -> {SPLIT_DIR}")


def get_class_names() -> list[str]:
    """Return sorted list of class names from train dir."""
    return sorted(os.listdir(TRAIN_DIR))

# Transforms

def _img_size(model_name: str) -> int:
    return INCEPTION_SIZE if model_name == "inception_v3" else IMG_SIZE


def get_transforms(model_name: str, augment: bool = True):
    """Return appropriate transforms based on input size."""
    size = _img_size(model_name)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])

# Data loaders

def get_dataloaders(model_name: str = "resnet50", fraction: float = 1.0,
                    batch_size: int = BATCH_SIZE):
    """Return (train_loader, val_loader). Stratified sub-sampling for fraction < 1."""
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(model_name, augment=True))
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms(model_name, augment=False))

    if fraction < 1.0:
        seed_everything(SEED)
        targets = np.array(train_ds.targets)
        idx = []
        for c in range(NUM_CLASSES):
            ci = np.where(targets == c)[0]
            n  = max(1, int(len(ci) * fraction))
            idx.extend(np.random.choice(ci, n, replace=False).tolist())
        train_ds = Subset(train_ds, idx)
        print(f"  [data] Sub-sampled: {len(idx)}/{len(targets)} ({fraction*100:.0f}%)")

    common = dict(num_workers=NUM_WORKERS, pin_memory=_PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common)
    return train_ld, val_ld


def get_val_loader(model_name: str = "resnet50", batch_size: int = BATCH_SIZE):
    """Return val-only DataLoader (no augmentation)."""
    val_ds = datasets.ImageFolder(VAL_DIR, transform=get_transforms(model_name, augment=False))
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=_PIN_MEMORY,
                      persistent_workers=NUM_WORKERS > 0)

# Corruption transforms

class GaussianNoise:
    """Add Gaussian noise in pixel space (before normalization)."""
    def __init__(self, sigma: float):
        self.sigma = sigma
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t + torch.randn_like(t) * self.sigma, 0.0, 1.0)


class MotionBlur:
    """Horizontal motion blur via depthwise convolution."""
    def __init__(self, kernel_size: int = 15):
        self.ks = kernel_size
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        k = torch.zeros(self.ks, self.ks)
        k[self.ks // 2, :] = 1.0 / self.ks
        k = k.unsqueeze(0).unsqueeze(0).repeat(t.shape[0], 1, 1, 1)
        return F.conv2d(t.unsqueeze(0), k, padding=self.ks // 2, groups=t.shape[0]).squeeze(0)


class BrightnessShift:
    """Additive brightness shift in pixel space (before normalization)."""
    def __init__(self, factor: float = 0.3):
        self.factor = factor
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t + self.factor, 0.0, 1.0)


def corrupted_loader(corruption: str, model_name: str = "resnet50",
                     batch_size: int = BATCH_SIZE, **kwargs):
    """Build a DataLoader with corruption applied in pixel space, before normalization."""
    size = _img_size(model_name)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    base = [transforms.Resize((size, size)), transforms.ToTensor()]

    if corruption == "gauss":
        t = transforms.Compose(base + [GaussianNoise(kwargs.get("sigma", 0.1)), norm])
    elif corruption == "motion_blur":
        t = transforms.Compose(base + [MotionBlur(15), norm])
    elif corruption == "brightness":
        t = transforms.Compose(base + [BrightnessShift(0.3), norm])
    else:  # clean
        t = transforms.Compose(base + [norm])

    ds = datasets.ImageFolder(VAL_DIR, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=_PIN_MEMORY,
                      persistent_workers=NUM_WORKERS > 0)
