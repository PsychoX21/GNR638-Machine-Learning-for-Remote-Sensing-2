"""
Data loading utilities with train/val split support
Optimized Hybrid Version:
- Images preloaded + resized once
- Augmentation applied dynamically per __getitem__
- Fully YAML-configurable augmentation
"""

import cv2
from pathlib import Path
import random
import zipfile


# ==========================================================
# Dataset Auto-Extraction
# ==========================================================

def ensure_dataset_extracted(dataset_name):
    datasets_dir = Path("datasets")
    zip_path = datasets_dir / f"{dataset_name}.zip"
    extract_path = datasets_dir / dataset_name

    if extract_path.exists():
        return str(extract_path)

    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
        print(f"Extracted to {extract_path}")
        return str(extract_path)

    raise FileNotFoundError(
        f"Neither {zip_path} nor {extract_path} found."
    )


# ==========================================================
# ImageFolderDataset
# ==========================================================

class ImageFolderDataset:

    def __init__(self,
                 root_dir,
                 image_size=32,
                 channels=3,
                 train=True,
                 val_split=0.2,
                 augmentation=None,
                 seed=42):

        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.channels = channels
        self.train = train
        self.val_split = val_split
        self.augmentation = augmentation or {}

        # --------------------------
        # Discover classes
        # --------------------------
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # --------------------------
        # Collect samples
        # --------------------------
        all_samples = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    all_samples.append((str(img_path), class_idx))

        # Use a local random instance to ensure the shuffle is identical
        # for both train=True and train=False splits given the same seed.
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1 - val_split))

        if train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        print(f"Preloading {len(self.samples)} "
              f"{'train' if train else 'val'} samples...")

        # --------------------------
        # Preload images into RAM
        # --------------------------
        self.base_images = []
        self.labels = []

        for img_path, label in self.samples:

            if self.channels == 1:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            img = cv2.resize(img, (self.image_size, self.image_size))

            if self.channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.base_images.append(img)
            self.labels.append(label)

        print(f"Preload complete. {len(self.base_images)} images in RAM.")

    # ==========================================================
    # Dataset interface
    # ==========================================================

    def __len__(self):
        return len(self.base_images)

    def __getitem__(self, idx):

        img = self.base_images[idx]
        label = self.labels[idx]

        if self.train and self.augmentation.get("enabled", False):
            img = self._augment(img.copy())

        # Normalize
        img = img.astype("float32")
        img /= 255.0

        # Convert HWC → CHW
        if self.channels == 1:
            img_list = img.flatten().tolist()
        else:
            ch = cv2.split(img)
            img_list = []
            img_list.extend(ch[0].flatten().tolist())
            img_list.extend(ch[1].flatten().tolist())
            img_list.extend(ch[2].flatten().tolist())

        return img_list, label

    # ==========================================================
    # Augmentation
    # ==========================================================

    def _augment(self, img):

        # ------------------------------------
        # Random Crop with Padding (CIFAR)
        # ------------------------------------
        pad = self.augmentation.get("random_crop_padding", 0)
        if pad > 0:
            img = cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                borderType=cv2.BORDER_REFLECT
            )
            h, w = img.shape[:2]
            top = random.randint(0, h - self.image_size)
            left = random.randint(0, w - self.image_size)
            img = img[top:top+self.image_size,
                      left:left+self.image_size]

        # ------------------------------------
        # Horizontal Flip
        # ------------------------------------
        flip_prob = self.augmentation.get("horizontal_flip", 0.0)
        if flip_prob > 0 and random.random() < flip_prob:
            img = cv2.flip(img, 1)

        # ------------------------------------
        # Rotation
        # ------------------------------------
        max_rotation = self.augmentation.get("rotation", 0)
        if max_rotation > 0:
            angle = random.uniform(-max_rotation, max_rotation)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        # ------------------------------------
        # Brightness / Contrast
        # brightness & contrast are strengths in [0,1]
        # ------------------------------------
        brightness = self.augmentation.get("brightness", 0)
        contrast = self.augmentation.get("contrast", 0)

        if brightness > 0 or contrast > 0:

            alpha = 1.0
            beta = 0.0

            if contrast > 0:
                alpha = random.uniform(1 - contrast, 1 + contrast)

            if brightness > 0:
                beta = random.uniform(
                    -brightness * 255,
                    brightness * 255
                )

            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img


# ==========================================================
# DataLoader
# ==========================================================

class DataLoader:

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):

        indices = list(range(self.num_samples))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):

            batch_indices = indices[i:i + self.batch_size]

            images = []
            labels = []

            for idx in batch_indices:
                img, label = self.dataset[idx]
                images.append(img)
                labels.append(label)

            yield images, labels
