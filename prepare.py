"""
One-time data preparation for autoresearch vision experiments.
Prepares AIMS 3D printer fingerprint dataset from network share.

Usage:
    uv run prepare.py                  # full prep (cache images with downscale)

Data is stored in ~/.cache/autoresearch/vision/.
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DOWNSCALE_FACTOR = 2      # Factor to downscale images by
NUM_CLASSES = 6           # AIMS dataset has 6 printer classes
TRAIN_FRACTION = 1.00     # Use only 100% of training data
TIME_BUDGET = 3600        # 60 minute time budget in seconds

# Dataset configuration for AIMS cellphone dataset
DATASET_NAME = "aim_3_designs_all_views_cell"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
VISION_CACHE_DIR = os.path.join(CACHE_DIR, "vision")

# Network path for AIMS dataset
# On WSL2, access Windows network shares via /mnt/ or mount the share
# Alternative: use SMB mount in Linux
NETWORK_DATASET_PATH = "\\192.17.162.22\home\AIMS\datasets\aim_3_designs_all_views_cell"

# WSL2 mount path (if the share is mounted via /mnt/)
WSL2_MOUNT_PATH = "/mnt/aim_3_designs_all_views_cell"

# UNC mount path (Linux SMB mount)
UNC_MOUNT_PATH = "/mnt/192.17.162.22/home/AIMS/datasets/aim_3_designs_all_views_cell"

# Local fallback path (for when network is mounted locally)
LOCAL_DATASET_PATH = "data/aim_3_designs_all_views_cell"


def find_image_files(split="train"):
    """
    Find all image files in the specified split folder.

    AIMS dataset structure expected:
    - aim_3_designs_all_views_cell/
        - train/
            - Stratasys450mc-1/
            - Stratasys450mc-2/
            - ...
        - val/
            - Stratasys450mc-1/
            - ...
    """
    # Try paths in order: mounted network share, local, UNC mount, network share
    if os.path.exists(WSL2_MOUNT_PATH):
        base_dir = Path(WSL2_MOUNT_PATH)
    elif os.path.exists(LOCAL_DATASET_PATH):
        base_dir = Path(LOCAL_DATASET_PATH)
    elif os.path.exists(UNC_MOUNT_PATH):
        base_dir = Path(UNC_MOUNT_PATH)
    elif os.path.exists(NETWORK_DATASET_PATH):
        base_dir = Path(NETWORK_DATASET_PATH)
    else:
        raise FileNotFoundError(
            f"Dataset not found at {LOCAL_DATASET_PATH}, {UNC_MOUNT_PATH}, {WSL2_MOUNT_PATH}, or {NETWORK_DATASET_PATH}. "
            f"Please ensure the AIMS dataset is accessible."
        )

    split_dir = base_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist")

    image_files = []
    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    print(f"Found classes: {class_names}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        # Find all image files
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            for img_path in class_dir.glob(ext):
                image_files.append((img_path, class_idx, class_name))

    return image_files, class_names


def preprocess_image(image_path, downscale_factor=2):
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to the image file
        downscale_factor: Factor to downscale the image by (default 2)

    Returns:
        Preprocessed image tensor of shape (C, H//downscale, W//downscale)
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to open {image_path}: {e}")
        return None

    # Convert to tensor first
    img_tensor = transforms.ToTensor()(img)  # (C, H, W) in [0, 1]

    # Downscale using bilinear interpolation
    if downscale_factor > 1:
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        new_h, new_w = h // downscale_factor, w // downscale_factor
        img_tensor = transforms.Resize(
            (new_h, new_w),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )(img_tensor)

    # Per-image normalization (normalize each image to zero mean, unit variance)
    mean = img_tensor.mean(dim=(1, 2), keepdim=True)
    std = img_tensor.std(dim=(1, 2), keepdim=True) + 1e-6
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def create_cached_dataset(cache_file="cellphone_train_cache.pt", use_train_fraction=0.10):
    """
    Create a cached dataset from images. This speeds up training by pre-processing.

    Args:
        cache_file: Name of the cache file
        use_train_fraction: Fraction of training data to use (default 10%)

    Returns:
        Tuple of (images_tensor, labels_tensor, class_names)
        - images: list of tensors in (C, H', W') format where H'=H/2, W'=W/2
        - labels: list of class indices
        - class_names: list of class names
    """
    cache_path = os.path.join(VISION_CACHE_DIR, cache_file)

    if os.path.exists(cache_path):
        print(f"Dataset: loading cached dataset from {cache_path}")
        data = torch.load(cache_path, weights_only=False, map_location='cpu', mmap=True)
        return data["images"], data["labels"], data["class_names"]

    # Create cache directory
    os.makedirs(VISION_CACHE_DIR, exist_ok=True)

    print("Dataset: building training dataset...")
    t0 = time.time()

    train_images, class_names = find_image_files("train")

    print(f"Found {len(train_images)} training images across {len(class_names)} classes")

    # Sample only a fraction of training data
    if use_train_fraction < 1.0:
        n_samples = max(1, int(len(train_images) * use_train_fraction))
        random.seed(42)  # For reproducibility
        sampled_indices = random.sample(range(len(train_images)), n_samples)
        train_images = [train_images[i] for i in sampled_indices]
        print(f"Sampling {use_train_fraction*100:.0f}% of training data: {n_samples} images")

    images_list = []
    labels_list = []
    skipped = 0

    for img_path, label, class_name in tqdm(train_images, desc="Caching training images", unit="img"):
        try:
            img_tensor = preprocess_image(img_path, downscale_factor=DOWNSCALE_FACTOR)
            if img_tensor is not None:
                images_list.append(img_tensor)
                labels_list.append(label)
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} images due to errors")

    # Don't stack into single tensor - keep as list for variable sizes
    # Save as list of tensors
    torch.save({
        "images": images_list,
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "class_names": class_names,
        "image_shapes": [img.shape for img in images_list],
    }, cache_path)

    t1 = time.time()
    print(f"Dataset: built {len(images_list)} samples in {t1 - t0:.1f}s")
    print(f"Image shapes vary: {set(img.shape for img in images_list[:10])}")

    return images_list, torch.tensor(labels_list, dtype=torch.long), class_names


def create_val_dataset(cache_file="cellphone_val_cache.pt"):
    """
    Create validation dataset - uses ALL validation images.

    Returns:
        Tuple of (images_tensor, labels_tensor, class_names)
    """
    cache_path = os.path.join(VISION_CACHE_DIR, cache_file)

    if os.path.exists(cache_path):
        print(f"Validation dataset: loading cached dataset from {cache_path}")
        data = torch.load(cache_path, weights_only=False, map_location='cpu', mmap=True)
        return data["images"], data["labels"], data["class_names"]

    os.makedirs(VISION_CACHE_DIR, exist_ok=True)

    print("Validation dataset: building...")
    t0 = time.time()

    val_images, class_names = find_image_files("val")

    print(f"Found {len(val_images)} validation images")

    images_list = []
    labels_list = []
    skipped = 0

    for img_path, label, class_name in tqdm(val_images, desc="Caching validation images", unit="img"):
        try:
            img_tensor = preprocess_image(img_path, downscale_factor=DOWNSCALE_FACTOR)
            if img_tensor is not None:
                images_list.append(img_tensor)
                labels_list.append(label)
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} images due to errors")

    torch.save({
        "images": images_list,
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "class_names": class_names,
        "image_shapes": [img.shape for img in images_list],
    }, cache_path)

    t1 = time.time()
    print(f"Validation dataset: built {len(images_list)} samples in {t1 - t0:.1f}s")

    return images_list, torch.tensor(labels_list, dtype=torch.long), class_names


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class VisionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for vision images with variable sizes."""

    def __init__(self, images_list, labels_tensor):
        self.images = images_list  # List of tensors (C, H, W)
        self.labels = labels_tensor  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def collate_fn_variable(batch):
    """
    Custom collate function for variable-sized images.
    Pads images to the max size in the batch.
    """
    images, labels = zip(*batch)

    # Find max dimensions
    max_c = max(img.shape[0] for img in images)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Pad images to max size
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(
                img,
                (0, pad_w, 0, pad_h),  # left, right, top, bottom
                value=0.0
            )
        else:
            padded = img
        padded_images.append(padded)

    return torch.stack(padded_images, dim=0), torch.stack(labels, dim=0)


def make_dataloader(images_list, labels_tensor, batch_size, shuffle=True):
    """Create a DataLoader for the vision dataset with variable-sized images."""
    dataset = VisionDataset(images_list, labels_tensor)
    sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_variable,
        num_workers=2,
        pin_memory=True
    )


def make_val_dataloader(images_list, labels_tensor, batch_size):
    """Create a validation DataLoader (no shuffling)."""
    return make_dataloader(images_list, labels_tensor, batch_size, shuffle=False)


@torch.no_grad()
def evaluate_accuracy(model, val_loader, device="cuda"):
    """
    Compute top-1 accuracy on validation set.

    Returns:
        Top-1 accuracy as a decimal in [0, 1].
    """
    accuracy, correct, total = evaluate_accuracy_with_counts(model, val_loader, device)
    return accuracy


@torch.no_grad()
def evaluate_accuracy_with_counts(model, val_loader, device: str = "cuda"):
    """
    Compute top-1 accuracy and raw counts on the validation set.

    Returns:
        Tuple of (accuracy, correct, total), where accuracy is in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def get_num_classes():
    """Return the number of classification classes."""
    return NUM_CLASSES


def get_class_names():
    """Return the class names from the cached dataset."""
    cache_path = os.path.join(VISION_CACHE_DIR, "cellphone_train_cache.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False)
        return data.get("class_names", [])
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AIMS vision data for autoresearch")
    parser.add_argument("--verify", action="store_true", help="Just verify data exists, don't create cache")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of training data to use")
    args = parser.parse_args()

    print(f"Cache directory: {VISION_CACHE_DIR}")
    print(f"Looking for dataset at: {LOCAL_DATASET_PATH}, {WSL2_MOUNT_PATH}, or {NETWORK_DATASET_PATH}")
    print()

    if args.verify:
        # Just verify data exists
        if os.path.exists(LOCAL_DATASET_PATH):
            base_dir = Path(LOCAL_DATASET_PATH)
        elif os.path.exists(NETWORK_DATASET_PATH):
            base_dir = Path(NETWORK_DATASET_PATH)
        else:
            print("Error: Dataset directory not found!")
            sys.exit(1)

        for split in ["train", "val"]:
            split_dir = base_dir / split
            if split_dir.exists():
                class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                print(f"{split.capitalize()} directory exists with {len(class_dirs)} class folders")
            else:
                print(f"Warning: {split} directory not found")
    else:
        # Create cached datasets
        print("Creating training dataset cache...")
        train_images, train_labels, class_names = create_cached_dataset(
            use_train_fraction=args.train_fraction
        )
        print(f"Training data: {len(train_images)} samples")
        print(f"Labels shape: {train_labels.shape}")
        print(f"Class names: {class_names}")
        print()

        print("Creating validation dataset cache...")
        val_images, val_labels, val_class_names = create_val_dataset()
        print(f"Validation data: {len(val_images)} samples")
        print(f"Labels shape: {val_labels.shape}")
        print(f"Class names: {val_class_names}")

    print()
    print("Done! Ready to train.")
