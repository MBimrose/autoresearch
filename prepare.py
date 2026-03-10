"""
One-time data preparation for autoresearch vision experiments.
Downloads/organizes TinyImageNet data and creates dataloader.

Usage:
    uv run prepare_vision.py                  # full prep (index images, create splits)

Data is stored in ~/.cache/autoresearch/vision/.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 64           # image resolution (square)
PATCH_SIZE = 8            # patch size for ViT
NUM_CLASSES = 200         # TinyImageNet classes
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_SAMPLES = 10000        # number of samples for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
VISION_CACHE_DIR = os.path.join(CACHE_DIR, "vision")
DATA_DIR = Path("data/tiny-imagenet-200")

# Patch and embedding dimensions
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64 patches for 64x64 image with 8x8 patches


def find_image_files(split="train"):
    """Find all image files in the specified split folder."""
    if split == "test":
        return list(DATA_DIR.glob("test/images/*.JPEG"))
    elif split == "val":
        # Validation images are in val/labels and val/images
        val_dirs = []
        with open(DATA_DIR / "val" / "val_annotations.txt", "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    filename, wnid = parts[0], parts[1]
                    val_dirs.append((filename, wnid))
        return [(DATA_DIR / "val" / "images" / fn, wn) for fn, wn in val_dirs]
    elif split == "train":
        # Training images are organized by class subdirectories with images/ inside
        train_images = []
        with open(DATA_DIR / "wnids.txt", "r") as f:
            wnids = [line.strip() for line in f if line.strip()]
        
        for wnid in wnids:
            # Images are in train/wnid/images/*.JPEG (standard TinyImageNet format)
            class_images_dir = DATA_DIR / "train" / wnid / "images"
            if class_images_dir.exists():
                for img_file in class_images_dir.glob("*.JPEG"):
                    train_images.append((img_file, wnid))
        return train_images
    else:
        raise ValueError(f"Unknown split: {split}")


def build_wnid_to_idx_mapping():
    """Build mapping from WordNet ID to class index."""
    with open(DATA_DIR / "wnids.txt", "r") as f:
        wnids = [line.strip() for line in f if line.strip()]
    return {wnid: idx for idx, wnid in enumerate(wnids)}


def load_val_labels():
    """Load validation labels from val_annotations.txt."""
    labels = []
    with open(DATA_DIR / "val" / "val_annotations.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                filename, wnid = parts[0], parts[1]
    return labels


def preprocess_image(image_path):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("RGB")
    # Resize to target size
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    # Convert to numpy array and normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize using ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    # Transpose to (C, H, W) format
    return torch.from_numpy(arr.transpose(2, 0, 1))


def patchify(image_tensor):
    """
    Patchify an image tensor into patches.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) with ImageNet normalization
    
    Returns:
        Patches tensor of shape (num_patches, C * PATCH_SIZE * PATCH_SIZE)
    """
    C, H, W = image_tensor.shape
    assert H == IMAGE_SIZE and W == IMAGE_SIZE, f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {H}x{W}"
    
    # Reshape to (num_patches_H, PATCH_SIZE, num_patches_W, PATCH_SIZE, C)
    patches = image_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    # Reshape to (num_patches, C * PATCH_SIZE * PATCH_SIZE)
    patches = patches.contiguous().view(NUM_PATCHES, -1)
    return patches


def create_cached_dataset(cache_file="dataset_cache.pt"):
    """
    Create a cached dataset from images. This speeds up training by pre-processing.
    
    Returns:
        Tuple of (images_tensor, labels_tensor) - images in (B, C, H, W) format
    """
    cache_path = os.path.join(VISION_CACHE_DIR, cache_file)
    
    if os.path.exists(cache_path):
        print(f"Dataset: loading cached dataset from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["images"], data["labels"]
    
    # Create cache directory
    os.makedirs(VISION_CACHE_DIR, exist_ok=True)
    
    print("Dataset: building training dataset...")
    t0 = time.time()
    
    wnid_to_idx = build_wnid_to_idx_mapping()
    train_images = find_image_files("train")
    
    images_list = []
    labels_list = []
    
    for img_path, wnid in train_images:
        if wnid not in wnid_to_idx:
            continue
        try:
            img_tensor = preprocess_image(img_path)
            # Store as (C, H, W) - will be batched by dataloader
            images_list.append(img_tensor)
            labels_list.append(wnid_to_idx[wnid])
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
    
    images_tensor = torch.stack(images_list, dim=0)  # (N, C, H, W)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    # Save cache
    torch.save({
        "images": images_tensor,
        "labels": labels_tensor
    }, cache_path)
    
    t1 = time.time()
    print(f"Dataset: built {len(images_list)} samples in {t1 - t0:.1f}s")
    
    return images_tensor, labels_tensor


def create_val_dataset():
    """Create validation dataset from TinyImageNet val folder."""
    wnid_to_idx = build_wnid_to_idx_mapping()
    val_images = find_image_files("val")
    
    images_list = []
    labels_list = []
    
    for img_info in val_images:
        if isinstance(img_info, tuple):
            img_path, wnid = img_info
        else:
            img_path = img_info
        
        try:
            img_tensor = preprocess_image(str(img_path))
            # Store as (C, H, W) - same format as training data
            images_list.append(img_tensor)
            
            # Get label from filename or use default
            if isinstance(img_info, tuple):
                labels_list.append(wnid_to_idx.get(wnid, 0))
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
    
    if len(images_list) > 0:
        images_tensor = torch.stack(images_list[:EVAL_SAMPLES], dim=0)
        labels_tensor = torch.tensor(labels_list[:EVAL_SAMPLES], dtype=torch.long)
        return images_tensor, labels_tensor
    
    # Never fall back to training samples for validation, this would leak data.
    raise RuntimeError(
        "Validation dataset is empty or unreadable. "
        "Refusing to fall back to training samples. "
        "Please verify data/tiny-imagenet-200/val and val_annotations.txt."
    )


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train_vision.py)
# ---------------------------------------------------------------------------

class VisionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for vision images."""
    
    def __init__(self, images_tensor, labels_tensor):
        self.images = images_tensor  # (N, C, H, W)
        self.labels = labels_tensor  # (N,)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_dataloader(images_tensor, labels_tensor, batch_size, shuffle=True):
    """Create a DataLoader for the vision dataset."""
    dataset = VisionDataset(images_tensor, labels_tensor)
    sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )


def make_val_dataloader(images_tensor, labels_tensor, batch_size):
    """Create a validation DataLoader (no shuffling)."""
    # Images are already in (N, C, H, W) format from create_val_dataset
    return make_dataloader(images_tensor, labels_tensor, batch_size, shuffle=False)


@torch.no_grad()
def evaluate_accuracy(model, val_loader, device="cuda"):
    """
    Compute top-1 accuracy on validation set.
    
    Args:
        model: Vision Transformer model in eval mode
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
    
    Returns:
        Top-1 accuracy in percentage points [0, 100].
    """
    accuracy, _, _ = evaluate_accuracy_with_counts(model, val_loader, device)
    return accuracy


@torch.no_grad()
def evaluate_accuracy_with_counts(model, val_loader, device="cuda"):
    """
    Compute top-1 accuracy and raw counts on the validation set.

    Returns:
        Tuple of (accuracy_pct, correct, total), where accuracy_pct is in [0, 100].
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
        total += labels.size(0)  # samples
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, correct, total


def get_num_classes():
    """Return the number of classification classes."""
    return NUM_CLASSES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare vision data for autoresearch")
    parser.add_argument("--verify", action="store_true", help="Just verify data exists, don't create cache")
    args = parser.parse_args()
    
    print(f"Cache directory: {VISION_CACHE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print()
    
    # Check if required files exist
    if not DATA_DIR.exists():
        print("Error: data/tiny-imagenet-200 directory not found!")
        print("Please download TinyImageNet dataset and place it in the data folder.")
        sys.exit(1)
    
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    
    if args.verify:
        if train_dir.exists():
            print(f"Train directory exists with {len(list(train_dir.glob('*')))} class folders")
        else:
            print("Warning: Train directory not found")
        
        if (DATA_DIR / "wnids.txt").exists():
            with open(DATA_DIR / "wnids.txt", "r") as f:
                num_classes = len([l for l in f if l.strip()])
            print(f"Number of classes: {num_classes}")
        else:
            print("Warning: wnids.txt not found")
        
        if val_dir.exists():
            print("Validation directory exists")
        else:
            print("Warning: Validation directory not found")
    else:
        # Create cached dataset
        train_images, train_labels = create_cached_dataset()
        print(f"Training data shape: {train_images.shape}, labels shape: {train_labels.shape}")
        
        val_images, val_labels = create_val_dataset()
        print(f"Validation data shape: {val_images.shape}, labels shape: {val_labels.shape}")
    
    print()
    print("Done! Ready to train.")