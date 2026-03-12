"""
Ensemble evaluation with test-time augmentation.
Loads multiple model checkpoints and averages predictions.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from prepare import create_val_dataset, make_val_dataloader, NUM_CLASSES

DEVICE = torch.device("cuda")

# ---------------------------------------------------------------------------
# Model definition (copied from train.py to avoid importing training code)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet basic block with skip connection."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, device="cuda"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False, device=device),
                nn.BatchNorm2d(out_channels, device=device)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for image classification."""

    def __init__(self, block, num_blocks, num_classes=200, base_channels=64, device="cuda"):
        super().__init__()
        self.in_channels = base_channels
        self.device = device

        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(base_channels, device=device)

        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1, device=device)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2, device=device)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2, device=device)
        self.layer4 = self._make_layer(block, base_channels * 8, num_blocks[3], stride=2, device=device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes, bias=False, device=device)

    def _make_layer(self, block, channels, num_blocks, stride, device):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride, device))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESNET_BLOCKS = [2, 2, 2, 2]  # ResNet-18
BASE_CHANNELS = 64

# All 28 models for maximum ensemble diversity
MODEL_FILES = [
    # Original 4
    "model_seed42.pt",
    "model_seed123.pt",
    "model_seed456.pt",
    "model_seed789.pt",
    # Second batch 8
    "model_seed100.pt",
    "model_seed200.pt",
    "model_seed300.pt",
    "model_seed400.pt",
    "model_seed500.pt",
    "model_seed600.pt",
    "model_seed700.pt",
    "model_seed800.pt",
    # Third batch 16
    "model_seed1000.pt",
    "model_seed1001.pt",
    "model_seed1002.pt",
    "model_seed1003.pt",
    "model_seed1004.pt",
    "model_seed1005.pt",
    "model_seed1006.pt",
    "model_seed1007.pt",
    "model_seed1008.pt",
    "model_seed1009.pt",
    "model_seed1010.pt",
    "model_seed1011.pt",
    "model_seed1012.pt",
    "model_seed1013.pt",
    "model_seed1014.pt",
    "model_seed1015.pt",
]
USE_TTA = True  # Test-time augmentation (horizontal flip)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load a model from checkpoint."""
    model = ResNet(BasicBlock, RESNET_BLOCKS, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS, device=DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def tta_predict(model, images, device):
    """Predict with test-time augmentation (horizontal flip)."""
    # Original predictions
    logits_orig = model(images.to(device))
    probs_orig = F.softmax(logits_orig, dim=1)

    # Flipped predictions
    flipped = images.flip(dims=[-1])  # Horizontal flip
    logits_flip = model(flipped.to(device))
    probs_flip = F.softmax(logits_flip, dim=1)

    # Average predictions
    return (probs_orig + probs_flip) / 2

def evaluate_ensemble(models, val_loader, device, use_tta=False):
    """Evaluate ensemble of models with optional TTA."""
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions from all models
            all_probs = []
            for model in models:
                if use_tta:
                    probs = tta_predict(model, images, device)
                else:
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

            # Average ensemble predictions
            ensemble_probs = torch.stack(all_probs).mean(dim=0)
            preds = ensemble_probs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy, correct, total

def main():
    print(f"Loading {len(MODEL_FILES)} models...")
    models = []
    for mf in MODEL_FILES:
        if os.path.exists(mf):
            print(f"  Loading {mf}...")
            models.append(load_model(mf))
        else:
            print(f"  WARNING: {mf} not found, skipping")

    if not models:
        print("No models loaded!")
        return

    print(f"Loaded {len(models)} models")
    print(f"Using TTA: {USE_TTA}")

    # Load validation data
    print("Loading validation data...")
    val_images, val_labels = create_val_dataset()
    val_images = val_images.float()
    val_labels = val_labels.long()
    val_loader = make_val_dataloader(val_images, val_labels, batch_size=32)

    # Evaluate
    print("Evaluating...")
    accuracy, correct, total = evaluate_ensemble(models, val_loader, DEVICE, use_tta=USE_TTA)

    print("---")
    print(f"Ensemble size:    {len(models)}")
    print(f"TTA enabled:      {USE_TTA}")
    print(f"val_accuracy:     {accuracy:.6f}")
    print(f"val_correct:      {correct}")
    print(f"val_total:        {total}")

if __name__ == "__main__":
    main()
