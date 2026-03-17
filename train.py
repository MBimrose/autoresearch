"""
ConvNeXt-Large with light augmentations - targeting 90%+ accuracy.
Using AdamW optimizer with pretrained ImageNet weights.

Key insight: Light augmentations (flips, small color jitter) provide regularization
without distorting the image too much.

Configuration:
- Batch size: 8
- Optimizer: AdamW with LR=0.0001
- Augmentations: Horizontal flip (50%), small color jitter
- Time budget: 3600 seconds (60 minutes)

Usage: CUDA_VISIBLE_DEVICES=4 uv run train.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.transforms import ColorJitter

from prepare import NUM_CLASSES, TIME_BUDGET, make_dataloader, evaluate_accuracy_with_counts, create_cached_dataset, create_val_dataset, make_val_dataloader, VisionDataset


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.03
ADAM_BETAS = (0.9, 0.999)

# Light color jitter for regularization
color_jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)


def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    print("Loading datasets...")
    train_images, train_labels, class_names = create_cached_dataset()
    val_images, val_labels, _ = create_val_dataset()
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Model with pretrained weights
    print("Loading ConvNeXt-Large...")
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = convnext_large(weights=weights)

    # Replace classifier
    num_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(num_features, NUM_CLASSES)
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Custom training with augmentation
    train_dataset = VisionDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = make_val_dataloader(val_images, val_labels, BATCH_SIZE)

    total_training_time = 0
    step = 0
    best_val_acc = 0.0

    while total_training_time < TIME_BUDGET:
        torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_loader:
            # Apply light augmentations
            augmented_images = []
            for img in images:
                # Horizontal flip (50% chance)
                if torch.rand(0).item() > 0.5:
                    img = img.flip(dims=[-1])
                # Small color jitter
                img = color_jitter(img)
                augmented_images.append(img)

            images = torch.stack(augmented_images).to(device)
            labels = labels.to(device)

            with autocast_ctx:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            step += 1

        scheduler.step()
        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total

        torch.cuda.synchronize()
        total_training_time += time.time() - t0

        print(f"step {step:05d} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | time: {total_training_time:.0f}s", flush=True)

        # Validation
        if step % 50 == 0 or total_training_time >= TIME_BUDGET * 0.9:
            model.eval()
            with autocast_ctx:
                val_acc, val_correct, val_total = evaluate_accuracy_with_counts(model, val_loader, device)
            print(f"  Val acc: {val_acc:.4f} ({val_correct}/{val_total})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        if total_training_time >= TIME_BUDGET:
            break

    # Final eval
    model.eval()
    with autocast_ctx:
        final_val_acc, final_val_correct, final_val_total = evaluate_accuracy_with_counts(model, val_loader, device)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_accuracy:     {final_val_acc:.6f}")
    print(f"val_correct:      {final_val_correct}")
    print(f"val_total:        {final_val_total}")
    print(f"best_val_acc:     {best_val_acc:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")

if __name__ == "__main__":
    main()
