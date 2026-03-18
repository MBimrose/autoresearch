"""
Ensemble of 3 ConvNeXt-Large models trained with different seeds.
Each model votes on the final prediction.

Configuration:
- 3 models, different random seeds (42, 123, 456)
- Average probabilities at inference
- Time budget: 3600 seconds (60 minutes)

Usage: CUDA_VISIBLE_DEVICES=4 uv run train_ensemble3.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

from prepare import NUM_CLASSES, TIME_BUDGET, make_dataloader, evaluate_accuracy_with_counts, create_cached_dataset, create_val_dataset, make_val_dataloader


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.03
ADAM_BETAS = (0.9, 0.999)
SEEDS = [42, 123, 456]


class EnsembleModel(nn.Module):
    """Ensemble of 3 ConvNeXt-Large models."""
    def __init__(self, num_classes, seed_idx=0):
        super().__init__()
        self.seed_idx = seed_idx

    def forward(self, x):
        pass


def create_model(num_classes, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = convnext_large(weights=weights)
    num_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(num_features, num_classes)
    )
    return model


def main():
    t_start = time.time()
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    print("Loading datasets...")
    train_images, train_labels, class_names = create_cached_dataset()
    val_images, val_labels, _ = create_val_dataset()
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Create 3 models with different seeds
    print("Creating ensemble of 3 ConvNeXt-Large models...")
    models = []
    optimizers = []
    schedulers = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = create_model(NUM_CLASSES, seed)
        model = model.to(device)
        models.append(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    total_params = sum(p.numel() for p in models[0].parameters()) * len(models)
    print(f"Parameters per model: {total_params / len(SEEDS):,}, Total: {total_params:,}")

    train_loader = make_dataloader(train_images, train_labels, BATCH_SIZE, shuffle=True)
    val_loader = make_val_dataloader(val_images, val_labels, BATCH_SIZE)

    total_training_time = 0
    step = 0
    best_val_acc = 0.0

    while total_training_time < TIME_BUDGET:
        torch.cuda.synchronize()
        t0 = time.time()

        for m in models:
            m.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Train each model
            for i, model in enumerate(models):
                with autocast_ctx:
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)

                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

            epoch_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            step += 1

        for i in range(len(SEEDS)):
            schedulers[i].step()
        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total

        torch.cuda.synchronize()
        total_training_time += time.time() - t0

        print(f"step {step:05d} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | time: {total_training_time:.0f}s", flush=True)

        # Validation with ensemble voting
        if step % 50 == 0 or total_training_time >= TIME_BUDGET * 0.9:
            for m in models:
                m.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    # Average probabilities from all models
                    total_prob = None
                    for model in models:
                        with autocast_ctx:
                            logits = model(images)
                            prob = F.softmax(logits, dim=-1)
                            if total_prob is None:
                                total_prob = prob
                            else:
                                total_prob += prob
                    avg_prob = total_prob / len(models)
                    preds = avg_prob.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total
            print(f"  Val acc (ensemble): {val_acc:.4f} ({val_correct}/{val_total})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        if total_training_time >= TIME_BUDGET:
            break

    # Final eval with ensemble
    for m in models:
        m.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            total_prob = None
            for model in models:
                with autocast_ctx:
                    logits = model(images)
                    prob = F.softmax(logits, dim=-1)
                    if total_prob is None:
                        total_prob = prob
                    else:
                        total_prob += prob
            avg_prob = total_prob / len(models)
            preds = avg_prob.argmax(dim=-1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    final_val_acc = val_correct / val_total

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_accuracy:     {final_val_acc:.6f}")
    print(f"val_correct:      {val_correct}")
    print(f"val_total:        {val_total}")
    print(f"best_val_acc:     {best_val_acc:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {total_params / 1e6:.1f}")

if __name__ == "__main__":
    main()
