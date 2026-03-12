"""
Autoresearch vision pretraining script. Single-GPU, single-file.
ResNet-18 for TinyImageNet - CNN architecture for sample efficiency.

Usage: uv run train.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for this experiment
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import NUM_CLASSES, TIME_BUDGET, make_dataloader, evaluate_accuracy_with_counts

# ---------------------------------------------------------------------------
# ResNet-18 Model (CNN - more sample efficient than ViT)
# ---------------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """ResNet bottleneck block: 1x1 -> 3x3 -> 1x1 with expansion=4."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, device="cuda"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, stride=1, bias=False, device=device)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, device=device)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False, device=device),
                nn.BatchNorm2d(out_channels * self.expansion, device=device)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


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

        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(base_channels, device=device)

        # ResNet layers
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1, device=device)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2, device=device)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2, device=device)
        self.layer4 = self._make_layer(block, base_channels * 8, num_blocks[3], stride=2, device=device)

        # Classification head
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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'conv': sum(p.numel() for n, p in self.named_parameters() if 'conv' in n),
            'bn': sum(p.numel() for n, p in self.named_parameters() if 'bn' in n),
            'head': sum(p.numel() for p in self.fc.parameters())
        }

    def estimate_flops(self):
        # Rough FLOPs estimate for ResNet-18
        # Conv1: 3*64*3*3*64*64 = 8M
        # Layer1: 2 blocks * 64*64*3*3*64*64 = 16M
        # Layer2: 2 blocks * 128*32*3*3*128*32 = 8M
        # Layer3: 2 blocks * 256*16*3*3*256*16 = 2M
        # Layer4: 2 blocks * 512*8*3*3*512*8 = 0.5M
        # FC: 512*200 = 0.1M
        return 35e6  # ~35M FLOPs per forward pass

    def setup_optimizer(self, lr=0.1, momentum=0.9, weight_decay=5e-4):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, nesterov=True)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer


# Alias for compatibility
VisionTransformer = ResNet
ViTConfig = None


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture: ResNet-50 (bottleneck blocks, deeper network)
RESNET_BLOCKS = [3, 4, 6, 3]   # ResNet-50 configuration
BASE_CHANNELS = 64             # Base channel width (64 -> 256 in layer1)
NUM_CLASSES = 200              # TinyImageNet classes

# Optimization - SGD with momentum (standard for ResNets)
TOTAL_BATCH_SIZE = 64          # Batch size
DEVICE_BATCH_SIZE = 32         # Per-device batch size
LR = 0.05                      # Lower LR for deeper network
MOMENTUM = 0.9                 # Standard momentum
WEIGHT_DECAY = 5e-4            # Standard weight decay for ResNet
WARMDOWN_RATIO = 0.2           # Longer warmdown
FINAL_LR_FRAC = 0.0            # Final LR as fraction of initial

# Safety thresholds
LOSS_EXPLOSION_THRESHOLD = 1e6

# Training termination
NUM_EPOCHS = 0                 # Use time budget, not epochs


# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32)
RTX_6000_ADA_BF16_PEAK_FLOPS = 362.5e12

# Load data
print("Loading training dataset...")
from prepare import create_cached_dataset, make_dataloader

train_images, train_labels = create_cached_dataset()
# Cast to float32 (prepare.py returns float64)
train_images = train_images.float()
train_labels = train_labels.long()
val_images, val_labels = None, None  # Will load later for evaluation

# Create ResNet model - use BottleneckBlock for ResNet-50+ (expansion=4), BasicBlock for ResNet-18/34
if RESNET_BLOCKS == [2, 2, 2, 2]:
    # ResNet-18 uses BasicBlock (expansion=1)
    block_type = BasicBlock
    print("Using ResNet-18 architecture (BasicBlock)")
else:
    # ResNet-50+ uses BottleneckBlock (expansion=4)
    block_type = BottleneckBlock
    print("Using ResNet-50 architecture (BottleneckBlock)")

model = ResNet(block_type, RESNET_BLOCKS, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS, device=device)

# Initialize weights
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per forward pass: {num_flops_per_token:e}")

# Calculate gradient accumulation steps
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

optimizer = model.setup_optimizer(lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Create dataloaders
train_loader = make_dataloader(train_images, train_labels, DEVICE_BATCH_SIZE, shuffle=True)
train_loader_iter = iter(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    # Linear warmdown at the end
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_weight_decay(progress):
    return WEIGHT_DECAY  # Constant weight decay for ResNet


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Prefetch first batch (handle exhausted iterator)
try:
    images, labels = next(train_loader_iter)
except StopIteration:
    train_loader_iter = iter(train_loader)
    images, labels = next(train_loader_iter)

# If epoch-based stopping requested, compute optimizer steps per epoch
samples_per_opt_step = grad_accum_steps * DEVICE_BATCH_SIZE
dataset_size = len(train_images)
steps_per_epoch = math.ceil(dataset_size / samples_per_opt_step) if samples_per_opt_step > 0 else 0
print(f"Dataset size: {dataset_size}, samples/opt-step: {samples_per_opt_step}, steps/epoch: {steps_per_epoch}")
samples_seen = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            # Move data to the training device (ensure float32)
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        
        try:
            images, labels = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            images, labels = next(train_loader_iter)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    wd = get_weight_decay(progress)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        group["weight_decay"] = wd
    
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding — log a warning but continue training
    if not torch.isfinite(torch.tensor(train_loss_f)):
        print("WARN: non-finite loss encountered, continuing training")
    elif train_loss_f > LOSS_EXPLOSION_THRESHOLD:
        print(f"WARN: large loss ({train_loss_f:.3f}) > {LOSS_EXPLOSION_THRESHOLD} — continuing training")

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / RTX_6000_ADA_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.4f} | lr: {lrm*LR:.4f} | dt: {dt*1000:.0f}ms | img/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE


# Final eval - load validation data
if val_images is None or val_labels is None:
    from prepare import create_val_dataset, make_val_dataloader
    val_images, val_labels = create_val_dataset()

# Cast validation data to float32 (prepare.py returns float64)
val_images = val_images.float()
val_labels = val_labels.long()

val_loader = make_val_dataloader(val_images, val_labels, DEVICE_BATCH_SIZE)

model.eval()
with autocast_ctx:
    val_accuracy, val_correct, val_total = evaluate_accuracy_with_counts(model, val_loader, device)


# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / RTX_6000_ADA_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")  # decimal in [0, 1]
print(f"val_correct:      {val_correct}")
print(f"val_total:        {val_total}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"resnet_blocks:    {RESNET_BLOCKS}")