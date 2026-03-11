"""
Autoresearch vision pretraining script. Single-GPU, single-file.
ResNet-style ConvNet for faster convergence on images.

Usage: uv run train.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import NUM_CLASSES, TIME_BUDGET, make_dataloader, evaluate_accuracy_with_counts

# ---------------------------------------------------------------------------
# ResNet-style ConvNet Model
# ---------------------------------------------------------------------------

@dataclass
class ResNetConfig:
    image_size: int = 64
    num_classes: int = 200
    base_channels: int = 64
    block_depth: int = 3      # Number of residual blocks per stage
    width_multiplier: int = 1 # Width multiplier for the whole network


def norm(x):
    """RMSNorm for consistency."""
    return F.rms_norm(x, (x.size(-1),))


class ResidualBlock(nn.Module):
    """Residual block with 3x3 convolutions."""
    def __init__(self, in_channels, out_channels, stride=1, device="cuda"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False, device=device)

        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                                  bias=False, device=device)

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = F.relu(out)
        return out


class Stage(nn.Module):
    """A stage with multiple residual blocks."""
    def __init__(self, in_channels, out_channels, num_blocks, stride=2, device="cuda"):
        super().__init__()
        blocks = []
        # First block with stride
        blocks.append(ResidualBlock(in_channels, out_channels, stride=stride, device=device))
        # Remaining blocks with stride=1
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1, device=device))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    """Simple ResNet-style ConvNet."""
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config

        # Stages
        channels = [config.base_channels * (2**i) * config.width_multiplier
                    for i in range(4)]

        # Initial conv layer - outputs channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1,
                               padding=1, bias=False, device=device)

        # conv1 outputs channels[0], so stage1 takes channels[0] as input
        self.stage1 = Stage(channels[0], channels[0], config.block_depth, stride=1, device=device)
        self.stage2 = Stage(channels[0], channels[1], config.block_depth, stride=2, device=device)
        self.stage3 = Stage(channels[1], channels[2], config.block_depth, stride=2, device=device)
        self.stage4 = Stage(channels[2], channels[3], config.block_depth, stride=2, device=device)

        # Global average pooling + classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels[3], config.num_classes, bias=False, device=device)

    def init_weights(self):
        """Initialize weights with He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.conv1(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(1)
        logits = self.head(x)
        return logits

    def num_scaling_params(self):
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}

    def estimate_flops(self):
        """Estimate FLOPs."""
        # Rough estimate: 2 * params * activations
        nparams = sum(p.numel() for p in self.parameters())
        # For 64x64 images, roughly 2-3 MACs per param per pixel
        return nparams * 64 * 64 * 2

    def setup_optimizer(self, lr=0.1, weight_decay=0.0001, momentum=0.9):
        """Setup SGD optimizer with momentum."""
        param_groups = [
            dict(kind='adamw', params=list(self.parameters()), lr=lr,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        ]
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only) - same as train.py
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=True, fullgraph=False)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=True, fullgraph=False)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        # Replace missing gradients with zeros to allow stacking
        stacked_grads = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group['weight_decay'])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# ResNet-style ConvNet architecture
BLOCK_DEPTH = 3         # Number of residual blocks per stage
BASE_CHANNELS = 64      # Base channel count
WIDTH_MULTIPLIER = 2    # Width multiplier (1=baseline, 2=2x wider)

# Optimization - ConvNet trains with SGD-style optimization
DEVICE_BATCH_SIZE = 128    # Per-device batch size
TOTAL_BATCH_SIZE = 256     # Total batch size (with grad accum)
LEARNING_RATE = 0.1        # Initial learning rate (high for fast training)
WEIGHT_DECAY = 1e-4        # Standard weight decay
MOMENTUM = 0.9             # SGD momentum

# EMA (Exponential Moving Average) for weights - helps final accuracy
USE_EMA = True
EMA_DECAY = 0.995          # EMA decay factor

# LR schedule
WARMUP_RATIO = 0.1         # 10% warmup
WARMDOWN_RATIO = 0.5       # 50% warmdown with cosine
FINAL_LR_FRAC = 0.0        # Final LR as fraction of initial

# Safety thresholds
LOSS_EXPLOSION_THRESHOLD = 1e6  # if training loss exceeds this, issue a warning

# Training termination (set NUM_EPOCHS>0 to use epoch-based stopping)
NUM_EPOCHS = 5


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
from prepare import create_cached_dataset, make_dataloader, get_num_classes

train_images, train_labels = create_cached_dataset()
val_images, val_labels = None, None  # Will load later for evaluation

# Build ResNet-style model
config = ResNetConfig(
    image_size=64,
    num_classes=get_num_classes(),
    base_channels=BASE_CHANNELS,
    block_depth=BLOCK_DEPTH,
    width_multiplier=WIDTH_MULTIPLIER,
)
print(f"ResNet config: {asdict(config)}")

# Create model
model = ResNet(config, device=device)
model.init_weights()

# Create EMA model
if USE_EMA:
    import copy
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    print("Using EMA with decay =", EMA_DECAY)
else:
    ema_model = None

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per forward pass: {num_flops_per_token:e}")

# Gradient accumulation steps
assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

optimizer = model.setup_optimizer(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

# model = torch.compile(model, dynamic=False)

# Create dataloaders directly to avoid multiprocessing issues
# Ensure float32 precision for images
train_images = train_images.float()
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=DEVICE_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
train_loader_iter = iter(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)
# Cosine annealing with warmup

def get_lr_multiplier(progress):
    import math
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    else:
        # Cosine decay from 1.0 to 0.0
        return 0.5 * (1 + math.cos(math.pi * (progress - WARMUP_RATIO) / (1 - WARMUP_RATIO)))


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
        # Move images and labels to the training device with correct dtype
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device)

        with autocast_ctx:
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

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    optimizer.step()

    # Update EMA weights after optimizer step
    if ema_model is not None:
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(EMA_DECAY).add_(param, alpha=1 - EMA_DECAY)

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
    samples_per_sec = int(DEVICE_BATCH_SIZE * grad_accum_steps / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.4f} | lr: {lrm:.2f}x | dt: {dt*1000:.0f}ms | samples/s: {samples_per_sec:,} | remaining: {remaining:.0f}s    ", end="", flush=True)

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

total_samples = step * DEVICE_BATCH_SIZE * grad_accum_steps


# Final eval - load validation data
if val_images is None or val_labels is None:
    from prepare import create_val_dataset, make_val_dataloader
    val_images, val_labels = create_val_dataset()

val_loader = make_val_dataloader(val_images, val_labels, DEVICE_BATCH_SIZE)

# Evaluate with EMA model if available, otherwise use regular model
eval_model = ema_model if ema_model is not None else model
eval_model.eval()
with autocast_ctx:
    val_accuracy, val_correct, val_total = evaluate_accuracy_with_counts(eval_model, val_loader, device)


# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")  # decimal in [0, 1]
print(f"val_correct:      {val_correct}")
print(f"val_total:        {val_total}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_samples_M:  {total_samples / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"block_depth:      {BLOCK_DEPTH}")
print(f"width_multiplier: {WIDTH_MULTIPLIER}")