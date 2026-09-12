"""
training/optimizer.py — Optimizer AdamW dengan weight-decay grouping, plus
learning rate scheduler warmup linear diikuti cosine decay (pola umum untuk
melatih transformer).
"""

import math
import torch


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    """AdamW dengan weight decay hanya untuk parameter berdimensi >= 2
    (Linear/Embedding); bias dan LayerNorm tidak di-decay."""
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
):
    """Warmup linear sampai `max_lr`, lalu cosine decay turun ke `min_lr`."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step >= max_steps:
            return min_lr / max_lr
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        scaled_min = min_lr / max_lr
        return scaled_min + (1 - scaled_min) * coeff

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
