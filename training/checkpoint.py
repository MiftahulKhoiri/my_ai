"""
training/checkpoint.py — Simpan & muat checkpoint (state model, state
optimizer, step, dan best_val_loss) supaya training bisa dilanjutkan atau
model bisa dipakai langsung untuk inference.
"""

import os
import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    map_location: str = "cpu",
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path!r}. Pastikan sudah training "
            "dulu (python train.py) dan path checkpoint-nya benar "
            "(mis. checkpoints/best.pt atau checkpoints/final.pt)."
        )
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return {
        "step": ckpt.get("step", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
    }