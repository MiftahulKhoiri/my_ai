"""
training/checkpoint.py — Simpan & muat checkpoint (state model, state
optimizer, step, best_val_loss, dan total_steps) supaya training bisa
dilanjutkan atau model bisa dipakai langsung untuk inference.
"""

import glob
import os
import re
import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    total_steps: int = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "total_steps": total_steps,
        },
        path,
    )


def rotate_checkpoints(checkpoint_dir: str, keep_last_n, pattern: str = "step_*.pt") -> list[str]:
    """Hapus checkpoint periodik ("step_<n>.pt") paling lama, sisakan cuma
    `keep_last_n` yang terbaru (diurutkan dari nomor step di nama file, bukan
    mtime). Tidak pernah menyentuh best.pt/final.pt — itu checkpoint khusus
    di luar pola rotasi ini.

    Tanpa ini, step_<n>.pt menumpuk terus tiap `checkpoint_every`, bisa lama-
    lama memenuhi storage kartu SD di training yang panjang (relevan buat
    Raspberry Pi). `keep_last_n` None atau <=0 menonaktifkan rotasi (semua
    checkpoint periodik dibiarkan, perilaku lama).

    Return list path yang dihapus (buat logging/testing).
    """
    if keep_last_n is None or keep_last_n <= 0:
        return []

    def _step_of(path: str) -> int:
        m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, pattern)),
        key=_step_of,
    )

    removed = []
    for old_ckpt in checkpoints[:-keep_last_n]:
        try:
            os.remove(old_ckpt)
            removed.append(old_ckpt)
        except OSError as e:
            print(f"[peringatan] gagal hapus checkpoint lama {old_ckpt!r}: {e}")
    return removed


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
        "total_steps": ckpt.get("total_steps"),  # None kalau checkpoint lama (sebelum fitur ini)
    }