"""
data/dataset.py — Dataset PyTorch untuk language modeling.

Membaca file teks, meng-encode dengan tokenizer, lalu menyediakan potongan
(block) berurutan sepanjang `block_size` untuk training next-token
prediction: target adalah input yang digeser satu posisi ke kanan.
"""

import os
import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        self.block_size = block_size
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        # -1 karena tiap sampel butuh block_size+1 token untuk (input, target)
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def make_train_val_datasets(
    tokenizer: CharTokenizer,
    train_path: str,
    val_path: str,
    block_size: int,
    val_split: float = 0.1,
):
    """Bangun dataset train & val. Kalau val_path tidak ditemukan, otomatis
    split dari train_path (bagian akhir teks dipakai sebagai validasi)."""
    train_text = load_text(train_path)

    if val_path and os.path.exists(val_path):
        val_text = load_text(val_path)
    else:
        split_idx = int(len(train_text) * (1 - val_split))
        val_text = train_text[split_idx:]
        train_text = train_text[:split_idx]

    train_ds = TextDataset(train_text, tokenizer, block_size)
    val_ds = TextDataset(val_text, tokenizer, block_size)
    return train_ds, val_ds
