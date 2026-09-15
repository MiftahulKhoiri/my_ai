"""
data/dataset.py — Dataset PyTorch untuk language modeling.

Membaca file teks, meng-encode dengan tokenizer, lalu menyediakan potongan
(block) berurutan sepanjang `block_size` untuk training next-token
prediction: target adalah input yang digeser satu posisi ke kanan.
"""

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

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
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Path data tidak ditemukan: {path!r}. "
            "Cek path di config.py sudah benar."
        )

    # Jika path adalah satu file .txt
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    # Jika path adalah folder, ambil semua .txt
    if p.is_dir():
        txt_files = sorted(p.rglob("*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"Tidak ada file .txt di dalam folder: {path!r}"
            )

        texts = []

        for txt_file in txt_files:
            print(f"Loading: {txt_file}")

            with open(txt_file, "r", encoding="utf-8") as f:
                texts.append(f.read())

        return "\n\n".join(texts)

    raise ValueError(
        f"Path bukan file atau folder yang valid: {path!r}"
    )


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