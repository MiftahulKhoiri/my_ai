# data/tokenizer.py
"""
data/tokenizer.py — Tokenizer level karakter, sederhana dan tanpa dependensi
eksternal (tidak butuh sentencepiece/tokenizers). Cocok untuk dataset
kecil-menengah dan environment terbatas (Raspberry Pi, Termux).

Vocab dibangun langsung dari karakter unik yang muncul di corpus training,
ditambah 4 token khusus: <pad>, <bos>, <eos>, <unk>.
"""

import json
import os
from typing import List


class CharTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def fit(self, text: str) -> None:
        """Bangun vocab dari karakter unik pada teks training."""
        chars = sorted(set(text))
        vocab = list(self.SPECIAL_TOKENS) + chars
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.itos.get(i, "") for i in ids if i not in skip)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            tok.stoi = json.load(f)
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        return tok