# data/bpe_tokenizer.py
"""
data/bpe_tokenizer.py — Byte-level BPE (Byte Pair Encoding) tokenizer gaya
GPT-2, ditulis pure Python tanpa dependensi eksternal.

Karena beroperasi di level byte (bukan karakter Unicode), tokenizer ini
selalu bisa merepresentasikan teks apa pun — termasuk teks Indonesia dengan
huruf berimbuhan/diakritik — tanpa pernah butuh token <unk>.

Dibanding CharTokenizer, BPE belajar potongan sub-kata yang sering muncul
(mis. "meng", "nya", "kan") sebagai satu token, jadi satu kata butuh lebih
sedikit token dan model butuh lebih sedikit data untuk mulai membentuk
pola yang masuk akal.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm


class BPETokenizer:
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}   # (id1, id2) -> id gabungan, urut training
        self.vocab: Dict[int, bytes] = {}               # id -> potongan byte
        self.special_to_id: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Properti
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<eos>"]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, text: str, vocab_size: int = 1024, verbose: bool = True) -> None:
        """Latih BPE dari teks training.

        `vocab_size` = target ukuran vocab akhir (termasuk token spesial +
        256 byte dasar). Kalau corpus terlalu kecil/tidak cukup berulang
        untuk mencapai target itu, training berhenti lebih awal dan
        vocab akhir akan lebih kecil dari target (bukan error).
        """
        assert vocab_size > 256 + len(self.SPECIAL_TOKENS), (
            "vocab_size terlalu kecil, minimal lebih dari 256 + jumlah token spesial"
        )

        self.special_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        offset = len(self.SPECIAL_TOKENS)
        self.vocab = {i: tok.encode("utf-8") for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for b in range(256):
            self.vocab[offset + b] = bytes([b])

        ids = [b + offset for b in text.encode("utf-8")]

        num_merges = vocab_size - len(self.vocab)
        self.merges = {}

        iterator = tqdm(range(num_merges), desc="Training BPE", disable=not verbose)
        for _ in iterator:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                if verbose:
                    tqdm.write(
                        "Berhenti lebih awal: tidak ada lagi pasangan token yang "
                        f"berulang di corpus (vocab akhir: {self.vocab_size})"
                    )
                break

            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_id
            ids = self._merge(ids, best_pair, new_id)

        if verbose:
            tqdm.write(f"BPE selesai: {len(self.merges)} merge, vocab_size={self.vocab_size}")

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Ganti semua kemunculan `pair` yang berurutan di `ids` dengan `new_id`."""
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        offset = len(self.SPECIAL_TOKENS)
        ids = [b + offset for b in text.encode("utf-8")]

        # Terapkan merge sesuai urutan training (merge yang paling awal
        # dipelajari, diterapkan paling dulu), sampai tidak ada lagi
        # pasangan yang cocok dengan hasil merge apapun.
        while len(ids) >= 2:
            pairs_present = set(zip(ids, ids[1:]))
            candidates = pairs_present & self.merges.keys()
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, best_pair, self.merges[best_pair])

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        raw = b"".join(self.vocab[i] for i in ids if i not in skip and i in self.vocab)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Simpan / muat
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "special_to_id": self.special_to_id,
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "vocab": {str(i): list(b) for i, b in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.special_to_id = data["special_to_id"]
        tok.merges = {tuple(pair): new_id for pair, new_id in data["merges"]}
        tok.vocab = {int(i): bytes(b) for i, b in data["vocab"].items()}
        return tok