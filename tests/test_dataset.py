"""
tests/test_dataset.py — Tes TextDataset. Butuh torch terpasang untuk jalan
(otomatis di-skip kalau tidak ada, mis. saat dites tanpa environment penuh).

Jalankan dari root project:
    python -m unittest discover tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data.tokenizer import CharTokenizer

if TORCH_AVAILABLE:
    from data.dataset import TextDataset, encode_corpus


@unittest.skipUnless(TORCH_AVAILABLE, "butuh torch terpasang")
class TestTextDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer()
        self.text = "abcdefghijklmnopqrstuvwxyz" * 3
        self.tokenizer.fit(self.text)

    def test_length(self):
        block_size = 8
        ds = TextDataset(self.text, self.tokenizer, block_size)
        # encode_corpus, bukan tokenizer.encode() polos: teks tanpa "\n\n"
        # tetap dapat 1 <eos> di akhir (lihat test_eos_insertion di bawah).
        expected = max(0, len(encode_corpus(self.text, self.tokenizer)) - block_size - 1)
        self.assertEqual(len(ds), expected)

    def test_shift_by_one(self):
        block_size = 8
        ds = TextDataset(self.text, self.tokenizer, block_size)
        x, y = ds[0]
        self.assertEqual(x.shape[0], block_size)
        self.assertEqual(y.shape[0], block_size)
        # y harus persis x digeser 1 posisi (target next-token prediction)
        self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_empty_when_text_too_short(self):
        short_text = "abc"
        tok = CharTokenizer()
        tok.fit(short_text)
        ds = TextDataset(short_text, tok, block_size=32)
        self.assertEqual(len(ds), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "butuh torch terpasang")
class TestEncodeCorpus(unittest.TestCase):
    """Tes khusus buat penyisipan <eos> per giliran (format USER:/ASSISTANT:
    dipisah baris kosong), supaya model belajar kapan harus berhenti dan
    generate() bisa stop otomatis di <eos>."""

    def setUp(self):
        self.tokenizer = CharTokenizer()
        self.corpus = (
            "USER: Apa itu ML?\nASSISTANT: Mesin belajar pola.\n\n"
            "USER: Halo\nASSISTANT: Hai juga!"
        )
        self.tokenizer.fit(self.corpus)

    def test_eos_inserted_between_turns(self):
        ids = encode_corpus(self.corpus, self.tokenizer)
        eos_positions = [i for i, tid in enumerate(ids) if tid == self.tokenizer.eos_id]
        # Dua giliran (dipisah satu baris kosong) -> tepat 2 <eos>.
        self.assertEqual(len(eos_positions), 2)
        # <eos> terakhir harus di posisi paling akhir (giliran kedua juga
        # ditutup <eos>, bukan cuma yang di tengah).
        self.assertEqual(eos_positions[-1], len(ids) - 1)

    def test_turn_content_roundtrips_around_eos(self):
        ids = encode_corpus(self.corpus, self.tokenizer)
        segments, current = [], []
        for tid in ids:
            if tid == self.tokenizer.eos_id:
                segments.append(self.tokenizer.decode(current))
                current = []
            else:
                current.append(tid)
        self.assertEqual(
            segments,
            ["USER: Apa itu ML?\nASSISTANT: Mesin belajar pola.", "USER: Halo\nASSISTANT: Hai juga!"],
        )

    def test_appends_trailing_eos_even_without_blank_line(self):
        # Teks tanpa "\n\n" sama sekali tetap harus dapat 1 <eos> di akhir,
        # supaya setiap file training tetap punya sinyal "berhenti di sini".
        plain = "teks tanpa baris kosong sama sekali"
        tok = CharTokenizer()
        tok.fit(plain)
        ids = encode_corpus(plain, tok)
        self.assertEqual(ids[-1], tok.eos_id)
        self.assertEqual(sum(1 for tid in ids if tid == tok.eos_id), 1)


if __name__ == "__main__":
    unittest.main()