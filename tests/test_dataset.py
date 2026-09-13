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
    from data.dataset import TextDataset


@unittest.skipUnless(TORCH_AVAILABLE, "butuh torch terpasang")
class TestTextDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer()
        self.text = "abcdefghijklmnopqrstuvwxyz" * 3
        self.tokenizer.fit(self.text)

    def test_length(self):
        block_size = 8
        ds = TextDataset(self.text, self.tokenizer, block_size)
        expected = max(0, len(self.tokenizer.encode(self.text)) - block_size - 1)
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


if __name__ == "__main__":
    unittest.main()