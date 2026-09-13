"""
tests/test_tokenizer.py — Tes dasar untuk CharTokenizer & BPETokenizer:
round-trip encode/decode dan save/load.

Jalankan dari root project:
    python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import CharTokenizer
from data.bpe_tokenizer import BPETokenizer

SAMPLE_TEXT = (
    "Pada suatu hari di sebuah desa kecil di kaki gunung, hiduplah "
    "seorang anak bernama Rian. Ia senang belajar hal baru."
)


class TestCharTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = CharTokenizer()
        tok.fit(SAMPLE_TEXT)
        ids = tok.encode(SAMPLE_TEXT)
        self.assertEqual(tok.decode(ids), SAMPLE_TEXT)

    def test_special_tokens(self):
        tok = CharTokenizer()
        tok.fit(SAMPLE_TEXT)
        ids = tok.encode(SAMPLE_TEXT, add_special_tokens=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)

    def test_save_load(self):
        tok = CharTokenizer()
        tok.fit(SAMPLE_TEXT)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tok.json")
            tok.save(path)
            tok2 = CharTokenizer.load(path)
        self.assertEqual(tok.vocab_size, tok2.vocab_size)
        self.assertEqual(tok.encode(SAMPLE_TEXT), tok2.encode(SAMPLE_TEXT))


class TestBPETokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = BPETokenizer()
        tok.fit(SAMPLE_TEXT, vocab_size=300, verbose=False)
        ids = tok.encode(SAMPLE_TEXT)
        self.assertEqual(tok.decode(ids), SAMPLE_TEXT)

    def test_compresses_better_than_char_level(self):
        tok = BPETokenizer()
        tok.fit(SAMPLE_TEXT, vocab_size=300, verbose=False)
        ids = tok.encode(SAMPLE_TEXT)
        self.assertLess(len(ids), len(SAMPLE_TEXT))

    def test_save_load(self):
        tok = BPETokenizer()
        tok.fit(SAMPLE_TEXT, vocab_size=300, verbose=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bpe.json")
            tok.save(path)
            tok2 = BPETokenizer.load(path)
        self.assertEqual(tok.vocab_size, tok2.vocab_size)
        self.assertEqual(tok.encode(SAMPLE_TEXT), tok2.encode(SAMPLE_TEXT))

    def test_handles_non_ascii(self):
        # BPE level byte harus tetap benar untuk karakter non-ASCII
        # (huruf berdiakritik, emoji, dll) — inilah alasan byte-level
        # dipilih, supaya tidak pernah butuh token <unk>.
        text = "Selamat pagi! Apa kabar? Café résumé naïve 🙂"
        tok = BPETokenizer()
        tok.fit(text, vocab_size=280, verbose=False)
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)


if __name__ == "__main__":
    unittest.main()