"""
tests/test_config.py — Tes save/load Config (round-trip JSON).

Jalankan dari root project:
    python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, ModelConfig, TrainingConfig, DataConfig


class TestConfig(unittest.TestCase):
    def test_roundtrip(self):
        cfg = Config(
            model=ModelConfig(d_model=128, n_layers=2),
            training=TrainingConfig(batch_size=8, learning_rate=1e-3),
            data=DataConfig(tokenizer="bpe", bpe_vocab_size=512),
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg.save(path)
            loaded = Config.load(path)

        self.assertEqual(loaded.model.d_model, 128)
        self.assertEqual(loaded.model.n_layers, 2)
        self.assertEqual(loaded.training.batch_size, 8)
        self.assertEqual(loaded.training.learning_rate, 1e-3)
        self.assertEqual(loaded.data.tokenizer, "bpe")
        self.assertEqual(loaded.data.bpe_vocab_size, 512)

    def test_defaults(self):
        cfg = Config()
        self.assertEqual(cfg.data.tokenizer, "bpe")
        self.assertEqual(cfg.training.grad_accum_steps, 1)
        self.assertIsNone(cfg.training.num_threads)

    def test_load_missing_file_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            Config.load("/path/yang/jelas/tidak/ada/config.json")


if __name__ == "__main__":
    unittest.main()