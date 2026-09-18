"""
tests/test_checkpoint.py — Tes rotate_checkpoints (hapus checkpoint periodik
lama, sisakan N terbaru, tanpa pernah menyentuh best.pt/final.pt).

training/checkpoint.py import torch di level modul, jadi tes ini otomatis
di-skip kalau torch tidak terpasang (mis. saat dites tanpa environment penuh).

Jalankan dari root project:
    python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401  (cuma buat cek ketersediaan)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from training.checkpoint import rotate_checkpoints


def _touch(path: str) -> None:
    open(path, "w").close()


@unittest.skipUnless(TORCH_AVAILABLE, "butuh torch terpasang")
class TestRotateCheckpoints(unittest.TestCase):
    def test_keeps_only_last_n_by_step_number(self):
        with tempfile.TemporaryDirectory() as d:
            for step in [500, 1000, 1500, 2000, 2500]:
                _touch(os.path.join(d, f"step_{step}.pt"))
            _touch(os.path.join(d, "best.pt"))
            _touch(os.path.join(d, "final.pt"))

            rotate_checkpoints(d, keep_last_n=3)
            remaining = sorted(os.listdir(d))

        self.assertIn("best.pt", remaining)
        self.assertIn("final.pt", remaining)
        self.assertEqual(
            sorted(f for f in remaining if f.startswith("step_")),
            ["step_1500.pt", "step_2000.pt", "step_2500.pt"],
        )

    def test_disabled_when_keep_last_n_is_none_or_zero(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "step_500.pt"))
            _touch(os.path.join(d, "step_1000.pt"))

            rotate_checkpoints(d, keep_last_n=None)
            rotate_checkpoints(d, keep_last_n=0)

            remaining = sorted(os.listdir(d))

        self.assertEqual(remaining, ["step_1000.pt", "step_500.pt"])

    def test_no_error_when_fewer_checkpoints_than_keep_last_n(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "step_500.pt"))
            removed = rotate_checkpoints(d, keep_last_n=3)
            remaining = os.listdir(d)

        self.assertEqual(removed, [])
        self.assertEqual(remaining, ["step_500.pt"])

    def test_sorts_by_step_number_not_filename_string_order(self):
        # Urut string: "step_10000.pt" < "step_9000.pt" (karakter '1' < '9'),
        # tapi step 10000 harus dianggap LEBIH BARU. Pastikan rotate_checkpoints
        # bandingin angka step-nya, bukan nama filenya sebagai string.
        with tempfile.TemporaryDirectory() as d:
            for step in [9000, 10000]:
                _touch(os.path.join(d, f"step_{step}.pt"))

            rotate_checkpoints(d, keep_last_n=1)
            remaining = os.listdir(d)

        self.assertEqual(remaining, ["step_10000.pt"])


if __name__ == "__main__":
    unittest.main()