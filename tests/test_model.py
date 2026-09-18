"""
tests/test_model.py — Tes numerik untuk TransformerLM: bentuk output,
causal masking, dan yang paling penting, KONSISTENSI antara forward
dengan KV-cache vs tanpa cache (kelas bug yang gampang lolos diam-diam
kalau cuma dites lewat "modelnya jalan tanpa error").

Butuh torch terpasang untuk jalan (otomatis di-skip kalau tidak ada).

Jalankan dari root project:
    python -m unittest discover tests
    python -m unittest tests.test_model -v
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

if TORCH_AVAILABLE:
    from config import ModelConfig
    from model.transformer import TransformerLM

    class _ForcedEOSModel(TransformerLM):
        """Subclass TransformerLM cuma buat tes ALUR generate() (berhenti di
        <eos>, dst) secara terisolasi dari kualitas prediksi model sungguhan
        -- forward() diganti supaya token <eos> SELALU logit tertinggi,
        jadi hasil generate()-nya deterministik tanpa perlu model terlatih."""

        def __init__(self, config, eos_id):
            super().__init__(config)
            self._eos_id = eos_id

        def forward(self, idx, targets=None, past_kv=None, use_cache=False):
            b, t = idx.shape
            logits = torch.full((b, t, self.config.vocab_size), -1e9)
            logits[:, :, self._eos_id] = 1e9
            return logits, None, None


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=32, d_model=16, n_layers=2, n_heads=2,
        d_ff=32, max_seq_len=32, dropout=0.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@unittest.skipUnless(TORCH_AVAILABLE, "butuh torch terpasang")
class TestTransformerLM(unittest.TestCase):
    def test_forward_output_shape_and_loss(self):
        model = TransformerLM(_tiny_config())
        idx = torch.randint(0, 32, (2, 10))
        targets = torch.randint(0, 32, (2, 10))

        logits, loss, _ = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 10, 32))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        _, loss_none, _ = model(idx)
        self.assertIsNone(loss_none)

    def test_causal_mask_blocks_future_tokens(self):
        torch.manual_seed(0)
        model = TransformerLM(_tiny_config())
        model.eval()

        idx_a = torch.tensor([[1, 2, 3, 4, 5]])
        idx_b = idx_a.clone()
        idx_b[0, -1] = 31  # ubah token TERAKHIR saja

        with torch.no_grad():
            logits_a, _, _ = model(idx_a)
            logits_b, _, _ = model(idx_b)

        # Logit di posisi SEBELUM token yang diubah harus identik -- token
        # awal tidak boleh "mengintip" token yang lebih baru lewat attention.
        self.assertTrue(torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5))

    def test_kv_cache_matches_full_forward(self):
        """Tes paling penting di file ini: next-token logit dari forward
        penuh (tanpa cache) HARUS sama dengan hasil prefill+decode pakai
        KV-cache. Kalau beda, ada bug di logika cache (positional offset,
        mask, dst) -- akan sulit terdeteksi lewat pemakaian normal karena
        modelnya tetap "jalan", cuma hasilnya diam-diam salah."""
        torch.manual_seed(0)
        model = TransformerLM(_tiny_config())
        model.eval()

        prompt = torch.tensor([[1, 2, 3, 4]])
        next_token = torch.tensor([[7]])
        full_seq = torch.cat([prompt, next_token], dim=1)

        with torch.no_grad():
            logits_full, _, _ = model(full_seq, use_cache=False)
            logits_full_last = logits_full[:, -1, :]

            _, _, past_kv = model(prompt, use_cache=True)
            logits_cached, _, _ = model(next_token, past_kv=past_kv, use_cache=True)
            logits_cached_last = logits_cached[:, -1, :]

        self.assertTrue(torch.allclose(logits_full_last, logits_cached_last, atol=1e-4))

    def test_generate_without_eos_runs_full_length(self):
        torch.manual_seed(0)
        model = TransformerLM(_tiny_config())
        model.eval()

        idx = torch.tensor([[1, 2, 3]])
        out = model.generate(idx, max_new_tokens=10, temperature=1.0)
        self.assertEqual(out.shape[1], 3 + 10)

    def test_generate_stops_at_eos_immediately(self):
        eos_id = 5
        model = _ForcedEOSModel(_tiny_config(), eos_id=eos_id)

        idx = torch.tensor([[1, 2, 3]])
        out = model.generate(idx, max_new_tokens=20, temperature=1.0, eos_id=eos_id)

        # Prompt 3 token + tepat 1 token baru (<eos>) -- harus berhenti jauh
        # sebelum max_new_tokens=20 tercapai.
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(out[0, -1].item(), eos_id)


if __name__ == "__main__":
    unittest.main()