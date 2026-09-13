"""
model/attention.py — Causal multi-head self-attention, dengan dukungan
KV-cache untuk decoding autoregresif yang efisien.

Tanpa cache: query di posisi i memperhatikan semua key di posisi <= i
(mask segitiga bawah) — dipakai saat training dan saat prefill prompt.

Dengan cache: satu token baru (t=1) diproses sekaligus digabung dengan
key/value dari token-token sebelumnya (past_kv), tanpa perlu mengulang
komputasi atas seluruh konteks dari awal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model harus habis dibagi n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        b, t, c = x.shape

        qkv = self.qkv_proj(x)                      # (b, t, 3*c)
        q, k, v = qkv.split(c, dim=2)

        # (b, t, c) -> (b, n_heads, t, head_dim)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            assert t == 1, (
                "Decoding dengan cache hanya mendukung satu token baru per langkah"
            )
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if past_kv is None:
            # Forward penuh (training / prefill): terapkan mask causal biasa.
            t_k = k.size(2)
            att = att.masked_fill(self.causal_mask[:, :, :t, :t_k] == 0, float("-inf"))
        # Kalau past_kv ada, query barunya selalu yang paling akhir, jadi
        # boleh melihat seluruh key (lama + baru) tanpa mask tambahan.

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v                                 # (b, n_heads, t, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, c)

        return self.resid_dropout(self.out_proj(out)), present_kv