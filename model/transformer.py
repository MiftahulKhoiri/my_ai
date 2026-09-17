"""
model/transformer.py — TransformerBlock (pre-norm attention + pre-norm FFN)
dan model decoder-only penuh (gaya GPT) untuk language modeling, dengan
dukungan KV-cache supaya generate() tidak perlu mengulang seluruh konteks
di setiap langkah.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .layers import TokenAndPositionalEmbedding, FeedForward, init_weights


class TransformerBlock(nn.Module):
    """Satu blok transformer decoder dengan residual connection di tiap
    sub-layer (attention dan feed-forward)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, present_kv


class TransformerLM(nn.Module):
    """Decoder-only transformer untuk language modeling (gaya GPT)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = TokenAndPositionalEmbedding(
            config.vocab_size, config.d_model, config.max_seq_len, config.dropout
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.d_ff,
                    config.max_seq_len, config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.embed.token_emb.weight

        self.apply(init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_kv=None, use_cache: bool = False):
        # start_pos = panjang cache sejauh ini (0 kalau belum ada cache),
        # supaya positional embedding token baru tetap absolut & konsisten.
        start_pos = past_kv[0][0].size(2) if past_kv is not None else 0
        x = self.embed(idx, start_pos=start_pos)

        new_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, present = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_kv.append(present)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_kv

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        eos_id: int = None,
    ) -> torch.Tensor:
        """Generate token baru secara autoregresif memakai KV-cache.

        Prompt awal diproses sekali saja (prefill, membangun cache), lalu
        tiap token baru cukup satu forward pass ringan atas 1 token (bukan
        mengulang seluruh konteks dari awal setiap langkah seperti versi
        sebelumnya). Total posisi (prompt + token baru) dibatasi ke
        `max_seq_len`; kalau limit itu tercapai, generate berhenti lebih
        awal dan memberi peringatan, bukan diam-diam memotong konteks.

        `eos_id`: kalau diisi, generate berhenti lebih awal begitu SEMUA
        sequence di batch sudah menghasilkan token ini minimal sekali —
        tidak perlu menunggu `max_new_tokens` penuh. Ini cuma berguna kalau
        modelnya dilatih dengan <eos> disisipkan di data training (lihat
        `data/dataset.py::encode_corpus`); kalau None (default), perilaku
        sama seperti sebelumnya (selalu generate `max_new_tokens` penuh).
        """
        was_training = self.training
        self.eval()

        max_len = self.config.max_seq_len
        if idx.size(1) >= max_len:
            idx = idx[:, -(max_len - 1):]

        allowed_new_tokens = max_len - idx.size(1)
        if max_new_tokens > allowed_new_tokens:
            print(
                f"[peringatan] max_new_tokens dipotong dari {max_new_tokens} ke "
                f"{allowed_new_tokens} karena batas max_seq_len={max_len}"
            )
            max_new_tokens = allowed_new_tokens

        # --- Prefill: proses seluruh prompt sekali, bangun cache awal ---
        logits, _, past_kv = self(idx, use_cache=True)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None:
                finished = finished | (next_id.squeeze(-1) == eos_id)
                if finished.all():
                    break

            # --- Decode 1 token baru pakai cache, bukan ulang dari awal ---
            logits, _, past_kv = self(next_id, past_kv=past_kv, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

        self.train(was_training)
        return idx