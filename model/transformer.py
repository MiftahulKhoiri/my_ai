"""
model/transformer.py — TransformerBlock (pre-norm attention + pre-norm FFN)
dan model decoder-only penuh (gaya GPT) untuk language modeling.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """Generate token baru secara autoregresif dari konteks awal `idx`."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx
