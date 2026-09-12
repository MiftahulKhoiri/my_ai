"""
model/layers.py — Building block dasar untuk transformer: embedding
(token + posisi), feed-forward network, dan inisialisasi bobot ala GPT-2.
"""

import torch
import torch.nn as nn


class TokenAndPositionalEmbedding(nn.Module):
    """Gabungan token embedding + learned positional embedding."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        positions = torch.arange(t, device=idx.device).unsqueeze(0)  # (1, t)
        x = self.token_emb(idx) + self.pos_emb(positions)
        return self.dropout(x)


class FeedForward(nn.Module):
    """MLP posisi-demi-posisi dengan aktivasi GELU (desain standar GPT)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(module: nn.Module) -> None:
    """Inisialisasi ala GPT-2: normal kecil (std=0.02) untuk Linear/Embedding."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
