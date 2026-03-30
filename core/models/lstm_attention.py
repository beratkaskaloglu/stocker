"""
core/models/lstm_attention.py
LSTM + Multi-head Attention modeli — multi-horizon tahmin.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.models.horizons import MultiHorizonHead, DEFAULT_HORIZON_NAMES


class LSTMAttentionModel(nn.Module):
    """
    Input: (batch, seq_len=60, feature_dim)
        │
        ├─ Linear(feature_dim, 256)  → embedding
        ├─ LSTM(256, 512, layers=2, bidirectional) → (batch, seq_len, 1024)
        ├─ Multi-head Attention(8 heads) → (batch, seq_len, 1024)
        ├─ Layer Norm + Residual
        ├─ Global Avg Pool → (batch, 1024)
        ├─ FC(1024, 256) → ReLU → Dropout
        └─ MultiHorizonHead(256) → per-horizon (direction, price, confidence)
    """

    def __init__(
        self,
        feature_dim: int = 211,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        horizon_names: list[str] | None = None,
    ):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiHorizonHead(256, horizon_names or DEFAULT_HORIZON_NAMES)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.norm(attn_out + lstm_out)
        x = x.mean(dim=1)
        x = self.fc(x)
        return self.heads(x)
