"""
core/models/lstm_attention.py
LSTM + Multi-head Attention modeli.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAttentionModel(nn.Module):
    """
    PSEUDO (Architecture):

    Input: (batch, seq_len=60, feature_dim=211)
        │
        ├─ Linear(211, hidden=256)   → embedding
        │
        ├─ LSTM(256, 512, layers=2, dropout=0.3, bidirectional=True)
        │   Output: (batch, seq_len, 1024)   [bidirectional: 512*2]
        │
        ├─ Multi-head Attention(heads=8, d_model=1024)
        │   Output: (batch, seq_len, 1024)
        │
        ├─ Layer Norm + Residual
        │
        ├─ Global Avg Pool → (batch, 1024)
        │
        ├─ FC(1024, 256) → ReLU → Dropout(0.3)
        │
        └─ 3 çıkış kafası:
           ├─ direction_head:  FC(256, 3)     → logits {-1, 0, +1}
           ├─ price_head:      FC(256, 1)     → target price (normalized)
           └─ confidence_head: FC(256, 1) → Sigmoid → [0, 1]

    Loss:
        total = CrossEntropy(direction) + MSE(price) * 0.3 + BCE(confidence) * 0.1
    """

    def __init__(
        self,
        feature_dim: int = 211,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
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
        # Output heads
        self.direction_head = nn.Linear(256, 3)
        self.price_head = nn.Linear(256, 1)
        self.confidence_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict:
        # TODO: implement forward pass
        # 1. embedding
        # 2. lstm
        # 3. attention + residual + norm
        # 4. global avg pool
        # 5. fc
        # 6. 3 head çıktısı → dict
        raise NotImplementedError
