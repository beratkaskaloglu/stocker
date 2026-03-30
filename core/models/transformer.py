"""
core/models/transformer.py
Transformer Encoder — haber sentimenti + market feature fusion, multi-horizon tahmin.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from core.models.horizons import MultiHorizonHead, DEFAULT_HORIZON_NAMES


class TransformerModel(nn.Module):
    """
    Input: (batch, seq_len=60, feature_dim)
        │
        ├─ Linear(feature_dim, d_model=256) + Positional Encoding
        ├─ Transformer Encoder × 4 layers
        ├─ Mean pooling → (batch, 256)
        ├─ FC(256, 128) → GELU → Dropout
        └─ MultiHorizonHead(128) → per-horizon (direction, price, confidence)
    """

    def __init__(
        self,
        feature_dim: int = 211,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_seq_len: int = 60,
        horizon_names: list[str] | None = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout))
        self.heads = MultiHorizonHead(128, horizon_names or DEFAULT_HORIZON_NAMES)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return self.heads(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
