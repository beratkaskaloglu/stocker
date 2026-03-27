"""
core/models/transformer.py
Transformer Encoder — haber sentimenti + market feature fusion.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    PSEUDO (Architecture):

    Input: (batch, seq_len=60, feature_dim=211)
        │
        ├─ Positional Encoding (sinusoidal)
        │
        ├─ Linear(211, d_model=256)
        │
        ├─ Transformer Encoder × 4 layers:
        │   each: MultiheadAttention(heads=8) → FF(1024) → LayerNorm
        │   Output: (batch, seq_len, 256)
        │
        ├─ [CLS token] veya mean pooling → (batch, 256)
        │
        ├─ FC(256, 128) → GELU → Dropout(0.3)
        │
        └─ 3 output head

    Transformer'ın avantajı: Sentiment vektörü ile fiyat feature'larını
    global attention ile ilişkilendirebilir.
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
        self.direction_head = nn.Linear(128, 3)
        self.price_head = nn.Linear(128, 1)
        self.confidence_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict:
        # x: (batch, seq_len, feature_dim)
        x = self.input_proj(x)                         # (batch, seq_len, d_model)
        x = self.pos_encoding(x)                       # + positional encoding
        x = self.encoder(x)                            # (batch, seq_len, d_model)
        x = x.mean(dim=1)                              # mean pooling → (batch, d_model)
        x = self.fc(x)                                 # (batch, 128)
        return {
            "direction_logits": self.direction_head(x),
            "price": self.price_head(x),
            "confidence": self.confidence_head(x),
        }


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
