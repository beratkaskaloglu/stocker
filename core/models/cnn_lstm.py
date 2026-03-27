"""
core/models/cnn_lstm.py
CNN-LSTM hybrid — yerel + global pattern tanıma.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    PSEUDO (Architecture):

    Input: (batch, seq_len=60, feature_dim=211)
        │
        ├─ Reshape → (batch, feature_dim=211, seq_len=60)  [Conv1d için]
        │
        ├─ Conv1d block x3:
        │   Conv1d(211→128, kernel=3) → BN → ReLU → MaxPool(2)
        │   Conv1d(128→256, kernel=3) → BN → ReLU → MaxPool(2)
        │   Conv1d(256→512, kernel=3) → BN → ReLU
        │   Output: (batch, 512, reduced_len)
        │
        ├─ Permute → (batch, reduced_len, 512)
        │
        ├─ LSTM(512, 256, layers=2, dropout=0.3)
        │   Output: (batch, reduced_len, 256)
        │
        ├─ Son zaman adımı: (batch, 256)
        │
        ├─ FC(256, 128) → ReLU → Dropout(0.3)
        │
        └─ 3 output head (direction, price, confidence)

    CNN'nin amacı: Kısa vadeli yerel patternleri (fibonacci, flag, triangle) yakalar
    LSTM'nin amacı: CNN çıktısından uzun vadeli bağımlılığı öğrenir
    """

    def __init__(self, feature_dim: int = 211, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 256, num_layers=2, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout))
        self.direction_head = nn.Linear(128, 3)
        self.price_head = nn.Linear(128, 1)
        self.confidence_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict:
        # x: (batch, seq_len, feature_dim)
        x = x.permute(0, 2, 1)                        # (batch, feature_dim, seq_len) for Conv1d
        x = self.cnn(x)                               # (batch, 512, reduced_len)
        x = x.permute(0, 2, 1)                        # (batch, reduced_len, 512)
        x, _ = self.lstm(x)                            # (batch, reduced_len, 256)
        x = x[:, -1, :]                               # son zaman adımı → (batch, 256)
        x = self.fc(x)                                 # (batch, 128)
        return {
            "direction_logits": self.direction_head(x),
            "price": self.price_head(x),
            "confidence": self.confidence_head(x),
        }
