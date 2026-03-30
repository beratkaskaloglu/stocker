"""
core/models/cnn_lstm.py
CNN-LSTM hybrid — yerel + global pattern tanıma, multi-horizon tahmin.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.models.horizons import MultiHorizonHead, DEFAULT_HORIZON_NAMES


class CNNLSTMModel(nn.Module):
    """
    Input: (batch, seq_len=60, feature_dim)
        │
        ├─ Conv1d x3: feature_dim→128→256→512, BN, ReLU, MaxPool
        ├─ LSTM(512, 256, layers=2)
        ├─ Son zaman adımı → (batch, 256)
        ├─ FC(256, 128) → ReLU → Dropout
        └─ MultiHorizonHead(128) → per-horizon (direction, price, confidence)
    """

    def __init__(self, feature_dim: int = 211, dropout: float = 0.3,
                 horizon_names: list[str] | None = None):
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
        self.heads = MultiHorizonHead(128, horizon_names or DEFAULT_HORIZON_NAMES)

    def forward(self, x: torch.Tensor) -> dict:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return self.heads(x)
