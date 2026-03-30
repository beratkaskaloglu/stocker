"""
core/models/resnet_gasf.py
ResNet — GASF 2D image pattern tanıma, multi-horizon tahmin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

from core.models.horizons import MultiHorizonHead, DEFAULT_HORIZON_NAMES


class ResNetGASFModel(nn.Module):
    """
    Input: (batch, 3, 64, 64) ← GASF RGB image
        │
        ├─ ResNet-18 (no pretrain, stride=1 for 64×64)
        ├─ FC(512, 128) → ReLU → Dropout
        └─ MultiHorizonHead(128) → per-horizon (direction, price, confidence)
    """

    def __init__(self, dropout: float = 0.3, horizon_names: list[str] | None = None):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout))
        self.heads = MultiHorizonHead(128, horizon_names or DEFAULT_HORIZON_NAMES)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.heads(x)
