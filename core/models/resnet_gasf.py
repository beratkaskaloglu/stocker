"""
core/models/resnet_gasf.py
ResNet — GASF 2D image pattern tanıma.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetGASFModel(nn.Module):
    """
    PSEUDO (Architecture):

    Input: (batch, 3, 64, 64)   ← GASF RGB image
        │
        ├─ ResNet-18 (pretrained=False, ilk conv stride=1 küçük image için)
        │   ilk Conv2d: kernel=3, stride=1, padding=1  (64x64 için override)
        │   MaxPool: kaldır veya stride=1
        │   Output: (batch, 512) — avgpool sonrası
        │
        ├─ FC(512, 128) → ReLU → Dropout(0.3)
        │
        └─ 3 output head

    Not: GASF 64×64 küçük image — ResNet18 yeterli (ResNet50 overkill)
    Pretrained ImageNet ağırlıkları kullanma — finansal pattern tamamen farklı
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        # İlk katmanı 64×64 için uyarla
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()  # MaxPool'u kaldır
        # Son FC'yi çıkar, feature extractor olarak kullan
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # avgpool dahil
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout))
        self.direction_head = nn.Linear(128, 3)
        self.price_head = nn.Linear(128, 1)
        self.confidence_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict:
        # x: (batch, 3, 64, 64)
        x = self.backbone(x)                           # (batch, 512, 1, 1)
        x = x.flatten(1)                               # (batch, 512)
        x = self.fc(x)                                 # (batch, 128)
        return {
            "direction_logits": self.direction_head(x),
            "price": self.price_head(x),
            "confidence": self.confidence_head(x),
        }
