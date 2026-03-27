"""
core/models/meta_learner.py
Ensemble combiner — 4 modelin çıktısını birleştirir.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    """
    PSEUDO (Architecture):

    Input: 4 modelden gelen çıktılar:
        - lstm_out:       {direction_logits(3), price(1), confidence(1)}
        - cnn_lstm_out:   {direction_logits(3), price(1), confidence(1)}
        - transformer_out:{direction_logits(3), price(1), confidence(1)}
        - resnet_out:     {direction_logits(3), price(1), confidence(1)}

        Toplam: 4 × 5 = 20 değer → concat → (batch, 20)
        │
        ├─ FC(20, 64) → ReLU → Dropout(0.2)
        │
        ├─ FC(64, 32) → ReLU
        │
        └─ Final heads:
           ├─ direction:   FC(32, 3)     → final direction logits
           ├─ price:       FC(32, 1)     → final target price
           └─ confidence:  FC(32, 1) → Sigmoid

    Alternatif: Stacking yerine weighted average (öğrenilen ağırlıklar)
        w = Softmax(FC(20, 4))  → her model için ağırlık
        final = Σ w_i * model_i_output

    İki strateji de kodda desteklenecek (mode='stacking' | 'weighted')
    """

    def __init__(self, mode: str = "stacking", dropout: float = 0.2):
        super().__init__()
        self.mode = mode
        if mode == "stacking":
            self.net = nn.Sequential(
                nn.Linear(20, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(),
            )
            self.direction_head = nn.Linear(32, 3)
            self.price_head = nn.Linear(32, 1)
            self.confidence_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        elif mode == "weighted":
            self.weight_net = nn.Sequential(nn.Linear(20, 4), nn.Softmax(dim=-1))

    def forward(self, model_outputs: list[dict]) -> dict:
        # model_outputs: 4 dict, her biri {direction_logits(3), price(1), confidence(1)}
        # Concat: 4 × 5 = 20
        flat = torch.cat([
            torch.cat([o["direction_logits"], o["price"], o["confidence"]], dim=-1)
            for o in model_outputs
        ], dim=-1)  # (batch, 20)

        if self.mode == "stacking":
            x = self.net(flat)                         # (batch, 32)
            return {
                "direction_logits": self.direction_head(x),
                "price": self.price_head(x),
                "confidence": self.confidence_head(x),
            }
        else:  # weighted
            weights = self.weight_net(flat)            # (batch, 4)
            # Stack each output type: (batch, 4, dim)
            directions = torch.stack([o["direction_logits"] for o in model_outputs], dim=1)
            prices = torch.stack([o["price"] for o in model_outputs], dim=1)
            confidences = torch.stack([o["confidence"] for o in model_outputs], dim=1)
            # Weighted sum
            w = weights.unsqueeze(-1)                  # (batch, 4, 1)
            return {
                "direction_logits": (directions * w).sum(dim=1),   # (batch, 3)
                "price": (prices * w).sum(dim=1),                  # (batch, 1)
                "confidence": (confidences * w).sum(dim=1).clamp(0, 1),  # (batch, 1)
            }
