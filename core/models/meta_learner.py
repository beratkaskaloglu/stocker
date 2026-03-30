"""
core/models/meta_learner.py
Ensemble combiner — 4 modelin multi-horizon çıktısını birleştirir.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.models.horizons import MultiHorizonHead, DEFAULT_HORIZON_NAMES


class MetaLearner(nn.Module):
    """
    Input: 4 modelden multi-horizon çıktılar:
        Her model → {horizon: {direction_logits(3), price(1), confidence(1)}}

    Her horizon için:
        4 model × 5 değer = 20 → FC(20,64) → FC(64,32) → output heads

    Output: {horizon: {direction_logits(3), price(1), confidence(1)}}
    """

    def __init__(self, mode: str = "stacking", dropout: float = 0.2,
                 horizon_names: list[str] | None = None):
        super().__init__()
        self.mode = mode
        self.horizon_names = horizon_names or DEFAULT_HORIZON_NAMES

        if mode == "stacking":
            # Per-horizon stacking networks
            self.nets = nn.ModuleDict()
            self.direction_heads = nn.ModuleDict()
            self.price_heads = nn.ModuleDict()
            self.confidence_heads = nn.ModuleDict()

            for h in self.horizon_names:
                key = h.replace("-", "_")
                self.nets[key] = nn.Sequential(
                    nn.Linear(20, 64), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(64, 32), nn.ReLU(),
                )
                self.direction_heads[key] = nn.Linear(32, 3)
                self.price_heads[key] = nn.Linear(32, 1)
                self.confidence_heads[key] = nn.Sequential(
                    nn.Linear(32, 1), nn.Sigmoid()
                )
        elif mode == "weighted":
            self.weight_nets = nn.ModuleDict()
            for h in self.horizon_names:
                key = h.replace("-", "_")
                self.weight_nets[key] = nn.Sequential(
                    nn.Linear(20, 4), nn.Softmax(dim=-1)
                )

    def forward(self, model_outputs: list[dict]) -> dict:
        """
        model_outputs: list of 4 dicts, each:
            {horizon_name: {direction_logits, price, confidence}}
        """
        result = {}

        for h in self.horizon_names:
            key = h.replace("-", "_")

            # Flatten: 4 models × 5 values = 20
            flat = torch.cat([
                torch.cat([
                    o[h]["direction_logits"],
                    o[h]["price"],
                    o[h]["confidence"],
                ], dim=-1)
                for o in model_outputs
            ], dim=-1)  # (batch, 20)

            if self.mode == "stacking":
                x = self.nets[key](flat)
                result[h] = {
                    "direction_logits": self.direction_heads[key](x),
                    "price": self.price_heads[key](x),
                    "confidence": self.confidence_heads[key](x),
                }
            else:  # weighted
                weights = self.weight_nets[key](flat)  # (batch, 4)
                directions = torch.stack([o[h]["direction_logits"] for o in model_outputs], dim=1)
                prices = torch.stack([o[h]["price"] for o in model_outputs], dim=1)
                confidences = torch.stack([o[h]["confidence"] for o in model_outputs], dim=1)
                w = weights.unsqueeze(-1)
                result[h] = {
                    "direction_logits": (directions * w).sum(dim=1),
                    "price": (prices * w).sum(dim=1),
                    "confidence": (confidences * w).sum(dim=1).clamp(0, 1),
                }

        return result
