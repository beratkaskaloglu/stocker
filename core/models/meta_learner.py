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
        # TODO: implement both modes
        raise NotImplementedError
