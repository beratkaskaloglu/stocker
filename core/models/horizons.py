"""
core/models/horizons.py
Multi-horizon tahmin konfigürasyonu.

Her model aynı backbone'u kullanır, ama her horizon için ayrı output head'ler var.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ─── Horizon tanımları ───────────────────────────────────────────────────────

@dataclass
class Horizon:
    name: str           # "1h", "4h", "1d", "15d", "1m"
    label: str          # "1 Saat", "4 Saat", "1 Gün", "15 Gün", "1 Ay"
    periods: int        # Kaç bar ileri (veri tipine göre: daily→gün, hourly→saat)
    data_type: str      # "intraday" veya "daily"


# Daily data horizons (5 yıl)
DAILY_HORIZONS = [
    Horizon("1d",  "1 Gün",    1,   "daily"),
    Horizon("15d", "15 Gün",   15,  "daily"),
    Horizon("1m",  "1 Ay",     21,  "daily"),    # ~21 işlem günü
    Horizon("3m",  "3 Ay",     63,  "daily"),    # ~63 işlem günü
    Horizon("6m",  "6 Ay",     126, "daily"),    # ~126 işlem günü
    Horizon("1y",  "1 Yıl",    252, "daily"),    # ~252 işlem günü
]

# Intraday data horizons (son 2 yıl, 1h bar)
INTRADAY_HORIZONS = [
    Horizon("1h",  "1 Saat", 1,   "intraday"),  # 1h bar → 1 bar ileri
    Horizon("4h",  "4 Saat", 4,   "intraday"),  # 1h bar → 4 bar ileri
]

# Tüm horizonlar
ALL_HORIZONS = INTRADAY_HORIZONS + DAILY_HORIZONS
DEFAULT_HORIZON_NAMES = [h.name for h in ALL_HORIZONS]

# Sadece daily horizon isimleri (dataset build için)
DAILY_HORIZON_NAMES = [h.name for h in DAILY_HORIZONS]
INTRADAY_HORIZON_NAMES = [h.name for h in INTRADAY_HORIZONS]

# Horizon → periods lookup
DAILY_HORIZON_PERIODS = {h.name: h.periods for h in DAILY_HORIZONS}
INTRADAY_HORIZON_PERIODS = {h.name: h.periods for h in INTRADAY_HORIZONS}


# ─── Multi-Horizon Output Head ──────────────────────────────────────────────

class MultiHorizonHead(nn.Module):
    """
    Tek backbone çıktısından N horizon için ayrı tahminler üretir.

    Her horizon için:
        - direction_logits: (batch, 3)  → SELL/HOLD/BUY
        - price:            (batch, 1)  → target price ratio
        - confidence:       (batch, 1)  → [0, 1]

    Output format:
        {
            "1d":  {"direction_logits": (B,3), "price": (B,1), "confidence": (B,1)},
            "15d": {"direction_logits": (B,3), "price": (B,1), "confidence": (B,1)},
            "1m":  {"direction_logits": (B,3), "price": (B,1), "confidence": (B,1)},
        }
    """

    def __init__(self, in_features: int, horizon_names: list[str] | None = None):
        super().__init__()
        self.horizon_names = horizon_names or DEFAULT_HORIZON_NAMES

        # Her horizon için ayrı head
        self.direction_heads = nn.ModuleDict()
        self.price_heads = nn.ModuleDict()
        self.confidence_heads = nn.ModuleDict()

        for h in self.horizon_names:
            key = h.replace("-", "_")  # ModuleDict key'lerde - kullanılamaz
            self.direction_heads[key] = nn.Linear(in_features, 3)
            self.price_heads[key] = nn.Linear(in_features, 1)
            self.confidence_heads[key] = nn.Sequential(
                nn.Linear(in_features, 1), nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """x: (batch, in_features) → per-horizon predictions."""
        output = {}
        for h in self.horizon_names:
            key = h.replace("-", "_")
            output[h] = {
                "direction_logits": self.direction_heads[key](x),
                "price": self.price_heads[key](x),
                "confidence": self.confidence_heads[key](x),
            }
        return output

    @property
    def num_horizons(self) -> int:
        return len(self.horizon_names)
