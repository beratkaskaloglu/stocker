"""
signals/generator.py
Meta-learner çıktısından trading sinyali üretir.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TradingSignal:
    symbol: str
    market: str
    timeframe: str
    direction: int          # -1=sell, 0=hold, +1=buy
    target_price: float
    confidence: float       # [0, 1]
    raw_logits: np.ndarray  # direction logits (debug için)
    adjusted_confidence: float  # Error module sonrası


class SignalGenerator:
    """
    PSEUDO:
    1. generate(symbol, meta_output: dict, error_score: float) → TradingSignal
       a. direction = argmax(meta_output['direction_logits']) - 1  → {-1,0,+1}
       b. target_price = meta_output['price'] * current_price (denormalize)
       c. confidence = meta_output['confidence']
       d. adjusted_confidence = confidence * (1 - error_score)
       e. TradingSignal(...)
    2. filter_signals(signals: list[TradingSignal], min_confidence=0.6) → list
       a. Düşük güven sinyallerini Hold'a çevir
    3. to_json(signal) → dict
       a. outputs/signals/latest.json formatı
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def generate(
        self,
        symbol: str,
        market: str,
        timeframe: str,
        meta_output: dict,
        error_score: float,
        current_price: float,
    ) -> TradingSignal:
        # TODO: implement
        raise NotImplementedError

    def filter_signals(self, signals: list[TradingSignal]) -> list[TradingSignal]:
        # TODO: implement
        raise NotImplementedError
