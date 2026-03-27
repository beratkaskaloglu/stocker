"""
signals/generator.py
Meta-learner çıktısından trading sinyali üretir.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

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
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


DIRECTION_MAP = {0: -1, 1: 0, 2: 1}  # logit index → direction


class SignalGenerator:
    """
    Meta-learner çıktısını alıp TradingSignal üretir.
    Error module skoru ile adjusted_confidence hesaplar.
    Düşük güvenli sinyalleri hold'a çevirir.
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
        """
        meta_output keys:
            direction_logits: np.ndarray shape (3,) — [sell, hold, buy]
            price: float — predicted price ratio (denormalize edilmesi gereken)
            confidence: float — [0, 1]
        """
        logits = np.asarray(meta_output["direction_logits"], dtype=np.float32)
        direction_idx = int(np.argmax(logits))
        direction = DIRECTION_MAP[direction_idx]

        target_price = float(meta_output["price"]) * current_price
        confidence = float(np.clip(meta_output["confidence"], 0.0, 1.0))
        adjusted_confidence = confidence * (1.0 - float(np.clip(error_score, 0.0, 1.0)))

        return TradingSignal(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            direction=direction,
            target_price=round(target_price, 4),
            confidence=round(confidence, 4),
            raw_logits=logits,
            adjusted_confidence=round(adjusted_confidence, 4),
        )

    def filter_signals(self, signals: list[TradingSignal]) -> list[TradingSignal]:
        """Düşük güvenli sinyalleri hold'a çevir."""
        filtered: list[TradingSignal] = []
        for sig in signals:
            if sig.adjusted_confidence < self.min_confidence:
                sig = TradingSignal(
                    symbol=sig.symbol,
                    market=sig.market,
                    timeframe=sig.timeframe,
                    direction=0,
                    target_price=sig.target_price,
                    confidence=sig.confidence,
                    raw_logits=sig.raw_logits,
                    adjusted_confidence=sig.adjusted_confidence,
                    timestamp=sig.timestamp,
                )
            filtered.append(sig)
        return filtered

    @staticmethod
    def to_json(signal: TradingSignal) -> dict:
        """outputs/signals/latest.json formatında dict döndürür."""
        return {
            "symbol": signal.symbol,
            "market": signal.market,
            "timeframe": signal.timeframe,
            "direction": signal.direction,
            "direction_label": {-1: "SELL", 0: "HOLD", 1: "BUY"}[signal.direction],
            "target_price": signal.target_price,
            "confidence": signal.confidence,
            "adjusted_confidence": signal.adjusted_confidence,
            "raw_logits": signal.raw_logits.tolist(),
            "timestamp": signal.timestamp,
        }
