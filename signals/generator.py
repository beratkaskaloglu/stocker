"""
signals/generator.py
Multi-horizon trading sinyali üretimi.

Her horizon (1h, 4h, 1d, 15d, 1m) için ayrı sinyal üretir.
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
    timeframe: str           # "1h", "4h", "1d", "15d", "1m"
    direction: int           # -1=sell, 0=hold, +1=buy
    target_price: float
    confidence: float        # [0, 1]
    raw_logits: np.ndarray   # direction logits (debug)
    adjusted_confidence: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


DIRECTION_MAP = {0: -1, 1: 0, 2: 1}  # logit index → direction


class SignalGenerator:
    """
    Multi-horizon model çıktısından TradingSignal üretir.

    Model output format:
        {
            "1d":  {"direction_logits": [3], "price": [1], "confidence": [1]},
            "15d": {"direction_logits": [3], "price": [1], "confidence": [1]},
            "1m":  {"direction_logits": [3], "price": [1], "confidence": [1]},
        }
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def generate_all(
        self,
        symbol: str,
        market: str,
        multi_output: dict,
        error_scores: dict[str, float] | None = None,
        current_price: float = 0.0,
    ) -> list[TradingSignal]:
        """
        Multi-horizon model çıktısından tüm timeframe'ler için sinyal üret.

        Returns: list of TradingSignal (one per horizon)
        """
        if error_scores is None:
            error_scores = {}

        signals = []
        for timeframe, output in multi_output.items():
            error_score = error_scores.get(timeframe, 0.0)
            signal = self.generate(
                symbol, market, timeframe, output, error_score, current_price
            )
            signals.append(signal)

        return self.filter_signals(signals)

    def generate(
        self,
        symbol: str,
        market: str,
        timeframe: str,
        meta_output: dict,
        error_score: float,
        current_price: float,
    ) -> TradingSignal:
        """Tek horizon için sinyal üret."""
        logits = np.asarray(meta_output["direction_logits"], dtype=np.float32)
        direction_idx = int(np.argmax(logits))
        direction = DIRECTION_MAP[direction_idx]

        target_price = float(meta_output["price"]) * current_price if current_price > 0 else 0.0
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
        filtered = []
        for sig in signals:
            if sig.adjusted_confidence < self.min_confidence:
                sig = TradingSignal(
                    symbol=sig.symbol, market=sig.market, timeframe=sig.timeframe,
                    direction=0, target_price=sig.target_price,
                    confidence=sig.confidence, raw_logits=sig.raw_logits,
                    adjusted_confidence=sig.adjusted_confidence,
                    timestamp=sig.timestamp,
                )
            filtered.append(sig)
        return filtered

    @staticmethod
    def to_json(signal: TradingSignal) -> dict:
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

    @staticmethod
    def signals_to_report(signals: list[TradingSignal]) -> dict:
        """Tüm horizon sinyallerini tek raporda birleştir."""
        return {
            "symbol": signals[0].symbol if signals else "",
            "market": signals[0].market if signals else "",
            "timestamp": signals[0].timestamp if signals else "",
            "horizons": {
                sig.timeframe: {
                    "direction": {-1: "SELL", 0: "HOLD", 1: "BUY"}[sig.direction],
                    "target_price": sig.target_price,
                    "confidence": sig.confidence,
                    "adjusted_confidence": sig.adjusted_confidence,
                }
                for sig in signals
            },
        }
