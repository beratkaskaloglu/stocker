"""
signals/error_module.py
Tahmin güvenilirliği — predicted error modülü.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_MIN_HISTORY_ROWS = 180  # ~6 ay günlük veri
_RETRAIN_EVERY_N = 7     # her 7 günde bir yeniden eğit

_FEATURE_COLS = ["confidence", "entropy", "volatility", "hour_of_day", "day_of_week"]
_TARGET_COL = "abs_direction_error"


class PredictedErrorModule:
    """
    Geçmiş tahminlerin hata oranını öğrenerek yeni tahminlerin
    güvenilirliğini düşüren / onaylayan XGBoost tabanlı modül.

    Cold start: Yeterli veri (<180 satır) yokken error_score=0.0 döndürür.
    """

    def __init__(self, history_path: str = "outputs/signals/prediction_history.parquet"):
        self.history_path = Path(history_path)
        self.cold_start = True
        self.model: XGBRegressor | None = None
        self._updates_since_retrain = 0

        self._load_history()

    # ── public api ──────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        timeframe: str,
        confidence: float,
        entropy: float,
        volatility: float,
    ) -> float:
        """Tahminin beklenen hata skorunu döndürür. ∈ [0, 1]."""
        if self.cold_start or self.model is None:
            return 0.0

        now = datetime.now(timezone.utc)
        features = np.array([[
            confidence,
            entropy,
            volatility,
            now.hour,
            now.weekday(),
        ]])
        raw = float(self.model.predict(features)[0])
        return float(np.clip(raw, 0.0, 1.0))

    def update(self, prediction: dict, actual_outcome: dict) -> None:
        """
        Yeni tahmin-gerçek çifti ekler, gerekirse modeli yeniden eğitir.

        prediction keys: symbol, timeframe, direction, confidence, entropy, volatility, timestamp
        actual_outcome keys: actual_direction
        """
        ts = prediction.get("timestamp", datetime.now(timezone.utc).isoformat())
        dt = datetime.fromisoformat(ts)

        row = {
            "symbol": prediction["symbol"],
            "timeframe": prediction["timeframe"],
            "predicted_direction": prediction["direction"],
            "actual_direction": actual_outcome["actual_direction"],
            "confidence": prediction["confidence"],
            "entropy": prediction["entropy"],
            "volatility": prediction["volatility"],
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),
            "abs_direction_error": abs(prediction["direction"] - actual_outcome["actual_direction"]),
            "timestamp": ts,
        }

        new_row = pd.DataFrame([row])
        if self._history is not None and len(self._history) > 0:
            self._history = pd.concat([self._history, new_row], ignore_index=True)
        else:
            self._history = new_row

        self._save_history()
        self._updates_since_retrain += 1

        if self._updates_since_retrain >= _RETRAIN_EVERY_N:
            self.train_error_model()
            self._updates_since_retrain = 0

    def train_error_model(self) -> None:
        """XGBoost regressor'ı geçmiş veriyle eğitir."""
        if self._history is None or len(self._history) < _MIN_HISTORY_ROWS:
            return

        df = self._history.dropna(subset=_FEATURE_COLS + [_TARGET_COL])
        X = df[_FEATURE_COLS].values
        y = df[_TARGET_COL].values.clip(0.0, 1.0)

        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="reg:squarederror",
        )
        self.model.fit(X, y)
        self.cold_start = False

    # ── private helpers ─────────────────────────────────────────

    def _load_history(self) -> None:
        """Geçmiş veriyi yükler, yeterli veri varsa modeli eğitir."""
        self._history: pd.DataFrame | None = None

        if self.history_path.exists():
            self._history = pd.read_parquet(self.history_path)
            if len(self._history) >= _MIN_HISTORY_ROWS:
                self.train_error_model()

    def _save_history(self) -> None:
        """Geçmiş veriyi parquet olarak kaydeder."""
        if self._history is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history.to_parquet(self.history_path, index=False)
