"""
core/features/returns.py
Günlük getiri ve volatilite hesabı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ReturnsCalculator:
    """
    DR(t) = ln(P(t) / P(t-1))  — log daily return
    DR²(t) = DR(t)²             — volatility proxy
    Rolling stats: mean, std, skew, kurtosis
    """

    @staticmethod
    def daily_return(prices: pd.Series) -> pd.Series:
        """DR(t) = ln(P(t) / P(t-1))"""
        return np.log(prices / prices.shift(1))

    @staticmethod
    def volatility_proxy(returns: pd.Series) -> pd.Series:
        """DR²(t) = DR(t)²"""
        return returns ** 2

    @staticmethod
    def rolling_stats(returns: pd.Series, window: int = 20) -> pd.DataFrame:
        """Rolling mean, std, skew, kurtosis."""
        rolling = returns.rolling(window=window, min_periods=1)
        return pd.DataFrame({
            "rolling_mean": rolling.mean(),
            "rolling_std": rolling.std(),
            "rolling_skew": rolling.apply(lambda x: x.skew() if len(x) >= 3 else 0.0, raw=False),
            "rolling_kurtosis": rolling.apply(lambda x: x.kurtosis() if len(x) >= 4 else 0.0, raw=False),
        })

    def compute_all(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Close fiyatindan DR, DR², rolling stats hesapla ve df'e ekle."""
        df = ohlcv_df.copy()
        dr = self.daily_return(df["close"])
        df["dr"] = dr.fillna(0.0)
        df["dr2"] = self.volatility_proxy(dr).fillna(0.0)

        stats = self.rolling_stats(dr.fillna(0.0))
        for col in stats.columns:
            df[col] = stats[col].fillna(0.0)

        return df
