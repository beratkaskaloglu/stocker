"""
core/features/returns.py
Günlük getiri ve volatilite hesabı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ReturnsCalculator:
    """
    PSEUDO:
    1. daily_return(prices: pd.Series) → pd.Series
       a. DR(t) = ln(P(t) / P(t-1))
       b. İlk değer NaN → dropna veya 0
    2. volatility_proxy(returns: pd.Series) → pd.Series
       a. DR²(t) = DR(t)²
    3. rolling_stats(returns, window=20) → pd.DataFrame
       a. rolling mean, std, skew, kurtosis
       b. Anlık zaman serisi istatistikleri
    4. compute_all(ohlcv_df) → pd.DataFrame
       a. Close fiyatından DR ve DR² hesapla
       b. Original df'e sütun olarak ekle
    """

    @staticmethod
    def daily_return(prices: pd.Series) -> pd.Series:
        # DR(t) = ln(P(t) / P(t-1))
        return np.log(prices / prices.shift(1))

    @staticmethod
    def volatility_proxy(returns: pd.Series) -> pd.Series:
        return returns ** 2

    @staticmethod
    def rolling_stats(returns: pd.Series, window: int = 20) -> pd.DataFrame:
        # TODO: implement rolling stats
        raise NotImplementedError

    def compute_all(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: implement — DR ve DR² sütunlarını ekle
        raise NotImplementedError
