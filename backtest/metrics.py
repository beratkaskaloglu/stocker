"""
backtest/metrics.py
Performans metrikleri hesaplaması.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    """Yıllık Sharpe Ratio = (mean_return - risk_free) / std * sqrt(252)"""
    excess = returns - risk_free / 252
    return (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    """Sadece negatif volatilite kullanır."""
    excess = returns - risk_free / 252
    downside = excess[excess < 0].std()
    return (excess.mean() / downside) * np.sqrt(252) if downside > 0 else 0.0


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maksimum tepe-dip düşüşü."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()


def win_rate(trades: pd.DataFrame) -> float:
    """Kazançlı işlem oranı."""
    if len(trades) == 0:
        return 0.0
    return (trades["pnl"] > 0).sum() / len(trades)


def annualized_return(total_return: float, n_days: int) -> float:
    """Günlük toplam getiriyi yıllık getiribe çevir."""
    return (1 + total_return) ** (252 / n_days) - 1
