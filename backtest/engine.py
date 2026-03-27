"""
backtest/engine.py
Walk-forward backtest motoru.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    market: str
    start: str
    end: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series


class BacktestEngine:
    """
    PSEUDO:
    1. run(market, start, end, strategy='ensemble') → BacktestResult
       a. Veriyi train/val split'e ayır (walk-forward)
       b. Her pencere için:
          - Feature extract
          - Model predict
          - RL aksiyonu uygula
          - Portföy güncelle (trading cost dahil)
       c. Equity curve oluştur
       d. Metrikleri hesapla (metrics.py)
    2. walk_forward_splits(dates, window=252, step=21):
       a. Generator: (train_start, train_end, val_start, val_end)
    3. _apply_trade(portfolio, signal, price, cost=0.001):
       a. Stop-loss kontrolü
       b. Take-profit kontrolü
       c. Pozisyon güncelle
       d. Commission düş
    4. compare_benchmark(result, benchmark='SPY' or 'XU100') → dict
       a. Alpha, beta, information ratio hesapla
    """

    def __init__(self, initial_capital: float = 100_000, trading_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost

    def run(self, market: str, start: str, end: str) -> BacktestResult:
        # TODO: implement walk-forward backtest
        raise NotImplementedError

    def walk_forward_splits(self, dates: pd.DatetimeIndex, window: int = 252, step: int = 21):
        # TODO: generator — (train_start, train_end, val_start, val_end)
        raise NotImplementedError
