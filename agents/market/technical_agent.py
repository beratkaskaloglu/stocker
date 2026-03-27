"""
agents/market/technical_agent.py
Teknik analiz agenti — ta-lib tabanlı indikatörler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent, AgentOutput


class TechnicalAgent(BaseAgent):
    """
    30 teknik indikatör hesaplar → feature vektörü üretir.

    PSEUDO:
    1. analyze(symbol, ohlcv_df) → AgentOutput
       Hesaplanan indikatörler (hepsi normalize edilir):
       ┌─────────────────────────────────────────────────┐
       │ Trend:                                          │
       │   SMA(20), SMA(50), EMA(12), EMA(26)           │
       │   MACD, MACD_signal, MACD_hist                  │
       │   ADX(14), +DI, -DI                             │
       │                                                 │
       │ Momentum:                                       │
       │   RSI(14), Stoch_K, Stoch_D                     │
       │   MFI(14), Williams_R(14)                       │
       │   CCI(14), ROC(10)                              │
       │                                                 │
       │ Volatilite:                                     │
       │   Bollinger_Upper, Bollinger_Mid, Bollinger_Lower│
       │   ATR(14), Bollinger_Width                      │
       │                                                 │
       │ Hacim:                                          │
       │   OBV, VWAP, Volume_MA(20), Volume_Ratio        │
       │                                                 │
       │ Pattern:                                        │
       │   Golden_Cross (SMA20>SMA50), RSI_Oversold      │
       └─────────────────────────────────────────────────┘
       Toplam: 30 feature
    2. normalize_indicators(raw_values) → np.ndarray
       a. Price-based indikatörler: % değişim olarak normalize
       b. Oscillator'lar (RSI, Stoch): zaten 0-100, /100 ile normalize
       c. Volume: log1p transform
    """

    VECTOR_DIM = 30

    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        # TODO: ta library ile tüm indikatörleri hesapla
        raise NotImplementedError

    def _compute_indicators(self, df: pd.DataFrame) -> np.ndarray:
        # TODO: import ta; tüm indikatörleri hesapla
        raise NotImplementedError

    def get_vector_dim(self) -> int:
        return self.VECTOR_DIM
