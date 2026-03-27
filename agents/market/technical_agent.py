"""
agents/market/technical_agent.py
Teknik analiz agenti — ta (Technical Analysis Library) tabanlı 30 indikatör.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import ta

from agents.base_agent import AgentOutput, BaseAgent

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    """
    30 teknik indikatör hesaplar → normalize edilmiş feature vektörü üretir.

    Vektör sırası (aggregator ile uyumlu — index 113-142):
        Trend       (0-9):   SMA20, SMA50, EMA12, EMA26, MACD, MACD_signal,
                              MACD_hist, ADX, +DI, -DI
        Momentum   (10-16):  RSI, Stoch_K, Stoch_D, MFI, Williams_R, CCI, ROC
        Volatilite (17-21):  BB_Upper, BB_Mid, BB_Lower, ATR, BB_Width
        Hacim      (22-25):  OBV, VWAP, Volume_MA20, Volume_Ratio
        Pattern    (26-29):  Golden_Cross, Death_Cross, RSI_Oversold, RSI_Overbought
    """

    VECTOR_DIM = 30

    INDICATOR_NAMES = [
        # Trend (10)
        "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "adx", "di_plus", "di_minus",
        # Momentum (7)
        "rsi", "stoch_k", "stoch_d", "mfi",
        "williams_r", "cci", "roc",
        # Volatilite (5)
        "bb_upper", "bb_mid", "bb_lower", "atr", "bb_width",
        # Hacim (4)
        "obv", "vwap", "volume_ma_20", "volume_ratio",
        # Pattern (4)
        "golden_cross", "death_cross", "rsi_oversold", "rsi_overbought",
    ]

    def __init__(
        self,
        market: str = "US",
        timeframe: str = "1d",
        config: dict | None = None,
    ):
        super().__init__(market, timeframe, config or {})

    # ── Public API ──────────────────────────────────────────────────

    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentOutput:
        """
        OHLCV DataFrame üzerinden 30 teknik indikatör hesaplar.

        Parameters
        ----------
        data : DataFrame
            Kolonlar: 'open', 'high', 'low', 'close', 'volume'
            Index: datetime (sıralı)
        """
        if data.empty or len(data) < 50:
            return self._empty_output(symbol)

        raw = self._compute_indicators(data)
        vector = self._normalize(raw, data)

        return AgentOutput(
            symbol=symbol,
            market=self.market,
            timeframe=self.timeframe,
            vector=vector,
            metadata={
                "agent": "technical",
                "n_bars": len(data),
                "indicators": dict(zip(self.INDICATOR_NAMES, vector.tolist())),
            },
        )

    # ── Indicator computation ───────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> dict[str, float]:
        """Son bar için tüm indikatörleri hesaplar."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        last = len(df) - 1

        # ── Trend ──────────────────────────────────────────
        sma_20 = ta.trend.sma_indicator(close, window=20).iloc[last]
        sma_50 = ta.trend.sma_indicator(close, window=50).iloc[last]
        ema_12 = ta.trend.ema_indicator(close, window=12).iloc[last]
        ema_26 = ta.trend.ema_indicator(close, window=26).iloc[last]

        macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_val = macd_ind.macd().iloc[last]
        macd_signal = macd_ind.macd_signal().iloc[last]
        macd_hist = macd_ind.macd_diff().iloc[last]

        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        adx = adx_ind.adx().iloc[last]
        di_plus = adx_ind.adx_pos().iloc[last]
        di_minus = adx_ind.adx_neg().iloc[last]

        # ── Momentum ───────────────────────────────────────
        rsi = ta.momentum.rsi(close, window=14).iloc[last]

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stoch_k = stoch.stoch().iloc[last]
        stoch_d = stoch.stoch_signal().iloc[last]

        mfi = ta.volume.money_flow_index(high, low, close, volume, window=14).iloc[last]
        williams_r = ta.momentum.williams_r(high, low, close, lbp=14).iloc[last]
        cci = ta.trend.cci(high, low, close, window=14).iloc[last]
        roc = ta.momentum.roc(close, window=10).iloc[last]

        # ── Volatilite ─────────────────────────────────────
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[last]
        bb_mid = bb.bollinger_mavg().iloc[last]
        bb_lower = bb.bollinger_lband().iloc[last]
        bb_width = bb.bollinger_wband().iloc[last]

        atr = ta.volatility.average_true_range(high, low, close, window=14).iloc[last]

        # ── Hacim ──────────────────────────────────────────
        obv = ta.volume.on_balance_volume(close, volume).iloc[last]
        vwap = self._compute_vwap(df)
        volume_ma_20 = volume.rolling(window=20).mean().iloc[last]
        volume_ratio = (
            volume.iloc[last] / volume_ma_20
            if volume_ma_20 and volume_ma_20 > 0
            else 1.0
        )

        # ── Pattern flags ──────────────────────────────────
        sma20_series = ta.trend.sma_indicator(close, window=20)
        sma50_series = ta.trend.sma_indicator(close, window=50)

        golden_cross = float(
            sma20_series.iloc[last] > sma50_series.iloc[last]
            and sma20_series.iloc[last - 1] <= sma50_series.iloc[last - 1]
        ) if last > 0 else 0.0

        death_cross = float(
            sma20_series.iloc[last] < sma50_series.iloc[last]
            and sma20_series.iloc[last - 1] >= sma50_series.iloc[last - 1]
        ) if last > 0 else 0.0

        rsi_oversold = float(rsi < 30) if not np.isnan(rsi) else 0.0
        rsi_overbought = float(rsi > 70) if not np.isnan(rsi) else 0.0

        return {
            "sma_20": sma_20, "sma_50": sma_50,
            "ema_12": ema_12, "ema_26": ema_26,
            "macd": macd_val, "macd_signal": macd_signal, "macd_hist": macd_hist,
            "adx": adx, "di_plus": di_plus, "di_minus": di_minus,
            "rsi": rsi, "stoch_k": stoch_k, "stoch_d": stoch_d,
            "mfi": mfi, "williams_r": williams_r, "cci": cci, "roc": roc,
            "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
            "atr": atr, "bb_width": bb_width,
            "obv": obv, "vwap": vwap,
            "volume_ma_20": volume_ma_20, "volume_ratio": volume_ratio,
            "golden_cross": golden_cross, "death_cross": death_cross,
            "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought,
        }

    # ── Normalization ───────────────────────────────────────────────

    def _normalize(self, raw: dict[str, float], df: pd.DataFrame) -> np.ndarray:
        """
        İndikatörleri normalize eder:
        - Price-based (SMA, EMA, BB): son close'a göre % değişim
        - Oscillator (RSI, Stoch, MFI): /100
        - Volume: log1p transform
        - CCI, Williams: özel scaling
        - Pattern flags: zaten 0/1
        """
        close_last = df["close"].iloc[-1]

        def pct(val: float) -> float:
            """Price-relative normalization."""
            if np.isnan(val) or close_last == 0:
                return 0.0
            return (val - close_last) / close_last

        def scale100(val: float) -> float:
            """0-100 range → 0-1."""
            return val / 100.0 if not np.isnan(val) else 0.0

        def safe(val: float) -> float:
            return 0.0 if np.isnan(val) else float(val)

        vector = np.array([
            # Trend (price-based → % diff)
            pct(raw["sma_20"]),
            pct(raw["sma_50"]),
            pct(raw["ema_12"]),
            pct(raw["ema_26"]),
            safe(raw["macd"]) / close_last if close_last else 0.0,
            safe(raw["macd_signal"]) / close_last if close_last else 0.0,
            safe(raw["macd_hist"]) / close_last if close_last else 0.0,
            scale100(raw["adx"]),
            scale100(raw["di_plus"]),
            scale100(raw["di_minus"]),
            # Momentum
            scale100(raw["rsi"]),
            scale100(raw["stoch_k"]),
            scale100(raw["stoch_d"]),
            scale100(raw["mfi"]),
            safe(raw["williams_r"]) / 100.0,  # Williams R: -100..0 → -1..0
            np.clip(safe(raw["cci"]) / 300.0, -1.0, 1.0),  # CCI: ~±300 → ±1
            safe(raw["roc"]) / 100.0,
            # Volatilite
            pct(raw["bb_upper"]),
            pct(raw["bb_mid"]),
            pct(raw["bb_lower"]),
            safe(raw["atr"]) / close_last if close_last else 0.0,
            safe(raw["bb_width"]),  # zaten oran
            # Hacim
            np.log1p(abs(safe(raw["obv"]))) * np.sign(safe(raw["obv"])),
            pct(raw["vwap"]),
            np.log1p(safe(raw["volume_ma_20"])),
            safe(raw["volume_ratio"]) - 1.0,  # 1.0 = ortalama → 0 merkez
            # Pattern flags (0 veya 1)
            safe(raw["golden_cross"]),
            safe(raw["death_cross"]),
            safe(raw["rsi_oversold"]),
            safe(raw["rsi_overbought"]),
        ], dtype=np.float32)

        # NaN temizliği (son savunma hattı)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return vector

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> float:
        """Basit VWAP hesabı (session-based)."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].astype(float)
        cumvol = vol.cumsum()
        cumtp = (typical_price * vol).cumsum()
        vwap_series = cumtp / cumvol.replace(0, np.nan)
        return vwap_series.iloc[-1] if not vwap_series.empty else 0.0

    def _empty_output(self, symbol: str) -> AgentOutput:
        """Yetersiz veri durumunda sıfır vektör döndürür."""
        return AgentOutput(
            symbol=symbol,
            market=self.market,
            timeframe=self.timeframe,
            vector=np.zeros(self.VECTOR_DIM, dtype=np.float32),
            metadata={"agent": "technical", "n_bars": 0, "error": "insufficient_data"},
        )

    def get_vector_dim(self) -> int:
        return self.VECTOR_DIM
