"""
core/features/aggregator.py
Tüm feature kaynaklarını tek tensor'da birleştirir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from core.features.returns import ReturnsCalculator
from core.features.frequency import FrequencyFeatures
from core.features.gasf import GASFEncoder


class FeatureAggregator:
    """
    Tum feature kaynaklarini birlestirip model girisi icin tensor olusturur.

    Feature layout:
    ┌──────────────────────────────────────┐
    │  0-1:   DR, DR²                      │
    │  2-5:   Rolling stats (mean,std,skew,kurt) │
    │  6:     Shannon Entropy H_norm       │
    │  7-16:  Graph features (10)          │
    │  17-80: FFT features (64)            │
    │  81-100: Wavelet features (20)       │
    │  101-130: TA indicators (30)         │
    │  131-134: Sentiment vector (4)       │
    └──────────────────────────────────────┘
    """

    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        self.returns_calc = ReturnsCalculator()
        self.freq_features = FrequencyFeatures(n_fft_components=32, wavelet_level=4)

    def transform(
        self,
        ohlcv_df: pd.DataFrame,
        entropy_features: dict | None = None,
        sentiment_vectors: dict | None = None,
        technical_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        OHLCV + entropy + sentiment + technical → (n_samples, feature_dim) ndarray.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame with columns [open, high, low, close, volume]
        entropy_features : dict with keys 'shannon' (float), 'graph_features' (ndarray(10,))
        sentiment_vectors : dict with key 'vector' (ndarray(4,))
        technical_features : ndarray (n_samples, 30) from TechnicalAgent

        Returns
        -------
        np.ndarray shape (n_samples, feature_dim)
        """
        n = len(ohlcv_df)

        # 1. Returns: DR, DR²
        df = self.returns_calc.compute_all(ohlcv_df)
        dr = df["dr"].values.astype(np.float32)
        dr2 = df["dr2"].values.astype(np.float32)
        rolling_mean = df["rolling_mean"].values.astype(np.float32)
        rolling_std = df["rolling_std"].values.astype(np.float32)
        rolling_skew = df["rolling_skew"].values.astype(np.float32)
        rolling_kurt = df["rolling_kurtosis"].values.astype(np.float32)

        feature_columns = [dr, dr2, rolling_mean, rolling_std, rolling_skew, rolling_kurt]

        # 2. Shannon Entropy (scalar → broadcast)
        if entropy_features and "shannon" in entropy_features:
            shannon = np.full(n, entropy_features["shannon"], dtype=np.float32)
        else:
            shannon = np.zeros(n, dtype=np.float32)
        feature_columns.append(shannon)

        # 3. Graph features (10 values → broadcast)
        if entropy_features and "graph_features" in entropy_features:
            gf = np.asarray(entropy_features["graph_features"], dtype=np.float32)
            graph = np.tile(gf, (n, 1))  # (n, 10)
        else:
            graph = np.zeros((n, 10), dtype=np.float32)

        # 4. FFT + Wavelet (per-window)
        close = df["close"].values.astype(np.float64)
        freq_features_list = []
        for i in range(n):
            start = max(0, i - self.seq_len + 1)
            window = close[start:i + 1]
            if len(window) < 4:
                window = np.pad(window, (4 - len(window), 0), mode="edge")
            freq_features_list.append(self.freq_features.compute_all(window))
        freq_arr = np.stack(freq_features_list, axis=0)  # (n, 84)

        # 5. Technical indicators
        if technical_features is not None:
            tech = np.asarray(technical_features, dtype=np.float32)
            if tech.shape[0] != n:
                logger.warning(f"Technical features shape mismatch: {tech.shape[0]} vs {n}")
                tech = np.zeros((n, 30), dtype=np.float32)
        else:
            tech = np.zeros((n, 30), dtype=np.float32)

        # 6. Sentiment vector (4 values → broadcast)
        if sentiment_vectors and "vector" in sentiment_vectors:
            sv = np.asarray(sentiment_vectors["vector"], dtype=np.float32)
            sent = np.tile(sv, (n, 1))  # (n, 4)
        else:
            sent = np.zeros((n, 4), dtype=np.float32)

        # Stack all: (n, 6) + (n, 1) + (n, 10) + (n, 84) + (n, 30) + (n, 4) = (n, 135)
        scalar_features = np.column_stack(feature_columns)  # (n, 7)
        all_features = np.hstack([scalar_features, graph, freq_arr, tech, sent])

        return all_features.astype(np.float32)

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        """Robust scaling (IQR based) — outlier toleransli."""
        q25 = np.percentile(tensor, 25, axis=0)
        q75 = np.percentile(tensor, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0  # avoid division by zero
        median = np.median(tensor, axis=0)
        scaled = (tensor - median) / iqr
        return np.nan_to_num(scaled, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    def to_sequences(self, tensor: np.ndarray) -> np.ndarray:
        """Sliding window → (n_windows, seq_len, feature_dim)."""
        n, f = tensor.shape
        if n < self.seq_len:
            # Pad at the beginning
            pad = np.zeros((self.seq_len - n, f), dtype=tensor.dtype)
            tensor = np.vstack([pad, tensor])
            n = len(tensor)

        windows = []
        for i in range(self.seq_len, n + 1):
            windows.append(tensor[i - self.seq_len:i])

        return np.stack(windows, axis=0)
