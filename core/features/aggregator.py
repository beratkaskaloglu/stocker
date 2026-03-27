"""
core/features/aggregator.py
Tüm feature kaynaklarını tek tensor'da birleştirir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureAggregator:
    """
    PSEUDO:
    1. __init__: config'den feature_dim, seq_len al
    2. transform(
           ohlcv_df,
           entropy_features: dict,    # shannon + graph features
           sentiment_vectors: dict,   # FinBERT çıktıları
           timeframe: str
       ) → np.ndarray  shape: (seq_len, feature_dim)

       Sütun sırası:
       ┌─────────────────────────────────────────────┐
       │  0-1:   DR, DR²                             │
       │  2:     Shannon Entropy H_norm              │
       │  3-12:  Graph features (5 × in/out)        │
       │  13-44: FFT features (32)                   │
       │  45-92: Wavelet features (48)               │
       │  93-112: TVP-VAR coefs (20)                 │
       │  113-142: TA-Lib indicators (30)            │
       │  143-206: GASF summary (64 — flattened mean)│
       │  207-210: Sentiment vector (4)              │
       └─────────────────────────────────────────────┘
       Toplam: 211 feature

    3. normalize(tensor) → np.ndarray
       a. Robust scaler (IQR) — outlier toleranslı
       b. NaN → 0 fill
    4. to_sequences(flat_df, seq_len=60) → np.ndarray
       a. Kayan pencere (sliding window)
       b. Output shape: (n_windows, seq_len, feature_dim)
    """

    FEATURE_DIM = 211

    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len

    def transform(
        self,
        ohlcv_df: pd.DataFrame,
        entropy_features: dict,
        sentiment_vectors: dict,
        timeframe: str,
    ) -> np.ndarray:
        # TODO: implement full pipeline
        raise NotImplementedError

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        # TODO: robust scaling
        raise NotImplementedError

    def to_sequences(self, tensor: np.ndarray) -> np.ndarray:
        # TODO: sliding window → (n_windows, seq_len, feature_dim)
        raise NotImplementedError
