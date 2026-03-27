"""
core/entropy/shannon.py
Shannon Entropy — her hisse için belirsizlik ölçümü.
"""
from __future__ import annotations

import numpy as np


class ShannonEntropy:
    """
    Shannon Entropy: H(X) = -Σ p(x_i) * ln(p(x_i))
    Normalized: H_norm = H / ln(bins) → [0, 1]
    """

    def __init__(self, bins: int = 50):
        self.bins = bins

    def compute(self, returns: np.ndarray) -> float:
        """
        Tek bir hisse senedinin getiri serisi için normalize Shannon entropy hesaplar.

        Parameters
        ----------
        returns : np.ndarray – shape (T,), log-return veya simple return serisi

        Returns
        -------
        float – H_norm ∈ [0, 1]
               0'a yakın: düşük belirsizlik (tahmin edilebilir)
               1'e yakın: yüksek belirsizlik (uniform dağılım)
        """
        returns = returns[~np.isnan(returns)]
        if len(returns) < 2:
            return 0.0

        # Histogram ile olasılık dağılımı
        counts, _ = np.histogram(returns, bins=self.bins, density=False)
        p = counts / counts.sum()

        # p=0 olan binleri filtrele (log(0) tanımsız)
        p = p[p > 0]

        # H = -Σ p(x_i) * ln(p(x_i))
        h = -np.sum(p * np.log(p))

        # Normalize: H_norm = H / ln(bins) → [0, 1]
        h_max = np.log(self.bins)
        if h_max == 0:
            return 0.0

        return float(h / h_max)

    def compute_batch(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Birden fazla hisse için toplu entropy hesabı.

        Parameters
        ----------
        returns_matrix : np.ndarray – shape (n_stocks, T) veya (T, n_stocks)
                         Her satır bir hissenin getiri serisi.

        Returns
        -------
        np.ndarray – shape (n_stocks,), her hisse için H_norm
        """
        # (T, n_stocks) ise transpose et → (n_stocks, T)
        if returns_matrix.ndim == 2 and returns_matrix.shape[0] > returns_matrix.shape[1]:
            returns_matrix = returns_matrix.T

        n_stocks = returns_matrix.shape[0]
        result = np.zeros(n_stocks)
        for i in range(n_stocks):
            result[i] = self.compute(returns_matrix[i])
        return result
