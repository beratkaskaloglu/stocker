"""
core/entropy/shannon.py
Shannon Entropy — her hisse için belirsizlik ölçümü.
"""
from __future__ import annotations

import numpy as np


class ShannonEntropy:
    """
    Formül: H(X) = -Σ p(x_i) * ln(p(x_i))

    PSEUDO:
    1. compute(returns: np.ndarray, bins=50) → float
       a. histogram ile olasılık dağılımı oluştur (density=True)
       b. p(x_i) = histogram / sum(histogram)
       c. 0 olan p değerlerini filtrele (log(0) = -inf)
       d. H = -sum(p * log(p))
       e. Normalize: H_norm = H / log(bins)   → [0, 1] arası
    2. compute_batch(returns_matrix: np.ndarray) → np.ndarray
       a. Her hisse için compute() çağır (vektörize)
       b. Shape: (n_stocks,)
    3. interpret(H_norm: float) → str
       a. H < 0.3: "düşük belirsizlik — tahmin edilebilir"
       b. H < 0.7: "orta belirsizlik"
       c. H >= 0.7: "yüksek belirsizlik — tahmin güç"
    """

    def __init__(self, bins: int = 50):
        self.bins = bins

    def compute(self, returns: np.ndarray) -> float:
        # TODO: implement
        raise NotImplementedError

    def compute_batch(self, returns_matrix: np.ndarray) -> np.ndarray:
        # TODO: implement — shape (n_stocks,)
        raise NotImplementedError
