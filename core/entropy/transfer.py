"""
core/entropy/transfer.py
Transfer Entropy — hisseler arası yönlü bilgi akışı (KSG estimator).
"""
from __future__ import annotations

import numpy as np


class TransferEntropy:
    """
    TE(X→Y): X'teki bilginin Y'yi ne kadar öngördüğü
    TE(Y→X): Y'deki bilginin X'i ne kadar öngördüğü
    NTE(X,Y) = TE(X→Y) - TE(Y→X)

    PSEUDO (KSG Estimator):
    1. compute(x, y, tau, k=1, l=1) → float
       a. Embedding vektörleri oluştur:
          - Y_future : y[tau:]
          - Y_past   : y[:-tau]  (l geçmiş adım)
          - X_past   : x[:-tau]  (k geçmiş adım)
       b. Joint ve marginal uzaklıkları hesapla (kNN, k=5)
       c. KSG formülü: TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
       d. Negatif değerleri 0'a clip (gürültü)
    2. compute_net(x, y, tau) → float
       a. NTE = compute(x→y, tau) - compute(y→x, tau)
    3. compute_matrix_gpu(returns_matrix, tau) → np.ndarray
       a. Shape: (n_stocks, n_stocks) — tüm çiftler
       b. S&P500: 503×503 = 252,009 çift (diyagonal hariç 252,006)
       c. GPU batch hesabı için torch.cdist veya custom CUDA kernel
       NOT: Bu fonksiyon çok ağır — GPU zorunlu
    """

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors

    def compute(self, x: np.ndarray, y: np.ndarray, tau: int, k: int = 1, l: int = 1) -> float:
        # TODO: KSG estimator implement
        raise NotImplementedError

    def compute_net(self, x: np.ndarray, y: np.ndarray, tau: int) -> float:
        # TODO: NTE = TE(x→y) - TE(y→x)
        raise NotImplementedError

    def compute_matrix_gpu(self, returns_matrix: np.ndarray, tau: int) -> np.ndarray:
        # TODO: GPU parallel implementation
        # returns: (n_stocks, n_stocks) NTE matrix
        raise NotImplementedError
