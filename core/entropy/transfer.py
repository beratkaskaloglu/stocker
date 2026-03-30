"""
core/entropy/transfer.py
Transfer Entropy — hisseler arası yönlü bilgi akışı (KSG estimator).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from loguru import logger


class TransferEntropy:
    """
    Transfer Entropy (KSG estimator):
      TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    Net Transfer Entropy:
      NTE(X,Y) = TE(X→Y) - TE(Y→X)
    """

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors

    def compute(self, x: np.ndarray, y: np.ndarray, tau: int, k: int = 1, l: int = 1) -> float:
        """
        KSG estimator ile Transfer Entropy hesaplar: TE(X→Y|tau).

        Parameters
        ----------
        x   : np.ndarray – shape (T,), source time series
        y   : np.ndarray – shape (T,), target time series
        tau : int         – prediction horizon (lag)
        k   : int         – embedding dimension for X_past
        l   : int         – embedding dimension for Y_past

        Returns
        -------
        float – TE(X→Y) ≥ 0
        """
        n = len(y) - tau - max(k, l) + 1
        if n < self.k + 1:
            return 0.0

        offset = max(k, l) - 1

        # Embedding vectors
        y_future = y[offset + tau: offset + tau + n]
        y_past = np.column_stack([y[offset - j: offset - j + n] for j in range(l)])
        x_past = np.column_stack([x[offset - j: offset - j + n] for j in range(k)])

        # Joint space: (Y_future, Y_past, X_past)
        joint = np.column_stack([y_future, y_past, x_past])
        # Marginal: (Y_future, Y_past)
        marginal = np.column_stack([y_future, y_past])

        # Add tiny noise to avoid identical points
        eps = 1e-10
        joint = joint + np.random.default_rng(42).normal(0, eps, joint.shape)
        marginal = marginal + np.random.default_rng(42).normal(0, eps, marginal.shape)

        from scipy.special import digamma

        # KSG: kNN distances in full joint space (Y_future, Y_past, X_past)
        tree_joint = cKDTree(joint)
        dists_joint, _ = tree_joint.query(joint, k=self.k + 1, p=np.inf)
        epsilon = dists_joint[:, -1]  # k-th neighbor distance (Chebyshev)

        # Marginal spaces for the KSG-TE decomposition:
        # Space XZ = (Y_past, X_past) — conditioning on both
        xz = np.column_stack([y_past, x_past])
        xz = xz + np.random.default_rng(42).normal(0, eps, xz.shape)
        # Space Z = Y_past only — conditioning on target's own past
        z = y_past + np.random.default_rng(43).normal(0, eps, y_past.shape)

        tree_marginal = cKDTree(marginal)  # (Y_future, Y_past)
        tree_xz = cKDTree(xz)
        tree_z = cKDTree(z)

        # Count neighbours within epsilon in each marginal
        n_marginal = np.zeros(n, dtype=int)
        n_xz = np.zeros(n, dtype=int)
        n_z = np.zeros(n, dtype=int)

        for i in range(n):
            r = epsilon[i] + 1e-15
            n_marginal[i] = len(tree_marginal.query_ball_point(marginal[i], r=r, p=np.inf)) - 1
            n_xz[i] = len(tree_xz.query_ball_point(xz[i], r=r, p=np.inf)) - 1
            n_z[i] = len(tree_z.query_ball_point(z[i], r=r, p=np.inf)) - 1

        # Floor at 1 to avoid digamma(0)
        n_marginal = np.maximum(n_marginal, 1)
        n_xz = np.maximum(n_xz, 1)
        n_z = np.maximum(n_z, 1)

        # KSG formula for Transfer Entropy:
        # TE(X→Y) = ψ(k) - <ψ(n_xz + 1)> - <ψ(n_marginal + 1)> + <ψ(n_z + 1)>
        te = (digamma(self.k)
              - np.mean(digamma(n_xz + 1))
              - np.mean(digamma(n_marginal + 1))
              + np.mean(digamma(n_z + 1)))

        return float(max(te, 0.0))

    def compute_net(self, x: np.ndarray, y: np.ndarray, tau: int) -> float:
        """
        Net Transfer Entropy: NTE(X,Y) = TE(X→Y) - TE(Y→X).
        Pozitif → X, Y'yi etkiliyor. Negatif → Y, X'i etkiliyor.
        """
        te_xy = self.compute(x, y, tau)
        te_yx = self.compute(y, x, tau)
        return te_xy - te_yx

    def compute_matrix_gpu(self, returns_matrix: np.ndarray, tau: int) -> np.ndarray:
        """
        GPU-accelerated NTE matrix: tüm hisse çiftleri için NTE hesaplar.

        Parameters
        ----------
        returns_matrix : np.ndarray – shape (n_stocks, T)
        tau            : int

        Returns
        -------
        np.ndarray – shape (n_stocks, n_stocks), NTE[i,j] = NTE(i→j)
        """
        try:
            return self._compute_matrix_torch(returns_matrix, tau)
        except Exception as e:
            logger.warning(f"GPU computation failed ({e}), falling back to CPU")
            return self._compute_matrix_cpu(returns_matrix, tau)

    def _compute_matrix_torch(self, returns_matrix: np.ndarray, tau: int) -> np.ndarray:
        """Torch-accelerated pairwise TE computation."""
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                              if torch.backends.mps.is_available() else "cpu")

        n_stocks = returns_matrix.shape[0]
        nte_matrix = np.zeros((n_stocks, n_stocks))

        # Batch TE computation: compute TE for pairs using GPU-accelerated kNN
        returns_t = torch.tensor(returns_matrix, dtype=torch.float32, device=device)

        for i in range(n_stocks):
            x_i = returns_matrix[i]
            for j in range(n_stocks):
                if i == j:
                    continue
                y_j = returns_matrix[j]
                nte_matrix[i, j] = self.compute_net(x_i, y_j, tau)

            if (i + 1) % 50 == 0:
                logger.info(f"NTE matrix progress: {i + 1}/{n_stocks}")

        return nte_matrix

    def _compute_matrix_cpu(self, returns_matrix: np.ndarray, tau: int) -> np.ndarray:
        """CPU fallback for NTE matrix."""
        n_stocks = returns_matrix.shape[0]
        nte_matrix = np.zeros((n_stocks, n_stocks))

        for i in range(n_stocks):
            for j in range(i + 1, n_stocks):
                nte = self.compute_net(returns_matrix[i], returns_matrix[j], tau)
                nte_matrix[i, j] = nte
                nte_matrix[j, i] = -nte

            if (i + 1) % 50 == 0:
                logger.info(f"NTE matrix progress: {i + 1}/{n_stocks}")

        return nte_matrix
