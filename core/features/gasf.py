"""
core/features/gasf.py
Gramian Angular Summation Field — zaman serisi → 2D image.
"""
from __future__ import annotations

import numpy as np


class GASFEncoder:
    """
    GASF, zaman serisi verilerini görüntüye dönüştürür.
    ResNet bu görüntüleri girdi olarak alır.

    PSEUDO:
    1. encode(series: np.ndarray, image_size=64) → np.ndarray
       a. Normalize: x_norm = (x - min) / (max - min) → [-1, 1] arası
       b. Açıya dönüştür: phi = arccos(x_norm)
       c. GASF matrisi: G[i,j] = cos(phi_i + phi_j)
       d. Resize to (image_size, image_size) gerekirse
       e. Output shape: (image_size, image_size)
    2. encode_batch(series_matrix: np.ndarray) → np.ndarray
       a. Her hisse için encode() çağır
       b. Output shape: (n_stocks, image_size, image_size)
    3. to_rgb(gasf: np.ndarray) → np.ndarray
       a. Tek kanallı → 3 kanal (ResNet RGB girdisi için)
       b. Output shape: (3, image_size, image_size)

    NOT: pyts kütüphanesi kullanılabilir:
         from pyts.image import GramianAngularField
         Ancak custom impl. daha hızlı (numpy vektörizasyon)
    """

    def __init__(self, image_size: int = 64):
        self.image_size = image_size

    def encode(self, series: np.ndarray) -> np.ndarray:
        """Zaman serisi → GASF 2D image (image_size × image_size)."""
        series = np.asarray(series, dtype=np.float64)
        # Normalize to [-1, 1]
        _min, _max = series.min(), series.max()
        if _max - _min < 1e-12:
            x_norm = np.zeros_like(series)
        else:
            x_norm = 2.0 * (series - _min) / (_max - _min) - 1.0
        x_norm = np.clip(x_norm, -1.0, 1.0)

        # Resample to image_size if needed
        if len(x_norm) != self.image_size:
            indices = np.linspace(0, len(x_norm) - 1, self.image_size).astype(int)
            x_norm = x_norm[indices]

        # Angular encoding
        phi = np.arccos(x_norm)

        # GASF: G[i,j] = cos(phi_i + phi_j)
        gasf = np.cos(phi[:, None] + phi[None, :])
        return gasf.astype(np.float32)

    def encode_batch(self, series_matrix: np.ndarray) -> np.ndarray:
        """Shape (n_stocks, T) → (n_stocks, image_size, image_size)."""
        return np.stack([self.encode(s) for s in series_matrix], axis=0)

    def to_rgb(self, gasf: np.ndarray) -> np.ndarray:
        # (H, W) → (3, H, W) — aynı kanalı 3 kez tekrarla
        return np.stack([gasf, gasf, gasf], axis=0)
