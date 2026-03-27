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
        # TODO: implement GASF encoding
        raise NotImplementedError

    def encode_batch(self, series_matrix: np.ndarray) -> np.ndarray:
        # TODO: implement batch encoding
        raise NotImplementedError

    def to_rgb(self, gasf: np.ndarray) -> np.ndarray:
        # (H, W) → (3, H, W) — aynı kanalı 3 kez tekrarla
        return np.stack([gasf, gasf, gasf], axis=0)
