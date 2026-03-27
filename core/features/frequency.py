"""
core/features/frequency.py
FFT + Wavelet — frekans domain feature extraction.
"""
from __future__ import annotations

import numpy as np


class FrequencyFeatures:
    """
    PSEUDO:
    1. fft_features(series: np.ndarray, n_components=32) → np.ndarray
       a. np.fft.fft(series)
       b. Güç spektrumu: |FFT|²
       c. En güçlü n_components frekans bileşenini al
       d. [amplitude, frequency, phase] tuple'larını döndür
       e. Output shape: (n_components * 2,) — amp + freq
    2. wavelet_features(series: np.ndarray, wavelet='db4', level=4) → np.ndarray
       a. pywt.wavedec(series, wavelet, level=level)
       b. Her decomposition level için: mean, std, energy, entropy
       c. Çok ölçekli (multi-scale) zaman-frekans temsili
       d. Output shape: (level * 4,)
    3. compute_all(series) → np.ndarray
       a. FFT features + Wavelet features birleştir
       b. Output shape: (n_components*2 + level*4,)
    """

    def __init__(self, n_fft_components: int = 32, wavelet: str = "db4", wavelet_level: int = 4):
        self.n_fft = n_fft_components
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level

    def fft_features(self, series: np.ndarray) -> np.ndarray:
        # TODO: implement
        raise NotImplementedError

    def wavelet_features(self, series: np.ndarray) -> np.ndarray:
        # TODO: implement using PyWavelets
        raise NotImplementedError

    def compute_all(self, series: np.ndarray) -> np.ndarray:
        # TODO: concat fft + wavelet features
        raise NotImplementedError
