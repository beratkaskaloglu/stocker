"""
core/features/frequency.py
FFT + Wavelet — frekans domain feature extraction.
"""
from __future__ import annotations

import numpy as np


class FrequencyFeatures:
    """
    FFT ve Wavelet ile frekans domain feature extraction.
    FFT: en guclu frekans bilesenleri (amplitude + frequency)
    Wavelet: cok olcekli zaman-frekans temsili (mean, std, energy, entropy)
    """

    def __init__(self, n_fft_components: int = 32, wavelet: str = "db4", wavelet_level: int = 4):
        self.n_fft = n_fft_components
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level

    def fft_features(self, series: np.ndarray) -> np.ndarray:
        """FFT → en guclu n_fft frekans bileseni (amplitude + frequency)."""
        series = np.asarray(series, dtype=np.float64)
        n = len(series)

        # FFT hesapla
        fft_vals = np.fft.fft(series)
        freqs = np.fft.fftfreq(n)

        # Sadece pozitif frekanslari al
        pos_mask = freqs > 0
        fft_vals = fft_vals[pos_mask]
        freqs = freqs[pos_mask]

        # Guc spektrumu
        power = np.abs(fft_vals) ** 2

        # En guclu n_fft bilesenini sec
        n_components = min(self.n_fft, len(power))
        top_indices = np.argsort(power)[::-1][:n_components]

        amplitudes = np.sqrt(power[top_indices])
        frequencies = freqs[top_indices]

        # Normalize
        amp_max = amplitudes.max() if amplitudes.max() > 0 else 1.0
        amplitudes = amplitudes / amp_max

        # Pad if needed
        if n_components < self.n_fft:
            amplitudes = np.pad(amplitudes, (0, self.n_fft - n_components))
            frequencies = np.pad(frequencies, (0, self.n_fft - n_components))

        return np.concatenate([amplitudes, frequencies]).astype(np.float32)

    def wavelet_features(self, series: np.ndarray) -> np.ndarray:
        """Wavelet decomposition → her level icin (mean, std, energy, entropy)."""
        import pywt

        series = np.asarray(series, dtype=np.float64)

        # Wavelet decomposition
        max_level = pywt.dwt_max_level(len(series), self.wavelet)
        level = min(self.wavelet_level, max_level)
        if level < 1:
            level = 1

        coeffs = pywt.wavedec(series, self.wavelet, level=level)

        features = []
        for c in coeffs:
            if len(c) == 0:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue
            features.append(float(np.mean(c)))
            features.append(float(np.std(c)))
            features.append(float(np.sum(c ** 2)))  # energy
            # Wavelet entropy
            p = np.abs(c) / (np.sum(np.abs(c)) + 1e-10)
            features.append(float(-np.sum(p * np.log(p + 1e-10))))

        # Pad/trim to fixed size: (wavelet_level + 1) * 4
        expected = (self.wavelet_level + 1) * 4
        if len(features) < expected:
            features.extend([0.0] * (expected - len(features)))
        elif len(features) > expected:
            features = features[:expected]

        return np.array(features, dtype=np.float32)

    def compute_all(self, series: np.ndarray) -> np.ndarray:
        """FFT + Wavelet features birlestir."""
        fft_f = self.fft_features(series)
        wav_f = self.wavelet_features(series)
        return np.concatenate([fft_f, wav_f])
