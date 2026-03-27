"""
tests/test_entropy.py
Entropy modülü testleri.
"""
import numpy as np
import pytest
from core.entropy.shannon import ShannonEntropy
from core.entropy.transfer import TransferEntropy


class TestShannonEntropy:
    def test_uniform_distribution_max_entropy(self):
        """Düzgün dağılım maksimum entropy vermelidir."""
        se = ShannonEntropy(bins=10)
        uniform = np.random.uniform(0, 1, 1000)
        # TODO: se.compute(uniform) > 0.8 assert et
        pytest.skip("Not implemented yet")

    def test_constant_series_zero_entropy(self):
        """Sabit seri sıfır entropy vermelidir."""
        se = ShannonEntropy()
        constant = np.ones(100)
        # TODO: se.compute(constant) == 0 assert et
        pytest.skip("Not implemented yet")


class TestTransferEntropy:
    def test_independent_series_near_zero(self):
        """Bağımsız seriler arasında TE ≈ 0 olmalıdır."""
        te = TransferEntropy()
        x = np.random.randn(500)
        y = np.random.randn(500)
        # TODO: |te.compute(x, y, tau=1)| < 0.05 assert et
        pytest.skip("Not implemented yet")

    def test_causal_series_positive_te(self):
        """X'in Y'yi geciktirdiği durumda TE(X→Y) > TE(Y→X) olmalıdır."""
        te = TransferEntropy()
        x = np.random.randn(500)
        y = np.roll(x, 1) + np.random.randn(500) * 0.1  # y = x(t-1) + noise
        # TODO: te.compute(x, y, tau=1) > te.compute(y, x, tau=1) assert et
        pytest.skip("Not implemented yet")
