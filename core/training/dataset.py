"""
core/training/dataset.py
PyTorch Dataset for supervised model training.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StockerDataset(Dataset):
    """
    Trading dataset for supervised training.

    Expects:
        features:  np.ndarray (N, seq_len, feature_dim)
        targets:   dict with keys:
            direction:  np.ndarray (N,)     — {0, 1, 2} = {down, hold, up}
            price:      np.ndarray (N, 1)   — normalized target price
            confidence: np.ndarray (N, 1)   — [0, 1]
    """

    def __init__(self, features: np.ndarray, targets: dict[str, np.ndarray]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.direction = torch.tensor(targets["direction"], dtype=torch.long)
        self.price = torch.tensor(targets["price"], dtype=torch.float32)
        self.confidence = torch.tensor(targets["confidence"], dtype=torch.float32)

        if self.price.ndim == 1:
            self.price = self.price.unsqueeze(1)
        if self.confidence.ndim == 1:
            self.confidence = self.confidence.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "direction": self.direction[idx],
            "price": self.price[idx],
            "confidence": self.confidence[idx],
        }


def create_dataloaders(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    batch_size: int = 64,
    val_split: float = 0.2,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders with temporal split (no leakage)."""
    n = len(features)
    split_idx = int(n * (1 - val_split))

    train_ds = StockerDataset(features[:split_idx], {
        "direction": targets["direction"][:split_idx],
        "price": targets["price"][:split_idx],
        "confidence": targets["confidence"][:split_idx],
    })
    val_ds = StockerDataset(features[split_idx:], {
        "direction": targets["direction"][split_idx:],
        "price": targets["price"][split_idx:],
        "confidence": targets["confidence"][split_idx:],
    })

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_synthetic_data(
    n_samples: int = 1000,
    seq_len: int = 60,
    feature_dim: int = 50,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic data for testing the training pipeline."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((n_samples, seq_len, feature_dim)).astype(np.float32)
    targets = {
        "direction": rng.integers(0, 3, size=(n_samples,)),
        "price": rng.standard_normal((n_samples, 1)).astype(np.float32),
        "confidence": rng.random((n_samples, 1)).astype(np.float32),
    }
    return features, targets
