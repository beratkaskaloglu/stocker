"""
core/training/train_all.py
Train all 4 models + meta-learner in sequence — multi-horizon support.
Run: python -m core.training.train_all --market US --epochs 100 --device cuda --data data/sp500_dataset.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from core.models.lstm_attention import LSTMAttentionModel
from core.models.cnn_lstm import CNNLSTMModel
from core.models.transformer import TransformerModel
from core.models.resnet_gasf import ResNetGASFModel
from core.models.meta_learner import MetaLearner
from core.models.horizons import DAILY_HORIZONS
from core.training.trainer import SupervisedTrainer, TrainConfig


# ─── Model factory ──────────────────────────────────────────────────────────

def build_model(name: str, feature_dim: int, horizon_names: list[str] | None = None) -> torch.nn.Module:
    hn = horizon_names or [h.name for h in DAILY_HORIZONS]
    if name == "lstm_attention":
        return LSTMAttentionModel(feature_dim=feature_dim, hidden_size=256,
                                   num_layers=2, num_heads=8, horizon_names=hn)
    elif name == "cnn_lstm":
        return CNNLSTMModel(feature_dim=feature_dim, horizon_names=hn)
    elif name == "transformer":
        return TransformerModel(feature_dim=feature_dim, d_model=256, nhead=4,
                                 num_layers=3, horizon_names=hn)
    elif name == "resnet_gasf":
        return ResNetGASFModel(horizon_names=hn)
    else:
        raise ValueError(f"Unknown model: {name}")


# ─── Dataset ────────────────────────────────────────────────────────────────

class MultiHorizonDataset(Dataset):
    """
    .npz dataset with multi-horizon labels.

    Expected keys in npz:
        X: (N, seq_len, feature_dim)
        y_direction_1d, y_return_1d
        y_direction_15d, y_return_15d
        y_direction_1m, y_return_1m
    """

    def __init__(self, X: np.ndarray, labels: dict[str, tuple], horizon_names: list[str]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.horizon_names = horizon_names
        self.labels = {}
        for h in horizon_names:
            direction, returns = labels[h]
            self.labels[f"direction_{h}"] = torch.tensor(direction, dtype=torch.long)
            self.labels[f"return_{h}"] = torch.tensor(returns, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"features": self.X[idx]}
        for key, tensor in self.labels.items():
            item[key] = tensor[idx]
        return item


class GASFMultiHorizonDataset(Dataset):
    """GASF images + multi-horizon labels."""

    def __init__(self, images: np.ndarray, labels: dict[str, tuple], horizon_names: list[str]):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.horizon_names = horizon_names
        self.labels = {}
        for h in horizon_names:
            direction, returns = labels[h]
            self.labels[f"direction_{h}"] = torch.tensor(direction, dtype=torch.long)
            self.labels[f"return_{h}"] = torch.tensor(returns, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {"features": self.images[idx]}
        for key, tensor in self.labels.items():
            item[key] = tensor[idx]
        return item


def load_dataset(data_path: str) -> tuple[np.ndarray, dict, list[str]]:
    """
    Load dataset from either:
    - .npz file (legacy)
    - directory with .npy files (memory-efficient, for large datasets)
    """
    path = Path(data_path)

    # Directory format (memory-efficient)
    if path.is_dir():
        logger.info(f"Loading directory dataset from {path}/")
        shape = np.load(path / "X_shape.npy")
        n_samples, seq_len, feature_dim = int(shape[0]), int(shape[1]), int(shape[2])

        # Load X as memmap (no RAM spike)
        X = np.memmap(path / "X.npy.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, seq_len, feature_dim))

        horizon_names = list(np.load(path / "horizon_names.npy"))

        labels = {}
        for h in horizon_names:
            y_dir = np.load(path / f"y_direction_{h}.npy")
            y_ret = np.load(path / f"y_return_{h}.npy")
            labels[h] = (y_dir, y_ret)

        return X, labels, horizon_names

    # .npz format (legacy)
    data = np.load(str(path), allow_pickle=True)
    X = data["X"]

    if "horizon_names" in data:
        horizon_names = list(data["horizon_names"])
    else:
        horizon_names = ["1d"]

    labels = {}
    for h in horizon_names:
        dir_key = f"y_direction_{h}"
        ret_key = f"y_return_{h}"
        if dir_key in data:
            labels[h] = (data[dir_key], data[ret_key])
        elif "y_direction" in data:
            labels[h] = (data["y_direction"], data["y_return"])

    return X, labels, horizon_names


def create_synthetic_multi_horizon(
    n_samples: int = 2000, seq_len: int = 60, feature_dim: int = 50,
    horizon_names: list[str] | None = None,
) -> tuple[np.ndarray, dict, list[str]]:
    """Synthetic data for testing."""
    if horizon_names is None:
        horizon_names = ["1d", "15d", "1m"]

    X = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
    labels = {}
    for h in horizon_names:
        labels[h] = (
            np.random.randint(0, 3, size=n_samples).astype(np.int64),
            np.random.randn(n_samples).astype(np.float32) * 0.02,
        )
    return X, labels, horizon_names


# ─── Main training pipeline ─────────────────────────────────────────────────

def train_all_models(
    market: str = "US",
    epochs: int = 100,
    batch_size: int = 64,
    device: str = "auto",
    use_wandb: bool = False,
    data_path: str | None = None,
) -> dict:
    """Train all 4 base models sequentially."""
    logger.info(f"{'='*60}")
    logger.info(f"STOCKER TRAINING — {market} market")
    logger.info(f"{'='*60}")
    start = time.time()

    # ── Load data ──
    if data_path and Path(data_path).exists():
        logger.info(f"Loading data from {data_path}")
        X, labels, horizon_names = load_dataset(data_path)
    else:
        logger.warning("No data_path — using synthetic data for testing")
        X, labels, horizon_names = create_synthetic_multi_horizon()

    feature_dim = X.shape[2]
    seq_len = X.shape[1]
    n_samples = X.shape[0]
    logger.info(f"Data: {n_samples} samples, seq_len={seq_len}, features={feature_dim}")
    logger.info(f"Horizons: {horizon_names}")

    # For large datasets on limited RAM: subsample if needed
    max_samples = 200_000  # ~6GB RAM for (200K, 60, 125) float32
    if n_samples > max_samples:
        logger.warning(f"Dataset too large for RAM ({n_samples}). Subsampling to {max_samples}.")
        # Temporal subsample: take evenly spaced indices to preserve time distribution
        indices = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        X = np.array(X[indices])  # copy from memmap to RAM
        labels = {h: (d[indices], r[indices]) for h, (d, r) in labels.items()}
        n_samples = max_samples
        logger.info(f"  Subsampled: {n_samples} samples")

    # Ensure X is a numpy array (not memmap) for DataLoader
    if isinstance(X, np.memmap):
        logger.info("Converting memmap to array...")
        X = np.array(X)

    # Temporal split
    split_idx = int(n_samples * 0.8)
    labels_train = {h: (d[:split_idx], r[:split_idx]) for h, (d, r) in labels.items()}
    labels_val = {h: (d[split_idx:], r[split_idx:]) for h, (d, r) in labels.items()}

    # ── Train sequence models ──
    results = {}
    sequence_models = ["lstm_attention", "cnn_lstm", "transformer"]

    for model_name in sequence_models:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'─'*40}")

        model = build_model(model_name, feature_dim, horizon_names)
        config = TrainConfig(
            market=market, model_name=model_name, epochs=epochs,
            batch_size=batch_size, device=device, use_wandb=use_wandb,
        )

        train_ds = MultiHorizonDataset(X[:split_idx], labels_train, horizon_names)
        val_ds = MultiHorizonDataset(X[split_idx:], labels_val, horizon_names)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        trainer = SupervisedTrainer(model, config)
        result = trainer.train(train_loader, val_loader)
        results[model_name] = result
        logger.info(f"{model_name} done — best_val_loss={result['best_val_loss']:.4f}")

    # ── ResNet (GASF images) ──
    logger.info(f"\n{'─'*40}")
    logger.info("Training: resnet_gasf")
    logger.info(f"{'─'*40}")

    from core.features.gasf import GASFEncoder
    gasf = GASFEncoder(image_size=64)

    gasf_images = []
    for i in range(n_samples):
        series = X[i, :, 0]  # first feature as price proxy
        img = gasf.encode(series)
        gasf_images.append(gasf.to_rgb(img))
    gasf_features = np.stack(gasf_images, axis=0)

    resnet_model = build_model("resnet_gasf", feature_dim, horizon_names)
    resnet_config = TrainConfig(
        market=market, model_name="resnet_gasf", epochs=epochs,
        batch_size=batch_size, device=device, use_wandb=use_wandb,
    )

    resnet_train = GASFMultiHorizonDataset(gasf_features[:split_idx], labels_train, horizon_names)
    resnet_val = GASFMultiHorizonDataset(gasf_features[split_idx:], labels_val, horizon_names)

    resnet_trainer = SupervisedTrainer(resnet_model, resnet_config)
    result = resnet_trainer.train(
        DataLoader(resnet_train, batch_size=batch_size, shuffle=True),
        DataLoader(resnet_val, batch_size=batch_size, shuffle=False),
    )
    results["resnet_gasf"] = result
    logger.info(f"resnet_gasf done — best_val_loss={result['best_val_loss']:.4f}")

    # ── Summary ──
    elapsed = time.time() - start
    summary = {
        "market": market,
        "horizons": horizon_names,
        "total_time_seconds": elapsed,
        "total_time_minutes": elapsed / 60,
        "models": results,
    }

    summary_path = Path("outputs/models") / market / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL TRAINING COMPLETE — {elapsed/60:.1f} min")
    for name, r in results.items():
        logger.info(f"  {name:20s} val_loss={r['best_val_loss']:.4f} ({r['epochs_trained']} epochs)")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"TensorBoard: tensorboard --logdir outputs/logs/{market}")
    logger.info(f"{'='*60}")

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all Stocker models")
    parser.add_argument("--market", default="US", choices=["US", "BIST"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--data", type=str, default=None, help="Path to .npz data file")
    args = parser.parse_args()

    train_all_models(
        market=args.market,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_wandb=args.wandb,
        data_path=args.data,
    )
