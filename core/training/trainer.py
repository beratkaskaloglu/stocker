"""
core/training/trainer.py
Supervised training loop — TensorBoard + wandb monitoring.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class TrainConfig:
    """Training configuration."""
    market: str = "US"
    model_name: str = "lstm_attention"
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15          # early stopping
    grad_clip: float = 1.0
    device: str = "auto"
    save_dir: str = "outputs/models"
    log_dir: str = "outputs/logs"
    use_wandb: bool = False
    wandb_project: str = "stocker"
    loss_weights: dict = field(default_factory=lambda: {
        "direction": 1.0,
        "price": 0.3,
        "confidence": 0.1,
    })


class SupervisedTrainer:
    """
    Full supervised training loop with:
    - TensorBoard logging (always)
    - Weights & Biases logging (optional)
    - Early stopping
    - Gradient clipping
    - Model checkpointing (best + periodic)
    - Learning rate scheduling
    - Per-epoch metrics JSON export
    """

    def __init__(self, model: nn.Module, config: TrainConfig,
                 symbol_names: list[str] | None = None):
        self.model = model
        self.config = config
        self.symbol_names = symbol_names  # index → symbol name mapping

        # Device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)
        logger.info(f"Device: {self.device}")

        # Optimizer + scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=7,
        )

        # Loss functions
        self.loss_dir = nn.CrossEntropyLoss()
        self.loss_price = nn.MSELoss()
        self.loss_conf = nn.BCELoss()

        # Directories
        self.save_dir = Path(config.save_dir) / config.market / config.model_name
        self.log_dir = Path(config.log_dir) / config.market / config.model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # wandb (optional)
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=f"{config.market}-{config.model_name}",
                    config={
                        "market": config.market,
                        "model": config.model_name,
                        "epochs": config.epochs,
                        "batch_size": config.batch_size,
                        "lr": config.learning_rate,
                        "device": str(self.device),
                    },
                )
                logger.info("wandb initialized")
            except ImportError:
                logger.warning("wandb not installed, skipping")

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: list[dict] = []

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
    ) -> dict:
        """Run full training loop. Returns training history."""
        logger.info(
            f"Training {self.config.model_name} on {self.config.market} "
            f"for {self.config.epochs} epochs"
        )
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            # --- Train ---
            train_metrics = self._train_epoch(train_loader, epoch)

            # --- Validate ---
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate(val_loader, epoch)

            # --- Log ---
            self._log_epoch(epoch, train_metrics, val_metrics)

            # --- Early stopping ---
            if val_loader is not None:
                val_loss = val_metrics["total_loss"]
                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  New best val_loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # --- Periodic checkpoint ---
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        # Final save
        self._save_checkpoint(epoch, is_best=False, tag="final")
        self.writer.close()

        if self.wandb_run:
            import wandb
            wandb.finish()

        # Export history
        history_path = self.log_dir / "training_history.json"
        history_path.write_text(json.dumps(self.history, indent=2))

        return {
            "epochs_trained": epoch,
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed,
            "history_path": str(history_path),
            "model_path": str(self.save_dir / "best.pt"),
            "tensorboard_dir": str(self.log_dir),
        }

    def _compute_loss(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute multi-horizon loss.

        out: model output — either multi-horizon dict or flat dict (backward compat)
            Multi-horizon: {"1d": {"direction_logits", "price", "confidence"}, "15d": ..., "1m": ...}
            Flat: {"direction_logits", "price", "confidence"}

        batch: data batch with keys like "direction_1d", "return_1d", etc.
               or legacy keys "direction", "price", "confidence"
        """
        w = self.config.loss_weights
        total_loss = torch.tensor(0.0, device=self.device)
        components = {}

        # Detect if model output is multi-horizon
        first_val = next(iter(out.values()))
        is_multi_horizon = isinstance(first_val, dict)

        if is_multi_horizon:
            horizons = list(out.keys())
            for h in horizons:
                h_out = out[h]
                # Try multi-horizon batch keys first, then fallback
                y_dir_key = f"direction_{h}"
                y_ret_key = f"return_{h}"

                if y_dir_key in batch:
                    y_dir = batch[y_dir_key].to(self.device)
                    y_ret = batch[y_ret_key].to(self.device).unsqueeze(-1)
                elif "direction" in batch:
                    # Fallback: single horizon batch (backward compat)
                    y_dir = batch["direction"].to(self.device)
                    y_ret = batch["price"].to(self.device)
                else:
                    continue

                l_dir = self.loss_dir(h_out["direction_logits"], y_dir)
                l_price = self.loss_price(h_out["price"], y_ret)

                # Confidence target: 1.0 if prediction correct, 0.0 otherwise
                with torch.no_grad():
                    pred_dir = h_out["direction_logits"].argmax(dim=1)
                    y_conf = (pred_dir == y_dir).float().unsqueeze(-1)
                l_conf = self.loss_conf(h_out["confidence"], y_conf)

                h_loss = w["direction"] * l_dir + w["price"] * l_price + w["confidence"] * l_conf
                total_loss = total_loss + h_loss

                components[f"{h}_direction"] = l_dir.item()
                components[f"{h}_price"] = l_price.item()

            total_loss = total_loss / len(horizons)  # average across horizons
        else:
            # Legacy flat output
            y_dir = batch["direction"].to(self.device)
            y_price = batch["price"].to(self.device)
            y_conf = batch["confidence"].to(self.device)

            l_dir = self.loss_dir(out["direction_logits"], y_dir)
            l_price = self.loss_price(out["price"], y_price)
            l_conf = self.loss_conf(out["confidence"], y_conf)
            total_loss = w["direction"] * l_dir + w["price"] * l_price + w["confidence"] * l_conf

            components["direction"] = l_dir.item()
            components["price"] = l_price.item()
            components["confidence"] = l_conf.item()

        return total_loss, components

    def _get_accuracy(self, out: dict, batch: dict) -> tuple[int, int]:
        """Get direction accuracy (uses 1d horizon if multi-horizon)."""
        first_val = next(iter(out.values()))
        is_multi_horizon = isinstance(first_val, dict)

        if is_multi_horizon:
            # Use first horizon for accuracy reporting
            h = list(out.keys())[0]
            logits = out[h]["direction_logits"]
            y_dir_key = f"direction_{h}"
            if y_dir_key in batch:
                y_dir = batch[y_dir_key].to(self.device)
            else:
                y_dir = batch["direction"].to(self.device)
        else:
            logits = out["direction_logits"]
            y_dir = batch["direction"].to(self.device)

        preds = logits.argmax(dim=1)
        correct = (preds == y_dir).sum().item()
        total = y_dir.size(0)
        return correct, total

    def _train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> dict:
        """Single training epoch — supports multi-horizon output."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        grad_norm = 0.0

        for batch in loader:
            x = batch["features"].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)

            loss, components = self._compute_loss(out, batch)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            c, t = self._get_accuracy(out, batch)
            correct += c
            total += t
            n_batches += 1

        return {
            "total_loss": total_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _validate(self, loader: torch.utils.data.DataLoader, epoch: int) -> dict:
        """Validation pass with per-class and per-symbol metrics."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        # Per-class tracking: 0=Down, 1=Hold, 2=Up
        class_names = ["down", "hold", "up"]
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        class_predicted = [0, 0, 0]

        # Per-symbol tracking (all horizons)
        has_symbols = self.symbol_names is not None
        if has_symbols:
            n_syms = len(self.symbol_names)
            # Per horizon per symbol: {horizon: {sym_idx: {correct, total}}}
            sym_horizon_correct = {}
            sym_horizon_total = {}

        for batch in loader:
            x = batch["features"].to(self.device)
            out = self.model(x)

            loss, _ = self._compute_loss(out, batch)
            total_loss += loss.item()

            c, t = self._get_accuracy(out, batch)
            correct += c
            total += t
            n_batches += 1

            # Detect multi-horizon
            first_val = next(iter(out.values()))
            is_multi = isinstance(first_val, dict)

            if is_multi:
                horizons = list(out.keys())
                # Per-class stats (first horizon only)
                h0 = horizons[0]
                logits_h0 = out[h0]["direction_logits"]
                y_key = f"direction_{h0}"
                y_dir_h0 = batch[y_key].to(self.device) if y_key in batch else batch["direction"].to(self.device)
                preds_h0 = logits_h0.argmax(dim=1)

                for cls in range(3):
                    mask_actual = (y_dir_h0 == cls)
                    mask_pred = (preds_h0 == cls)
                    class_total[cls] += mask_actual.sum().item()
                    class_predicted[cls] += mask_pred.sum().item()
                    class_correct[cls] += (mask_actual & mask_pred).sum().item()

                # Per-symbol per-horizon accuracy
                if has_symbols and "symbol_idx" in batch:
                    sym_idx = batch["symbol_idx"]
                    for h in horizons:
                        if h not in sym_horizon_correct:
                            sym_horizon_correct[h] = np.zeros(n_syms, dtype=np.int64)
                            sym_horizon_total[h] = np.zeros(n_syms, dtype=np.int64)

                        h_logits = out[h]["direction_logits"]
                        h_y_key = f"direction_{h}"
                        if h_y_key in batch:
                            h_y = batch[h_y_key].to(self.device)
                        else:
                            continue
                        h_preds = h_logits.argmax(dim=1)
                        h_correct = (h_preds == h_y).cpu().numpy()
                        h_sym = sym_idx.numpy()

                        for s in range(n_syms):
                            mask = (h_sym == s)
                            if mask.any():
                                sym_horizon_total[h][s] += mask.sum()
                                sym_horizon_correct[h][s] += h_correct[mask].sum()
            else:
                logits = out["direction_logits"]
                y_dir = batch["direction"].to(self.device)
                preds = logits.argmax(dim=1)
                for cls in range(3):
                    mask_actual = (y_dir == cls)
                    mask_pred = (preds == cls)
                    class_total[cls] += mask_actual.sum().item()
                    class_predicted[cls] += mask_pred.sum().item()
                    class_correct[cls] += (mask_actual & mask_pred).sum().item()

        per_class = {}
        for c, name in enumerate(class_names):
            precision = class_correct[c] / max(class_predicted[c], 1)
            recall = class_correct[c] / max(class_total[c], 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            per_class[name] = {
                "precision": precision, "recall": recall, "f1": f1,
                "actual_count": class_total[c], "predicted_count": class_predicted[c],
            }

        # Build per-symbol results
        per_symbol = {}
        if has_symbols and sym_horizon_correct:
            for h in sym_horizon_correct:
                for s in range(n_syms):
                    t_count = int(sym_horizon_total[h][s])
                    if t_count == 0:
                        continue
                    c_count = int(sym_horizon_correct[h][s])
                    sym_name = self.symbol_names[s]
                    if sym_name not in per_symbol:
                        per_symbol[sym_name] = {}
                    per_symbol[sym_name][h] = {
                        "accuracy": c_count / t_count,
                        "correct": c_count,
                        "total": t_count,
                    }

        return {
            "total_loss": total_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
            "per_class": per_class,
            "per_symbol": per_symbol,
        }

    def _log_epoch(self, epoch: int, train: dict, val: dict) -> None:
        """Log to TensorBoard, wandb, console, and history."""
        # Console
        msg = f"Epoch {epoch:3d} | train_loss={train['total_loss']:.4f} acc={train['accuracy']:.3f}"
        if val:
            msg += f" | val_loss={val['total_loss']:.4f} acc={val['accuracy']:.3f}"
        msg += f" | lr={train['lr']:.2e}"
        logger.info(msg)

        # Per-class detail (every 10 epochs)
        if val and "per_class" in val and (epoch % 10 == 0 or epoch == self.config.epochs):
            for cls_name, cls_data in val["per_class"].items():
                logger.info(
                    f"    {cls_name:5s}: P={cls_data['precision']:.3f} "
                    f"R={cls_data['recall']:.3f} F1={cls_data['f1']:.3f} "
                    f"(pred={cls_data['predicted_count']} actual={cls_data['actual_count']})"
                )

        # TensorBoard
        self.writer.add_scalar("Loss/train", train["total_loss"], epoch)
        self.writer.add_scalar("Accuracy/train", train["accuracy"], epoch)
        self.writer.add_scalar("Training/lr", train["lr"], epoch)
        self.writer.add_scalar("Training/grad_norm", train.get("grad_norm", 0), epoch)

        if val:
            self.writer.add_scalar("Loss/val", val["total_loss"], epoch)
            self.writer.add_scalar("Accuracy/val", val["accuracy"], epoch)
            if "per_class" in val:
                for cls_name, cls_data in val["per_class"].items():
                    self.writer.add_scalar(f"Precision/{cls_name}", cls_data["precision"], epoch)
                    self.writer.add_scalar(f"Recall/{cls_name}", cls_data["recall"], epoch)
                    self.writer.add_scalar(f"F1/{cls_name}", cls_data["f1"], epoch)

        # wandb
        if self.wandb_run:
            import wandb
            log_data = {f"train/{k}": v for k, v in train.items() if not isinstance(v, dict)}
            if val:
                log_data.update({f"val/{k}": v for k, v in val.items() if not isinstance(v, dict)})
                if "per_class" in val:
                    for cls_name, cls_data in val["per_class"].items():
                        log_data[f"val/precision_{cls_name}"] = cls_data["precision"]
                        log_data[f"val/recall_{cls_name}"] = cls_data["recall"]
                        log_data[f"val/f1_{cls_name}"] = cls_data["f1"]
            wandb.log(log_data, step=epoch)

        # Per-symbol logging (every 10 epochs + last epoch)
        if val and "per_symbol" in val and val["per_symbol"]:
            per_sym = val["per_symbol"]
            show_detail = (epoch % 10 == 0 or epoch == self.config.epochs)

            if show_detail:
                # Console: top 5 best + bottom 5 worst (1d horizon)
                first_h = None
                for sym_data in per_sym.values():
                    first_h = next(iter(sym_data.keys()), None)
                    break

                if first_h:
                    sym_accs = []
                    for sym, hdata in per_sym.items():
                        if first_h in hdata:
                            sym_accs.append((sym, hdata[first_h]["accuracy"], hdata[first_h]["total"]))
                    sym_accs.sort(key=lambda x: x[1], reverse=True)

                    logger.info(f"    ── Per-symbol accuracy ({first_h}) ──")
                    logger.info(f"    TOP 5:")
                    for sym, acc, cnt in sym_accs[:5]:
                        logger.info(f"      {sym:6s} acc={acc:.3f} (n={cnt})")
                    logger.info(f"    BOTTOM 5:")
                    for sym, acc, cnt in sym_accs[-5:]:
                        logger.info(f"      {sym:6s} acc={acc:.3f} (n={cnt})")
                    avg_acc = np.mean([a for _, a, _ in sym_accs])
                    logger.info(f"    AVG: {avg_acc:.3f} ({len(sym_accs)} symbols)")

            # TensorBoard: per-symbol accuracy for each horizon
            for sym, hdata in per_sym.items():
                for h, metrics in hdata.items():
                    self.writer.add_scalar(f"Symbol_{h}/{sym}", metrics["accuracy"], epoch)

            # Save per-symbol JSON snapshot
            sym_json_dir = self.log_dir / "per_symbol"
            sym_json_dir.mkdir(parents=True, exist_ok=True)
            sym_snapshot = {
                "epoch": epoch,
                "model": self.config.model_name,
                "symbols": {}
            }
            for sym, hdata in per_sym.items():
                sym_snapshot["symbols"][sym] = {
                    h: {"acc": round(m["accuracy"], 4), "n": m["total"]}
                    for h, m in hdata.items()
                }
            snapshot_path = sym_json_dir / f"epoch_{epoch:03d}.json"
            snapshot_path.write_text(json.dumps(sym_snapshot, indent=2))

        # History
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items() if not isinstance(v, dict)}}
        if val:
            record.update({f"val_{k}": v for k, v in val.items() if not isinstance(v, dict)})
            if "per_class" in val:
                for cls_name, cls_data in val["per_class"].items():
                    record[f"val_precision_{cls_name}"] = cls_data["precision"]
                    record[f"val_recall_{cls_name}"] = cls_data["recall"]
                    record[f"val_f1_{cls_name}"] = cls_data["f1"]
        self.history.append(record)

    def _save_checkpoint(self, epoch: int, is_best: bool = False, tag: str = "") -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "market": self.config.market,
                "model_name": self.config.model_name,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
            },
        }

        if is_best:
            path = self.save_dir / "best.pt"
            torch.save(state, path)
            logger.info(f"  Saved best model → {path}")

        if tag:
            path = self.save_dir / f"{tag}.pt"
            torch.save(state, path)
        elif not is_best:
            path = self.save_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, path)
