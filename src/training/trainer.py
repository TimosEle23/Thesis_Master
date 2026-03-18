"""
Training loop with early stopping, metric tracking, and computational cost measurement.

Handles both standard baseline models and TGNN models (which may need
adaptive graph regularization).
"""

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any

from src.training.losses import get_loss_function
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger("thesis")


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """Training pipeline for all models.
    
    Tracks:
      - Training/validation loss and accuracy per epoch
      - Best model checkpointing
      - Early stopping
      - Computational cost (training time, inference time, parameter count)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        save_dir: str = "results",
        experiment_name: str = "experiment",
    ):
        """
        Args:
            model: The model to train.
            device: Compute device.
            config: Training configuration dict with keys:
                epochs, patience, min_delta, batch_size, learning_rate,
                weight_decay, scheduler, gradient_clip_val, label_smoothing
            save_dir: Directory for saving checkpoints/logs.
            experiment_name: Name for this experiment run.
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Training config
        self.epochs = config.get("epochs", 200)
        self.patience = config.get("patience", 30)
        self.min_delta = config.get("min_delta", 0.001)
        self.grad_clip = config.get("gradient_clip_val", 1.0)
        self.adaptive_reg_weight = config.get("adaptive_reg_weight", 0.001)

        # Loss function
        self.criterion = get_loss_function(
            label_smoothing=config.get("label_smoothing", 0.0)
        )

        # Optimizer
        lr = config.get("learning_rate", 0.001)
        wd = config.get("weight_decay", 0.0001)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )

        # LR scheduler
        scheduler_type = config.get("scheduler", "cosine")
        scheduler_params = config.get("scheduler_params", {})
        self.scheduler = self._build_scheduler(scheduler_type, scheduler_params)

        # Early stopping
        self.early_stopper = EarlyStopping(
            patience=self.patience, min_delta=self.min_delta, mode="min"
        )

        # Tracking
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def _build_scheduler(self, scheduler_type, params):
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get("T_max", self.epochs),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 50),
                gamma=params.get("gamma", 0.5),
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
            )
        else:
            return None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Full training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            
        Returns:
            Dictionary with training history and timing information.
        """
        logger.info(f"Starting training: {self.experiment_name}")
        logger.info(f"  Epochs: {self.epochs}, Device: {self.device}")
        logger.info(f"  Parameters: {self.model.count_parameters():,}")

        total_train_time = 0.0

        for epoch in range(1, self.epochs + 1):
            # --- Train epoch ---
            epoch_start = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time

            # --- Validation epoch ---
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Logging
            if epoch % 10 == 0 or epoch <= 5:
                logger.info(
                    f"  Epoch {epoch:3d}/{self.epochs} | "
                    f"Train L:{train_loss:.4f} A:{train_acc:.4f} | "
                    f"Val L:{val_loss:.4f} A:{val_acc:.4f} | "
                    f"LR:{current_lr:.6f} | {epoch_time:.1f}s"
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint("best_model.pt")

            # LR scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopper(val_loss):
                logger.info(f"  Early stopping at epoch {epoch} (best: {self.best_epoch})")
                break

        logger.info(
            f"Training complete. Best val_loss={self.best_val_loss:.4f} "
            f"(acc={self.best_val_acc:.4f}) at epoch {self.best_epoch}"
        )

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_train_time_s": total_train_time,
            "avg_epoch_time_s": total_train_time / max(epoch, 1),
            "total_epochs": epoch,
        }

    def _train_epoch(self, loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            x, y = self._unpack_batch(batch)
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            # Add adaptive graph regularization if applicable
            if hasattr(self.model, "get_adaptive_reg_loss"):
                reg_loss = self.model.get_adaptive_reg_loss()
                if reg_loss.item() > 0:
                    loss = loss + self.adaptive_reg_weight * reg_loss.to(self.device)

            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            x, y = self._unpack_batch(batch)
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    def _unpack_batch(self, batch):
        """Unpack batch from either MTSDataset or GraphMTSDataset."""
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        elif isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate on test set with full metrics.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dictionary with accuracy, F1 scores, confusion matrix, timing.
        """
        # Load best model
        self._load_checkpoint("best_model.pt")
        self.model.eval()

        all_preds = []
        all_labels = []

        # Measure inference time
        inference_start = time.time()

        for batch in test_loader:
            x, y = self._unpack_batch(batch)
            x = x.to(self.device)

            logits = self.model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

        inference_time = time.time() - inference_start

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute all metrics
        metrics = compute_metrics(all_labels, all_preds)
        metrics["inference_time_s"] = inference_time
        metrics["n_parameters"] = self.model.count_parameters()
        metrics["predictions"] = all_preds
        metrics["true_labels"] = all_labels

        return metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
        }, path)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.save_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(f"Checkpoint not found: {path}")
