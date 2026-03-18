"""
Logging utilities for experiment tracking.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "thesis",
    log_dir: str = "results",
    level: str = "INFO",
    experiment_name: str = None,
) -> logging.Logger:
    """Set up a logger that writes to both console and file.
    
    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level.
        experiment_name: Optional experiment identifier for the log filename.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir) / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_handler = logging.FileHandler(log_path / f"{experiment_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Structured experiment logger that tracks metrics across epochs."""

    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=save_dir,
            experiment_name=experiment_name,
        )
        self.metrics_history = []

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics for a single epoch."""
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        self.metrics_history.append(record)
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
            f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
        )

    def log_test(self, test_metrics: dict):
        """Log final test metrics."""
        self.logger.info("=" * 60)
        self.logger.info("TEST RESULTS")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                self.logger.info(f"  {k}: {v:.4f}")
            else:
                self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 60)

    def save_metrics(self):
        """Save metrics history to CSV."""
        import pandas as pd
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.save_dir / "metrics_history.csv", index=False)
