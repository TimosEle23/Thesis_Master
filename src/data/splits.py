"""
Data splitting strategies.

  - UEA datasets: Predefined train/test split with validation carved from train
  - DSA dataset: Leave-One-Subject-Out (LOSO) cross-validation
"""

import logging
import numpy as np
from typing import Generator, Tuple, List
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger("thesis")


def get_data_splits(
    dataset_name: str,
    data: dict,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> object:
    """Get train/val/test splits for a dataset.
    
    For UEA datasets, returns a single split.
    For DSA, returns a LOSO cross-validation generator.
    
    Args:
        dataset_name: Name of the dataset.
        data: Preprocessed data dictionary.
        val_fraction: Fraction of training data for validation.
        seed: Random seed.
        
    Returns:
        For UEA: dict with X_train, y_train, X_val, y_val, X_test, y_test
        For DSA: LOSOSplitter object (iterable)
    """
    if dataset_name in ["BasicMotions", "Epilepsy"]:
        return _split_uea(data, val_fraction, seed)
    elif dataset_name == "DSA":
        return LOSOSplitter(data)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _split_uea(data: dict, val_fraction: float, seed: int) -> dict:
    """Split UEA dataset into train/val/test.
    
    UEA provides predefined train/test. We carve validation from train.
    """
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Stratified split of training data
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(X_train_full, y_train_full))
    
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    
    logger.info(
        f"UEA split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


class LOSOSplitter:
    """Leave-One-Subject-Out cross-validation for DSA dataset.
    
    Each fold holds out one subject for testing. Within the remaining
    subjects, one is held out for validation (the next subject in sequence).
    This ensures subject-independent evaluation.
    """

    def __init__(self, data: dict):
        self.X = data["X"]
        self.y = data["y"]
        self.subjects = data["subjects"]
        self.unique_subjects = np.sort(np.unique(self.subjects))
        self.n_folds = len(self.unique_subjects)
        
        logger.info(
            f"LOSO CV: {self.n_folds} folds, "
            f"subjects={self.unique_subjects.tolist()}"
        )

    def __len__(self):
        return self.n_folds

    def __iter__(self) -> Generator[dict, None, None]:
        """Yield one fold at a time.
        
        For each test subject, the validation subject is the next one
        in the circular order.
        """
        for fold_idx, test_subject in enumerate(self.unique_subjects):
            # Test: current subject
            test_mask = self.subjects == test_subject
            
            # Validation: next subject in circular order
            val_subject = self.unique_subjects[
                (fold_idx + 1) % self.n_folds
            ]
            val_mask = self.subjects == val_subject
            
            # Train: remaining subjects
            train_mask = ~(test_mask | val_mask)
            
            fold = {
                "fold": fold_idx,
                "test_subject": int(test_subject),
                "val_subject": int(val_subject),
                "X_train": self.X[train_mask],
                "y_train": self.y[train_mask],
                "X_val": self.X[val_mask],
                "y_val": self.y[val_mask],
                "X_test": self.X[test_mask],
                "y_test": self.y[test_mask],
            }
            
            logger.info(
                f"  Fold {fold_idx}: test=S{test_subject}, val=S{val_subject}, "
                f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
            )
            
            yield fold

    def get_fold(self, fold_idx: int) -> dict:
        """Get a specific fold by index."""
        for i, fold in enumerate(self):
            if i == fold_idx:
                return fold
        raise IndexError(f"Fold {fold_idx} out of range (max {self.n_folds - 1})")
