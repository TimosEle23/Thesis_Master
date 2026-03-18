"""
Preprocessing pipelines for all datasets.

Handles:
  - Loading raw data into standardised (N, T, C) format
  - Z-score normalisation (per-channel, fit on train only)
  - Sequence padding/truncation
  - Label encoding
  - Saving processed data
"""

import logging
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger("thesis")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def preprocess_dataset(dataset_name: str, force: bool = False) -> dict:
    """Preprocess a dataset and save to disk.
    
    Args:
        dataset_name: One of 'BasicMotions', 'Epilepsy', 'DSA'.
        force: If True, reprocess even if processed data exists.
        
    Returns:
        Dictionary with preprocessed data arrays and metadata.
    """
    output_dir = PROCESSED_DIR / dataset_name
    
    if output_dir.exists() and not force:
        logger.info(f"Loading preprocessed {dataset_name} from {output_dir}")
        return _load_processed(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name in ["BasicMotions", "Epilepsy"]:
        result = _preprocess_uea(dataset_name)
    elif dataset_name == "DSA":
        result = _preprocess_dsa()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    _save_processed(result, output_dir)
    logger.info(f"Preprocessed {dataset_name} saved to {output_dir}")
    
    return result


def _preprocess_uea(dataset_name: str) -> dict:
    """Preprocess a UEA dataset.
    
    UEA datasets from aeon come as (n_samples, n_channels, seq_len).
    We convert to (n_samples, seq_len, n_channels) for consistency.
    """
    dataset_dir = RAW_DIR / dataset_name
    
    # Load raw data (saved by downloader as .npy)
    X_train = np.load(dataset_dir / "X_train.npy", allow_pickle=True)
    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
    X_test = np.load(dataset_dir / "X_test.npy", allow_pickle=True)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)
    
    logger.info(f"Raw {dataset_name}: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # aeon format: (n_samples, n_channels, seq_len) → (n_samples, seq_len, n_channels)
    if X_train.ndim == 3:
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Handle NaN values (some UEA datasets have NaN for variable-length series)
    if np.any(np.isnan(X_train)):
        logger.warning(f"  {dataset_name} has NaN values — replacing with 0")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Z-score normalisation (per channel, fit on train)
    X_train_norm, X_test_norm, scaler_params = _normalise(X_train, X_test)
    
    n_samples, seq_len, n_channels = X_train_norm.shape
    n_classes = len(le.classes_)
    
    logger.info(
        f"  Processed {dataset_name}: shape=({n_samples}, {seq_len}, {n_channels}), "
        f"classes={n_classes}, labels={le.classes_}"
    )
    
    return {
        "X_train": X_train_norm,
        "y_train": y_train_enc.astype(np.int64),
        "X_test": X_test_norm,
        "y_test": y_test_enc.astype(np.int64),
        "label_encoder_classes": le.classes_,
        "scaler_mean": scaler_params["mean"],
        "scaler_std": scaler_params["std"],
        "metadata": {
            "dataset_name": dataset_name,
            "n_channels": n_channels,
            "seq_len": seq_len,
            "n_classes": n_classes,
            "n_train": len(X_train_norm),
            "n_test": len(X_test_norm),
            "class_names": le.classes_.tolist(),
        },
    }


def _preprocess_dsa() -> dict:
    """Preprocess the UCI Daily and Sports Activities dataset.
    
    Data is loaded as (n_samples, seq_len, n_channels) = (9120, 125, 45).
    Subject IDs are preserved for Leave-One-Subject-Out CV.
    """
    from src.data.download import load_dsa_raw
    
    X, y, subjects = load_dsa_raw()
    
    logger.info(f"Raw DSA: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
    logger.info(f"  Activities: {np.unique(y)}, Subjects: {np.unique(subjects)}")
    
    # X is already (N, T, C) = (9120, 125, 45)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    
    # For DSA, we don't split train/test here — LOSO CV is done in splits.py
    # But we still compute and store global normalisation statistics
    
    # Compute global mean/std per channel (across all samples)
    # Actual normalisation will be done per-fold in the training pipeline
    n_samples, seq_len, n_channels = X.shape
    flat = X.reshape(-1, n_channels)
    global_mean = flat.mean(axis=0)
    global_std = flat.std(axis=0)
    global_std[global_std < 1e-8] = 1.0  # Avoid division by zero
    
    # Activity names
    activity_names = [
        "sitting", "standing", "lying_on_back", "lying_on_right",
        "ascending_stairs", "descending_stairs", "standing_in_elevator",
        "moving_in_elevator", "walking_in_parking", "walking_treadmill_flat",
        "walking_treadmill_incline", "running_treadmill", "exercising_stepper",
        "exercising_cross_trainer", "cycling_horizontal", "cycling_vertical",
        "rowing", "jumping", "playing_basketball",
    ]
    
    # Sensor unit mapping
    sensor_units = {
        "torso": list(range(0, 9)),
        "right_arm": list(range(9, 18)),
        "left_arm": list(range(18, 27)),
        "right_leg": list(range(27, 36)),
        "left_leg": list(range(36, 45)),
    }
    
    channel_names = []
    for unit_name in ["torso", "right_arm", "left_arm", "right_leg", "left_leg"]:
        for sensor_type in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                           "mag_x", "mag_y", "mag_z"]:
            channel_names.append(f"{unit_name}_{sensor_type}")
    
    n_classes = len(np.unique(y))
    
    logger.info(
        f"  Processed DSA: shape=({n_samples}, {seq_len}, {n_channels}), "
        f"classes={n_classes}, subjects={len(np.unique(subjects))}"
    )
    
    return {
        "X": X,
        "y": y,
        "subjects": subjects,
        "scaler_mean": global_mean,
        "scaler_std": global_std,
        "metadata": {
            "dataset_name": "DSA",
            "n_channels": n_channels,
            "seq_len": seq_len,
            "n_classes": n_classes,
            "n_samples": n_samples,
            "n_subjects": len(np.unique(subjects)),
            "activity_names": activity_names,
            "channel_names": channel_names,
            "sensor_units": sensor_units,
        },
    }


def _normalise(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Per-channel Z-score normalisation fitted on training data.
    
    Args:
        X_train: (n_train, seq_len, n_channels)
        X_test: (n_test, seq_len, n_channels)
        
    Returns:
        X_train_norm, X_test_norm, scaler_params dict
    """
    n_channels = X_train.shape[2]
    
    # Compute per-channel statistics from training data
    # Reshape to (n_train * seq_len, n_channels)
    flat_train = X_train.reshape(-1, n_channels)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std < 1e-8] = 1.0  # Avoid division by zero
    
    # Apply normalisation
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32), \
           {"mean": mean, "std": std}


def normalise_fold(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray = None):
    """Normalise a single CV fold (fit on train, transform val/test).
    
    Used for LOSO cross-validation in the DSA dataset.
    """
    n_channels = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
    
    flat_train = X_train.reshape(-1, n_channels)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std < 1e-8] = 1.0
    
    X_train_norm = ((X_train - mean) / std).astype(np.float32)
    X_val_norm = ((X_val - mean) / std).astype(np.float32)
    
    if X_test is not None:
        X_test_norm = ((X_test - mean) / std).astype(np.float32)
        return X_train_norm, X_val_norm, X_test_norm
    
    return X_train_norm, X_val_norm


def _save_processed(data: dict, output_dir: Path):
    """Save processed data arrays and metadata."""
    import json
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.save(output_dir / f"{key}.npy", value)
        elif key == "metadata":
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(value, f, indent=2, default=str)
        elif key == "label_encoder_classes":
            np.save(output_dir / "label_encoder_classes.npy", value)
        elif key.startswith("scaler_"):
            np.save(output_dir / f"{key}.npy", value)


def _load_processed(output_dir: Path) -> dict:
    """Load preprocessed data from disk."""
    import json
    
    data = {}
    
    for npy_file in output_dir.glob("*.npy"):
        key = npy_file.stem
        data[key] = np.load(npy_file, allow_pickle=True)
    
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            data["metadata"] = json.load(f)
    
    return data
