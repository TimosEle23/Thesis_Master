"""
Dataset loaders for Phase 1 (UEA Archive) and Phase 2 (UCI DSA).

The user has already placed the raw data in the workspace:
  - BasicMotions/ (UEA .ts format)
  - Epilepsy/     (UEA .ts format)  
  - UCI/          (DSA, a01-a19/p1-p8/s01-s60.txt)

Phase 1: BasicMotions, Epilepsy from UEA Multivariate Time Series Classification Archive
Phase 2: Daily and Sports Activities from UCI Machine Learning Repository
"""

import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("thesis")

# ---- Constants ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Local dataset directories (already downloaded by the user)
LOCAL_BASIC_MOTIONS = PROJECT_ROOT / "BasicMotions"
LOCAL_EPILEPSY = PROJECT_ROOT / "Epilepsy"
LOCAL_DSA = PROJECT_ROOT / "UCI"

UEA_DATASETS = ["BasicMotions", "Epilepsy"]


def ensure_dirs():
    """Create data directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)


# =============================================================================
# UEA .ts file parser
# =============================================================================

def _parse_ts_file(filepath: Path) -> tuple:
    """Parse a UEA .ts (time series) format file.
    
    The .ts format has a header with metadata lines starting with @ and
    then data lines where dimensions are separated by ':' and the last
    token after the final ':' is the class label.
    
    Returns:
        X: np.ndarray of shape (n_samples, n_dimensions, series_length)
        y: np.ndarray of shape (n_samples,) with string class labels
    """
    metadata = {}
    data_lines = []
    in_data = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                parts = line.split(" ", 1)
                key = parts[0].lower()
                val = parts[1].strip() if len(parts) > 1 else ""
                if key == "@data":
                    in_data = True
                    continue
                metadata[key] = val
            elif in_data:
                data_lines.append(line)

    n_dims = int(metadata.get("@dimensions", 1))
    logger.info(f"  Parsing {filepath.name}: {len(data_lines)} samples, {n_dims} dimensions")

    X_list = []
    y_list = []

    for data_line in data_lines:
        # Dimensions separated by ':', last token is class label
        # "val,val,...:val,val,...:...:label"
        parts = data_line.split(":")
        label = parts[-1].strip()
        dim_parts = parts[:-1]

        if len(dim_parts) != n_dims:
            # Sometimes label is embedded as last value in last dimension
            # Try splitting last dim on final comma
            last_dim_vals = dim_parts[-1].rsplit(",", 1)
            if len(last_dim_vals) == 2:
                dim_parts[-1] = last_dim_vals[0]
                label = last_dim_vals[1].strip()

        sample_dims = []
        for dim_str in dim_parts:
            vals = [float(v) for v in dim_str.split(",") if v.strip()]
            sample_dims.append(vals)

        X_list.append(sample_dims)
        y_list.append(label)

    # Convert to numpy: (n_samples, n_dims, series_length)
    # All series should be equal length for these datasets
    series_len = len(X_list[0][0])
    X = np.array(X_list, dtype=np.float32)  # (N, D, T)
    y = np.array(y_list)

    logger.info(f"  Parsed: X={X.shape}, y classes={np.unique(y)}, series_len={series_len}")
    return X, y


def download_uea_dataset(dataset_name: str, force: bool = False) -> Path:
    """Load a UEA dataset from the local .ts files.
    
    Reads the _TRAIN.ts and _TEST.ts files already present in the workspace,
    parses them, and saves as .npy for fast subsequent loading.
    
    Args:
        dataset_name: 'BasicMotions' or 'Epilepsy'.
        force: If True, re-parse even if .npy files exist.
        
    Returns:
        Path to the dataset directory with .npy files.
    """
    ensure_dirs()
    
    # Map to local directory
    if dataset_name == "BasicMotions":
        local_dir = LOCAL_BASIC_MOTIONS
    elif dataset_name == "Epilepsy":
        local_dir = LOCAL_EPILEPSY
    else:
        raise ValueError(f"Unknown UEA dataset: {dataset_name}")
    
    output_dir = RAW_DIR / dataset_name
    
    # Check if already parsed
    if (output_dir / "X_train.npy").exists() and not force:
        logger.info(f"Dataset {dataset_name} already parsed at {output_dir}")
        return output_dir
    
    # Check local files exist
    train_ts = local_dir / f"{dataset_name}_TRAIN.ts"
    test_ts = local_dir / f"{dataset_name}_TEST.ts"
    
    if not train_ts.exists():
        raise FileNotFoundError(f"Train file not found: {train_ts}")
    if not test_ts.exists():
        raise FileNotFoundError(f"Test file not found: {test_ts}")
    
    logger.info(f"Parsing {dataset_name} from local .ts files...")
    
    X_train, y_train = _parse_ts_file(train_ts)
    X_test, y_test = _parse_ts_file(test_ts)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    logger.info(
        f"  {dataset_name}: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"classes={np.unique(y_train)}"
    )
    return output_dir


def download_dsa_dataset(force: bool = False) -> Path:
    """Load the UCI Daily and Sports Activities dataset from local files.
    
    The user has placed the dataset at UCI/ in the workspace root.
    Structure: UCI/a{01-19}/p{1-8}/s{01-60}.txt
    Each file: 125 rows × 45 comma-separated columns.
    
    Args:
        force: If True, re-process even if output exists.
        
    Returns:
        Path to the dataset directory.
    """
    ensure_dirs()
    
    if not LOCAL_DSA.exists():
        raise FileNotFoundError(f"DSA data not found at {LOCAL_DSA}")
    
    # Verify structure
    _verify_dsa_structure(LOCAL_DSA)
    
    return LOCAL_DSA


def _verify_dsa_structure(dataset_dir: Path):
    """Verify the DSA dataset has the expected structure."""
    activity_dirs = sorted([d for d in dataset_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("a")])
    n_activities = len(activity_dirs)
    
    sample_activity = dataset_dir / "a01"
    subject_dirs = sorted([d for d in sample_activity.iterdir() 
                          if d.is_dir() and d.name.startswith("p")]) if sample_activity.exists() else []
    n_subjects = len(subject_dirs)
    
    if n_subjects > 0:
        sample_segments = list((sample_activity / "p1").glob("s*.txt"))
        n_segments = len(sample_segments)
    else:
        n_segments = 0
    
    logger.info(
        f"  DSA verified: {n_activities} activities, {n_subjects} subjects, "
        f"{n_segments} segments per subject"
    )


def load_dsa_raw(dataset_dir: Path = None) -> tuple:
    """Load raw DSA data into numpy arrays from local UCI/ directory.
    
    Returns:
        X: np.ndarray of shape (n_samples, seq_len, n_channels) = (9120, 125, 45)
        y: np.ndarray of shape (n_samples,) with activity labels 0-18
        subjects: np.ndarray of shape (n_samples,) with subject ids 0-7
    """
    if dataset_dir is None:
        dataset_dir = LOCAL_DSA

    X_list = []
    y_list = []
    subject_list = []

    for activity_idx in range(1, 20):  # a01 to a19
        activity_dir = dataset_dir / f"a{activity_idx:02d}"
        if not activity_dir.exists():
            logger.warning(f"Missing activity directory: {activity_dir}")
            continue

        for subject_idx in range(1, 9):  # p1 to p8
            subject_dir = activity_dir / f"p{subject_idx}"
            if not subject_dir.exists():
                continue

            for segment_idx in range(1, 61):  # s01 to s60
                segment_file = subject_dir / f"s{segment_idx:02d}.txt"
                if not segment_file.exists():
                    continue

                # Each file: 125 rows x 45 columns (comma-separated)
                data = np.loadtxt(segment_file, delimiter=",")
                X_list.append(data)
                y_list.append(activity_idx - 1)  # 0-indexed
                subject_list.append(subject_idx - 1)  # 0-indexed

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subject_list, dtype=np.int64)

    logger.info(f"  Loaded DSA: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
    return X, y, subjects


def download_all_datasets(force: bool = False):
    """Download all datasets for both phases."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL DATASETS")
    logger.info("=" * 60)

    # Phase 1: UEA datasets
    for name in UEA_DATASETS:
        try:
            download_uea_dataset(name, force=force)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

    # Phase 2: DSA
    try:
        download_dsa_dataset(force=force)
    except Exception as e:
        logger.error(f"Failed to download DSA: {e}")

    logger.info("Download complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all_datasets()
