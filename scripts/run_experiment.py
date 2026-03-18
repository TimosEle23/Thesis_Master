"""
Run a single experiment: one model × one dataset × one graph mode × one seed.

Usage:
    python -m scripts.run_experiment \
        --dataset BasicMotions \
        --model gconv_lstm \
        --graph_mode predefined \
        --seed 42 \
        --phase 1

Outputs are saved to results/<phase>/<dataset>/<model>_<graph_mode>_seed<s>/
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging_utils import setup_logger
from src.data.download import download_all_datasets
from src.data.preprocessing import preprocess_dataset, normalise_fold
from src.data.splits import get_data_splits
from src.data.datasets import MTSDataset, GraphMTSDataset, graph_collate_fn
from src.graphs.predefined import build_predefined_graph
from src.models.factory import build_model, BASELINE_MODELS, TGNN_MODELS
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics, measure_inference_time, count_parameters

logger = logging.getLogger("thesis")


def load_config(phase: int) -> dict:
    """Load and merge phase config with defaults."""
    config_dir = PROJECT_ROOT / "configs"

    # Load defaults
    with open(config_dir / "default.yaml") as f:
        defaults = yaml.safe_load(f)

    # Load phase-specific config
    phase_file = config_dir / f"phase{phase}.yaml"
    if phase_file.exists():
        with open(phase_file) as f:
            phase_cfg = yaml.safe_load(f)
    else:
        phase_cfg = {}

    # Deep merge: phase overrides defaults
    config = _deep_merge(defaults, phase_cfg)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _get_graph_config(dataset_name: str, graph_mode: str, n_channels: int) -> dict:
    """Get graph configuration for a TGNN experiment."""
    if graph_mode == "predefined":
        if dataset_name == "DSA":
            return build_predefined_graph(dataset_name, granularity="coarse")
        else:
            return build_predefined_graph(dataset_name)

    elif graph_mode == "predefined_coarse":
        return build_predefined_graph("DSA", granularity="coarse")

    elif graph_mode == "predefined_fine":
        return build_predefined_graph("DSA", granularity="fine")

    elif graph_mode == "adaptive":
        # For adaptive mode, we still need num_nodes and node_features_dim
        # The actual adjacency is learned; we build a skeleton config
        if dataset_name == "DSA":
            return {
                "num_nodes": 5,
                "node_features_dim": 9,
                "edge_index": None,
                "edge_weight": None,
                "adj_matrix": None,
            }
        else:
            return {
                "num_nodes": n_channels,
                "node_features_dim": 1,
                "edge_index": None,
                "edge_weight": None,
                "adj_matrix": None,
            }
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")


def build_dataloaders(
    dataset_name: str,
    split_data: dict,
    model_name: str,
    graph_config: dict,
    graph_mode: str,
    batch_size: int,
    num_workers: int = 0,
):
    """Build train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    if model_name in BASELINE_MODELS:
        train_ds = MTSDataset(X_train, y_train)
        val_ds = MTSDataset(X_val, y_val)
        test_ds = MTSDataset(X_test, y_test)
        collate = None
    else:
        # TGNN model — needs graph structure
        edge_index = graph_config.get("edge_index")
        edge_weight = graph_config.get("edge_weight")
        sensor_groups = graph_config.get("sensor_groups")
        node_feat_dim = graph_config.get("node_features_dim", 1)

        # Determine node_mode
        node_mode = "unit" if node_feat_dim > 1 else "channel"

        # For adaptive mode, use identity graph structure  as placeholder
        if edge_index is None:
            num_nodes = graph_config["num_nodes"]
            # Self-loop identity
            edge_index = np.array([list(range(num_nodes)), list(range(num_nodes))])
            edge_weight = np.ones(num_nodes, dtype=np.float32)

        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if isinstance(edge_weight, torch.Tensor):
            edge_weight = edge_weight.numpy()

        train_ds = GraphMTSDataset(
            X_train, y_train, edge_index, edge_weight,
            node_mode=node_mode, sensor_groups=sensor_groups,
        )
        val_ds = GraphMTSDataset(
            X_val, y_val, edge_index, edge_weight,
            node_mode=node_mode, sensor_groups=sensor_groups,
        )
        test_ds = GraphMTSDataset(
            X_test, y_test, edge_index, edge_weight,
            node_mode=node_mode, sensor_groups=sensor_groups,
        )
        collate = graph_collate_fn

    pin = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate,
    )
    return train_loader, val_loader, test_loader


def run_uea_experiment(
    config: dict,
    dataset_name: str,
    model_name: str,
    graph_mode: str,
    seed: int,
    results_dir: Path,
) -> dict:
    """Run a single UEA experiment (Phase 1)."""
    set_seed(seed)
    device = get_device(config["experiment"].get("device", "auto"))

    # 1. Prepare data
    data = preprocess_dataset(dataset_name)
    splits = get_data_splits(dataset_name, data, seed=seed)
    metadata = data["metadata"]

    n_channels = metadata["n_channels"]
    n_classes = metadata["n_classes"]
    seq_len = metadata["seq_len"]

    # 2. Build graph config (if TGNN)
    graph_config = None
    if model_name in TGNN_MODELS:
        graph_config = _get_graph_config(dataset_name, graph_mode, n_channels)

    # 3. Build model
    model_cfg_key = model_name
    model_cfg = config.get("models", {}).get(model_cfg_key, {})
    model = build_model(
        model_name, n_channels, n_classes, seq_len,
        model_cfg, graph_config, graph_mode,
    )

    # 4. Build dataloaders
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name, splits, model_name, graph_config or {},
        graph_mode, batch_size,
    )

    # 5. Train
    exp_name = f"{dataset_name}_{model_name}_{graph_mode}_seed{seed}"
    trainer = Trainer(
        model=model, device=device, config=train_cfg,
        save_dir=str(results_dir), experiment_name=exp_name,
    )
    train_result = trainer.train(train_loader, val_loader)

    # 6. Evaluate
    test_metrics = trainer.evaluate(test_loader)

    # 7. Measure inference time
    inf_timing = measure_inference_time(model, test_loader, device)
    test_metrics.update(inf_timing)

    # 8. Compile results
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "graph_mode": graph_mode if model_name in TGNN_MODELS else "none",
        "seed": seed,
        "phase": 1,
        "n_parameters": count_parameters(model),
        **{k: v for k, v in test_metrics.items()
           if k not in ("predictions", "true_labels", "confusion_matrix")},
        "best_epoch": train_result["best_epoch"],
        "total_train_time_s": train_result["total_train_time_s"],
        "total_epochs": train_result["total_epochs"],
    }

    # Save
    out_dir = results_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save detailed outputs
    np.save(out_dir / "predictions.npy", test_metrics["predictions"])
    np.save(out_dir / "true_labels.npy", test_metrics["true_labels"])
    np.save(out_dir / "confusion_matrix.npy", np.array(test_metrics["confusion_matrix"]))

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(train_result["history"], f, indent=2)

    logger.info(f"Results saved: {out_dir}")
    return result


def run_dsa_experiment(
    config: dict,
    model_name: str,
    graph_mode: str,
    seed: int,
    results_dir: Path,
    max_folds: int = None,
) -> dict:
    """Run a DSA LOSO experiment (Phase 2).

    Iterates over all LOSO folds, trains/evaluates each, then aggregates.
    """
    set_seed(seed)
    device = get_device(config["experiment"].get("device", "auto"))

    # 1. Prepare data
    data = preprocess_dataset("DSA")
    loso = get_data_splits("DSA", data)
    metadata = data["metadata"]

    n_channels = metadata["n_channels"]
    n_classes = metadata["n_classes"]
    seq_len = metadata["seq_len"]

    # 2. Graph config
    graph_config = None
    if model_name in TGNN_MODELS:
        graph_config = _get_graph_config("DSA", graph_mode, n_channels)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)

    fold_results = []

    n_folds = max_folds or len(loso)
    for fold_data in loso:
        fold_idx = fold_data["fold"]
        if fold_idx >= n_folds:
            break

        set_seed(seed + fold_idx)

        logger.info(f"=== Fold {fold_idx}/{n_folds-1} (test=S{fold_data['test_subject']}) ===")

        # Normalise this fold
        X_train, X_val, X_test = normalise_fold(
            fold_data["X_train"], fold_data["X_val"], fold_data["X_test"]
        )
        fold_split = {
            "X_train": X_train, "y_train": fold_data["y_train"],
            "X_val": X_val, "y_val": fold_data["y_val"],
            "X_test": X_test, "y_test": fold_data["y_test"],
        }

        # Build model fresh for each fold
        model_cfg = config.get("models", {}).get(model_name, {})
        model = build_model(
            model_name, n_channels, n_classes, seq_len,
            model_cfg, graph_config, graph_mode,
        )

        # Dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            "DSA", fold_split, model_name, graph_config or {},
            graph_mode, batch_size,
        )

        # Train
        exp_name = f"DSA_{model_name}_{graph_mode}_seed{seed}_fold{fold_idx}"
        trainer = Trainer(
            model=model, device=device, config=train_cfg,
            save_dir=str(results_dir), experiment_name=exp_name,
        )
        train_result = trainer.train(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader)
        test_metrics["fold"] = fold_idx
        test_metrics["test_subject"] = fold_data["test_subject"]
        test_metrics["best_epoch"] = train_result["best_epoch"]
        test_metrics["total_train_time_s"] = train_result["total_train_time_s"]

        fold_results.append(test_metrics)

    # Aggregate across folds
    from src.evaluation.metrics import aggregate_loso_results
    aggregated = aggregate_loso_results(fold_results)

    result = {
        "dataset": "DSA",
        "model": model_name,
        "graph_mode": graph_mode if model_name in TGNN_MODELS else "none",
        "seed": seed,
        "phase": 2,
        "n_parameters": count_parameters(model),
        **{k: v for k, v in aggregated.items()
           if not isinstance(v, (list, np.ndarray))},
    }

    # Save
    exp_dir = results_dir / f"DSA_{model_name}_{graph_mode}_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(exp_dir / "fold_results.json", "w") as f:
        fold_serialisable = []
        for fr in fold_results:
            fr_s = {k: v for k, v in fr.items()
                    if not isinstance(v, np.ndarray)}
            fold_serialisable.append(fr_s)
        json.dump(fold_serialisable, f, indent=2, default=str)

    logger.info(f"DSA results saved: {exp_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: BasicMotions, Epilepsy, DSA")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name: lstm, tcn, transformer, gconv_lstm, gconv_gru, a3tgcn, dcrnn")
    parser.add_argument("--graph_mode", type=str, default="predefined",
                        help="Graph mode: predefined, predefined_coarse, predefined_fine, adaptive, none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Max number of LOSO folds for DSA (default: all)")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.phase)

    # Setup logging
    results_base = Path(args.results_dir) if args.results_dir else \
        PROJECT_ROOT / "results" / f"phase{args.phase}"
    results_base.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(results_base))

    logger.info(f"{'='*60}")
    logger.info(f"Experiment: {args.dataset} / {args.model} / {args.graph_mode} / seed={args.seed}")
    logger.info(f"{'='*60}")

    # Ensure data is downloaded
    download_all_datasets()

    # Run
    start_time = time.time()

    if args.dataset in ["BasicMotions", "Epilepsy"]:
        result = run_uea_experiment(
            config, args.dataset, args.model, args.graph_mode,
            args.seed, results_base,
        )
    elif args.dataset == "DSA":
        result = run_dsa_experiment(
            config, args.model, args.graph_mode,
            args.seed, results_base, max_folds=args.max_folds,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    elapsed = time.time() - start_time
    logger.info(f"Total wall time: {elapsed:.1f}s")
    logger.info(f"Final accuracy: {result.get('accuracy', result.get('accuracy_mean', 'N/A'))}")

    return result


if __name__ == "__main__":
    main()
