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
import os
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
from src.graphs.predefined import (
    build_predefined_graph,
    build_correlation_graph,
    build_dtw_graph,
    build_knn_graph,
    build_random_predefined_graph,
    build_correlation_graph_topN,
    build_correlation_graph_nothresh,
    build_correlation_graph_datadriven,
)
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


def _get_graph_config(
    dataset_name: str,
    graph_mode: str,
    n_channels: int,
    X_train: np.ndarray = None,
) -> dict:
    """Get graph configuration for a TGNN experiment.

    Supported graph_mode values:
      Frozen graphs:
        predefined              — anatomical / sensor-axis binary
        predefined_coarse       — DSA 5-node anatomical binary
        predefined_fine         — DSA 45-node anatomical binary
        predefined_correlation  — signed Pearson r threshold (data-driven)
        predefined_dtw          — DTW1 unconstrained similarity (data-driven)
        predefined_dtw_windowed — DTW2 windowed similarity (data-driven)
        predefined_knn_correlation — mutual k-NN by |r| (data-driven)
        predefined_knn_dtw      — mutual k-NN by DTW similarity (data-driven)
      Adaptive (learned via embeddings):
        adaptive                — random init (Xavier)
        adaptive_binary         — SVD-seeded from anatomical binary
        adaptive_correlation    — SVD-seeded from signed-correlation graph
        adaptive_dtw            — SVD-seeded from DTW similarity graph
        adaptive_knn            — SVD-seeded from k-NN correlation graph

    Data-driven modes require X_train (n_samples, seq_len, n_channels).
    """
    # ── Frozen anatomical / sensor-axis ─────────────────────────────
    if graph_mode == "predefined":
        if dataset_name == "DSA":
            return build_predefined_graph(dataset_name, granularity="coarse")
        else:
            return build_predefined_graph(dataset_name)

    if graph_mode == "predefined_coarse":
        return build_predefined_graph("DSA", granularity="coarse")

    if graph_mode == "predefined_fine":
        return build_predefined_graph("DSA", granularity="fine")

    # ── Frozen data-driven ──────────────────────────────────────────
    if graph_mode in ("predefined_correlation", "predefined_dtw",
                      "predefined_dtw_windowed", "predefined_knn_correlation",
                      "predefined_knn_dtw"):
        if X_train is None:
            raise ValueError(f"X_train required for graph_mode='{graph_mode}'")
        # Choose a reasonable k for KNN/DTW based on channel count
        k_default = max(2, min(3, n_channels - 1))
        if graph_mode == "predefined_correlation":
            return build_correlation_graph(X_train, threshold=0.3)
        if graph_mode == "predefined_dtw":
            return build_dtw_graph(X_train, k=k_default, window=None)
        if graph_mode == "predefined_dtw_windowed":
            return build_dtw_graph(X_train, k=k_default, window=10)
        if graph_mode == "predefined_knn_correlation":
            return build_knn_graph(X_train, k=k_default, metric="correlation")
        if graph_mode == "predefined_knn_dtw":
            return build_knn_graph(X_train, k=k_default, metric="dtw")

    # ── Adaptive: random or seeded from a predefined graph ──────────
    if graph_mode == "adaptive":
        # Random init — skeleton config only, model learns adjacency
        if dataset_name == "DSA":
            return {
                "num_nodes": 5,
                "node_features_dim": 9,
                "edge_index": None,
                "edge_weight": None,
                "adj_matrix": None,
                "sensor_groups": {
                    "torso": list(range(0, 9)),
                    "right_arm": list(range(9, 18)),
                    "left_arm": list(range(18, 27)),
                    "right_leg": list(range(27, 36)),
                    "left_leg": list(range(36, 45)),
                },
            }
        else:
            return {
                "num_nodes": n_channels,
                "node_features_dim": 1,
                "edge_index": None,
                "edge_weight": None,
                "adj_matrix": None,
            }

    if graph_mode.startswith("adaptive_"):
        # Seeded adaptive: build the predefined graph, hand its adj_matrix
        # to the adaptive learner (model uses SVD seeding when adj_matrix
        # is present in graph_config and canonical mode == "adaptive").
        init_source = graph_mode[len("adaptive_"):]
        # Element-wise (no-SVD) variants: strip "nosvd_" prefix, mark adaptive_mode
        adaptive_mode = "factored"
        if init_source.startswith("nosvd_"):
            init_source = init_source[len("nosvd_"):]
            adaptive_mode = "elementwise"
        # Adaptive seeded from a RANDOM-predefined graph (feedback #4, parallel to
        # the domain-seeded adaptive variants):
        #   adaptive_random_predef_seed{N}         -> factored (SVD) seed from random graph
        #   adaptive_nosvd_random_predef_seed{N}   -> elementwise seed from random graph
        if init_source.startswith("random_predef_seed"):
            try:
                seed_val = int(init_source[len("random_predef_seed"):])
            except ValueError:
                raise ValueError(f"Bad adaptive random_predef graph_mode: {graph_mode}")
            gran = "coarse" if dataset_name == "DSA" else "default"
            cfg = build_random_predefined_graph(dataset_name, seed=seed_val, granularity=gran)
            cfg["adaptive_mode"] = adaptive_mode
            return cfg
        source_map = {
            "binary": "predefined" if dataset_name != "DSA" else "predefined_coarse",
            "correlation": "predefined_correlation",
            "dtw": "predefined_dtw",
            "knn": "predefined_knn_correlation",
            "correlation_mean": "predefined_correlation_mean",
            "correlation_meanstd": "predefined_correlation_meanstd",
            "correlation_top50": "predefined_correlation_top50",
            "correlation_nothresh": "predefined_correlation_nothresh",
        }
        if init_source not in source_map:
            raise ValueError(
                f"Unknown adaptive init source '{init_source}'. "
                f"Choose from {list(source_map.keys())}."
            )
        cfg = _get_graph_config(
            dataset_name, source_map[init_source], n_channels, X_train=X_train,
        )
        cfg["adaptive_mode"] = adaptive_mode
        return cfg

    # ── Random-predefined null model (feedback #4) ──────────────────
    if graph_mode.startswith("random_predef_seed"):
        try:
            seed_val = int(graph_mode[len("random_predef_seed"):])
        except ValueError:
            raise ValueError(f"Bad random_predef graph_mode: {graph_mode}")
        gran = "coarse" if dataset_name == "DSA" else "default"
        return build_random_predefined_graph(dataset_name, seed=seed_val, granularity=gran)

    # ── Correlation-graph threshold sweep (feedback #3) ─────────────
    if graph_mode == "predefined_correlation_nothresh":
        if X_train is None:
            raise ValueError(f"X_train required for graph_mode='{graph_mode}'")
        return build_correlation_graph_nothresh(X_train)
    if graph_mode == "predefined_correlation_top50":
        if X_train is None:
            raise ValueError(f"X_train required for graph_mode='{graph_mode}'")
        return build_correlation_graph_topN(X_train, fraction=0.5)

    # Data-driven threshold: |r| >= mean(|r|)  or  mean(|r|)+std(|r|)
    if graph_mode == "predefined_correlation_mean":
        if X_train is None:
            raise ValueError(f"X_train required for graph_mode='{graph_mode}'")
        return build_correlation_graph_datadriven(X_train, stat="mean")
    if graph_mode == "predefined_correlation_meanstd":
        if X_train is None:
            raise ValueError(f"X_train required for graph_mode='{graph_mode}'")
        return build_correlation_graph_datadriven(X_train, stat="meanstd")

    raise ValueError(f"Unknown graph_mode: {graph_mode}")


def _canonical_graph_mode(graph_mode: str) -> str:
    """Map a specific graph_mode string to what the model expects.

    The TGNN models only distinguish 'adaptive' from frozen graphs. All
    'adaptive_*' variants (random, binary, correlation, dtw, knn) are
    handled by the AdaptiveGraphLearner — the init source is conveyed via
    `adj_matrix` in graph_config rather than the mode string.
    """
    if graph_mode.startswith("adaptive"):
        return "adaptive"
    return graph_mode


def build_dataloaders(
    dataset_name: str,
    split_data: dict,
    model_name: str,
    graph_config: dict,
    graph_mode: str,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool | None = None,
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

    pin = pin_memory if pin_memory is not None else device.type == "cuda"

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


def _maybe_compile(model, device):
    """Optionally wrap the model in torch.compile to cut kernel-launch overhead.

    These recurrent TGNNs are launch-bound (125 sequential timesteps, each a
    tiny graph-conv on a small graph), so the GPU sits mostly idle waiting on
    kernel launches. torch.compile fuses kernels / (in reduce-overhead mode)
    captures CUDA graphs to remove that overhead. It only changes speed, not
    numerics/results. Enabled on CUDA by default (TORCH_COMPILE=1); set
    TORCH_COMPILE=0 to disable, TORCH_COMPILE_MODE to pick the mode. The first
    epoch of each fold pays a one-time compile cost — judge speed from epoch 2+.
    Falls back to eager if compilation raises.
    """
    if device.type != "cuda" or os.environ.get("TORCH_COMPILE", "1") != "1":
        return model
    mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    try:
        compiled = torch.compile(model, mode=mode)
        logger.info(f"  torch.compile ON (mode={mode}) — epoch 1 includes one-time compile")
        return compiled
    except Exception as e:  # pragma: no cover
        logger.warning(f"  torch.compile failed ({e}); running eager")
        return model


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
    exp_cfg = config.get("experiment", {})
    device = get_device(exp_cfg.get("device", "auto"))

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
        graph_config = _get_graph_config(
            dataset_name, graph_mode, n_channels,
            X_train=splits["X_train"],
        )

    # 3. Build model
    model_cfg_key = model_name
    model_cfg = config.get("models", {}).get(model_cfg_key, {})
    canonical_mode = _canonical_graph_mode(graph_mode)
    model = build_model(
        model_name, n_channels, n_classes, seq_len,
        model_cfg, graph_config, canonical_mode,
    )

    # 4. Build dataloaders
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = exp_cfg.get("num_workers", 0)
    pin_cfg = exp_cfg.get("pin_memory")
    pin_memory = (pin_cfg if pin_cfg is not None else device.type == "cuda")
    pin_memory = pin_memory and device.type == "cuda"
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name, splits, model_name, graph_config or {},
        canonical_mode, batch_size, device,
        num_workers=num_workers, pin_memory=pin_memory,
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
    exp_cfg = config.get("experiment", {})
    device = get_device(exp_cfg.get("device", "auto"))

    # 1. Prepare data
    data = preprocess_dataset("DSA")
    loso = get_data_splits("DSA", data)
    metadata = data["metadata"]

    n_channels = metadata["n_channels"]
    n_classes = metadata["n_classes"]
    seq_len = metadata["seq_len"]

    # 2. Graph config — for data-driven modes, must be rebuilt per fold using
    #    that fold's (normalised) X_train. For static modes it's idempotent.
    is_tgnn = model_name in TGNN_MODELS
    canonical_mode = _canonical_graph_mode(graph_mode)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = exp_cfg.get("num_workers", 0)
    pin_cfg = exp_cfg.get("pin_memory")
    pin_memory = (pin_cfg if pin_cfg is not None else device.type == "cuda")
    pin_memory = pin_memory and device.type == "cuda"

    # Create exp_dir early so fold caches can be written incrementally
    exp_dir = results_dir / f"DSA_{model_name}_{graph_mode}_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    n_folds = max_folds or len(loso)
    for fold_data in loso:
        fold_idx = fold_data["fold"]
        if fold_idx >= n_folds:
            break

        set_seed(seed + fold_idx)

        # ── Fold-level cache: skip if already completed in a previous run ──
        fold_cache = exp_dir / f"fold_{fold_idx}_results.json"
        if fold_cache.exists():
            with open(fold_cache) as f:
                cached = json.load(f)
            fold_results.append(cached)
            logger.info(
                f"  Fold {fold_idx}: CACHED "
                f"(acc={cached.get('accuracy', '?')})"
            )
            continue

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

        # Build graph from fold's training data only (no leakage)
        graph_config = None
        if is_tgnn:
            graph_config = _get_graph_config(
                "DSA", graph_mode, n_channels, X_train=X_train,
            )

        # Build model fresh for each fold
        model_cfg = config.get("models", {}).get(model_name, {})
        model = build_model(
            model_name, n_channels, n_classes, seq_len,
            model_cfg, graph_config, canonical_mode,
        )
        model = _maybe_compile(model, device)

        # Dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            "DSA", fold_split, model_name, graph_config or {},
            canonical_mode, batch_size, device,
            num_workers=num_workers, pin_memory=pin_memory,
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

        # ── Save fold result immediately (crash-safe) ──────────────────────
        fold_serialisable = {k: v for k, v in test_metrics.items()
                             if not isinstance(v, np.ndarray)}
        with open(fold_cache, "w") as f:
            json.dump(fold_serialisable, f, indent=2, default=str)
        logger.info(f"  Fold {fold_idx}: saved → {fold_cache.name}")

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

    # Save final aggregated results
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
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Override compute device (default: experiment.device from config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Max number of LOSO folds for DSA (default: all)")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.phase)

    # CLI override for device selection (e.g., force CPU to avoid GPU usage limits)
    if args.device is not None:
        config.setdefault("experiment", {})
        config["experiment"]["device"] = args.device

    # Env override for DataLoader workers. The Phase 2 config uses num_workers=8
    # to feed a GPU; on macOS/CPU those spawn-based workers deadlock, so local
    # runs must set FORCE_NUM_WORKERS=0 (load in the main process instead).
    if os.environ.get("FORCE_NUM_WORKERS") is not None:
        config.setdefault("experiment", {})
        config["experiment"]["num_workers"] = int(os.environ["FORCE_NUM_WORKERS"])

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
