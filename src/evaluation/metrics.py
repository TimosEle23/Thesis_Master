"""
Evaluation metrics for multivariate time series classification.

Computes accuracy, macro/weighted F1, per-class F1, confusion matrix,
and computational cost metrics.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger("thesis")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute the full set of classification metrics.

    Args:
        y_true: Ground-truth integer labels (N,).
        y_pred: Predicted integer labels  (N,).
        class_names: Optional list of class name strings.

    Returns:
        Dictionary containing:
            accuracy, macro_f1, weighted_f1,
            macro_precision, macro_recall,
            per_class_f1, per_class_precision, per_class_recall,
            confusion_matrix, classification_report_str
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_precision": per_class_prec.tolist(),
        "per_class_recall": per_class_rec.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    logger.info(
        f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | "
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    return results


def count_parameters(model) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def estimate_flops(model, input_shape: tuple, device=None) -> Optional[int]:
    """Estimate FLOPs using a forward pass with profiler.

    Falls back to None if torch profiler is unavailable.
    """
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity

        if device is None:
            device = next(model.parameters()).device

        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)

        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                model(dummy_input)

        events = prof.key_averages()
        total_flops = sum(e.flops for e in events if e.flops is not None)
        return total_flops
    except Exception as e:
        logger.debug(f"FLOPs estimation failed: {e}")
        return None


def measure_inference_time(
    model, dataloader, device, n_warmup: int = 5, n_runs: int = 20
) -> Dict[str, float]:
    """Measure average inference time per batch and per sample.

    Runs warmup iterations first, then times multiple passes.
    """
    import torch

    model.eval()
    model.to(device)

    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                x = batch["x"].to(device)
            else:
                x = batch[0].to(device)

            if i < n_warmup:
                model(x)
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append((elapsed, x.shape[0]))

            if i >= n_warmup + n_runs:
                break

    if not times:
        return {"avg_batch_time_ms": 0, "avg_sample_time_ms": 0}

    total_time = sum(t for t, _ in times)
    total_samples = sum(n for _, n in times)
    n_batches = len(times)

    return {
        "avg_batch_time_ms": (total_time / n_batches) * 1000,
        "avg_sample_time_ms": (total_time / total_samples) * 1000,
        "total_inference_s": total_time,
        "n_samples": total_samples,
    }


def aggregate_seed_results(results_list: list) -> Dict[str, Any]:
    """Aggregate metrics across multiple seed runs.

    Args:
        results_list: List of metric dictionaries from different seeds.

    Returns:
        Dictionary with mean ± std for each scalar metric.
    """
    scalar_keys = [
        "accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall",
    ]

    aggregated = {}
    for key in scalar_keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_values"] = values

    # Per-class F1 aggregation
    if "per_class_f1" in results_list[0]:
        per_class = np.array([r["per_class_f1"] for r in results_list])
        aggregated["per_class_f1_mean"] = per_class.mean(axis=0).tolist()
        aggregated["per_class_f1_std"] = per_class.std(axis=0).tolist()

    return aggregated


def aggregate_loso_results(fold_results: list) -> Dict[str, Any]:
    """Aggregate metrics across LOSO folds.

    Args:
        fold_results: List of metric dicts, one per fold.

    Returns:
        Dictionary with mean ± std, plus per-fold breakdown.
    """
    aggregated = aggregate_seed_results(fold_results)
    aggregated["n_folds"] = len(fold_results)

    # Weighted accuracy accounting for different fold sizes
    total_correct = 0
    total_samples = 0
    for r in fold_results:
        n = len(r.get("true_labels", []))
        total_correct += r["accuracy"] * n
        total_samples += n
    if total_samples > 0:
        aggregated["weighted_accuracy"] = total_correct / total_samples

    return aggregated
