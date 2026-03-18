"""
Run ALL experiments for both Phase 1 and Phase 2.

Orchestrates every combination of:
  - Dataset × Model × Graph mode × Seed

Usage:
    python -m scripts.run_all                          # Run everything
    python -m scripts.run_all --phase 1                # Phase 1 only
    python -m scripts.run_all --phase 2                # Phase 2 only
    python -m scripts.run_all --phase 1 --dry_run      # Preview commands
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logger
from src.data.download import download_all_datasets
from src.models.factory import BASELINE_MODELS, TGNN_MODELS
from scripts.run_experiment import load_config, run_uea_experiment, run_dsa_experiment

logger = logging.getLogger("thesis")


def get_phase1_jobs(config: dict) -> list:
    """Generate all Phase 1 experiment jobs."""
    seeds = config["experiment"]["seeds"]
    datasets = ["BasicMotions", "Epilepsy"]
    baselines = config.get("models_to_run", {}).get("baselines", BASELINE_MODELS)
    tgnns = config.get("models_to_run", {}).get("tgnns", TGNN_MODELS)
    graph_modes = config.get("models_to_run", {}).get("graph_modes", ["predefined", "adaptive"])

    jobs = []
    for ds in datasets:
        # Baseline models (no graph)
        for model in baselines:
            for seed in seeds:
                jobs.append({
                    "phase": 1, "dataset": ds, "model": model,
                    "graph_mode": "none", "seed": seed,
                })
        # TGNN models (with graph modes)
        for model in tgnns:
            for gm in graph_modes:
                for seed in seeds:
                    jobs.append({
                        "phase": 1, "dataset": ds, "model": model,
                        "graph_mode": gm, "seed": seed,
                    })
    return jobs


def get_phase2_jobs(config: dict) -> list:
    """Generate all Phase 2 experiment jobs."""
    seeds = config["experiment"]["seeds"]
    baselines = config.get("models_to_run", {}).get("baselines", BASELINE_MODELS)
    tgnns = config.get("models_to_run", {}).get("tgnns", TGNN_MODELS)
    graph_modes = config.get("models_to_run", {}).get(
        "graph_modes", ["predefined_coarse", "predefined_fine", "adaptive"]
    )

    jobs = []
    # Baselines
    for model in baselines:
        for seed in seeds:
            jobs.append({
                "phase": 2, "dataset": "DSA", "model": model,
                "graph_mode": "none", "seed": seed,
            })
    # TGNNs
    for model in tgnns:
        for gm in graph_modes:
            for seed in seeds:
                jobs.append({
                    "phase": 2, "dataset": "DSA", "model": model,
                    "graph_mode": gm, "seed": seed,
                })
    return jobs


def run_all(phases: list, dry_run: bool = False, max_folds: int = None):
    """Run all experiments for the specified phases."""
    all_results = []

    for phase in phases:
        config = load_config(phase)
        results_dir = PROJECT_ROOT / "results" / f"phase{phase}"
        results_dir.mkdir(parents=True, exist_ok=True)

        if phase == 1:
            jobs = get_phase1_jobs(config)
        else:
            jobs = get_phase2_jobs(config)

        logger.info(f"\n{'='*70}")
        logger.info(f"  PHASE {phase}: {len(jobs)} experiments")
        logger.info(f"{'='*70}")

        if dry_run:
            for i, j in enumerate(jobs):
                logger.info(
                    f"  [{i+1:3d}/{len(jobs)}] "
                    f"{j['dataset']:15s} | {j['model']:12s} | "
                    f"{j['graph_mode']:20s} | seed={j['seed']}"
                )
            continue

        # Check which experiments already have results
        pending = []
        for j in jobs:
            gm = j["graph_mode"]
            exp_name = f"{j['dataset']}_{j['model']}_{gm}_seed{j['seed']}"
            result_file = results_dir / exp_name / "results.json"
            if result_file.exists():
                logger.info(f"  SKIP (exists): {exp_name}")
                with open(result_file) as f:
                    all_results.append(json.load(f))
            else:
                pending.append(j)

        logger.info(f"  Pending: {len(pending)}, Skipped: {len(jobs)-len(pending)}")

        for i, job in enumerate(pending):
            logger.info(
                f"\n--- Job [{i+1}/{len(pending)}] "
                f"{job['dataset']} / {job['model']} / {job['graph_mode']} / seed={job['seed']} ---"
            )

            start = time.time()
            try:
                if job["dataset"] in ["BasicMotions", "Epilepsy"]:
                    result = run_uea_experiment(
                        config, job["dataset"], job["model"],
                        job["graph_mode"], job["seed"], results_dir,
                    )
                else:
                    result = run_dsa_experiment(
                        config, job["model"], job["graph_mode"],
                        job["seed"], results_dir, max_folds=max_folds,
                    )
                all_results.append(result)
                elapsed = time.time() - start
                acc = result.get("accuracy", result.get("accuracy_mean", "?"))
                logger.info(f"  Completed in {elapsed:.1f}s | Accuracy: {acc}")

            except Exception as e:
                logger.error(f"  FAILED: {e}", exc_info=True)
                all_results.append({
                    **job, "error": str(e), "accuracy": None,
                })

    # Save summary of all results
    summary_path = PROJECT_ROOT / "results" / "all_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results summary saved to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2],
                        help="Run only this phase (default: both)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print the experiment plan, don't run")
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Max LOSO folds for DSA (default: all 8)")
    args = parser.parse_args()

    # Setup
    log_dir = PROJECT_ROOT / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir))

    # Download data first
    if not args.dry_run:
        download_all_datasets()

    phases = [args.phase] if args.phase else [1, 2]
    results = run_all(phases, dry_run=args.dry_run, max_folds=args.max_folds)

    if not args.dry_run:
        # Quick summary
        logger.info(f"\n{'='*70}")
        logger.info("SUMMARY")
        logger.info(f"{'='*70}")
        for r in results:
            acc = r.get("accuracy", r.get("accuracy_mean", "ERROR"))
            status = "OK" if acc is not None and acc != "ERROR" else "FAIL"
            logger.info(
                f"  [{status}] {r.get('dataset','?'):15s} | "
                f"{r.get('model','?'):12s} | {r.get('graph_mode','?'):20s} | "
                f"seed={r.get('seed','?')} | acc={acc}"
            )


if __name__ == "__main__":
    main()
