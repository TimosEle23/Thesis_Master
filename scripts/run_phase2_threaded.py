"""Zombie-free Phase-2 (DSA / LOSO) sweep runner — GPU-friendly.

Mirrors scripts/run_phase1_threaded.py but for Phase 2:
  - Dataset: DSA (45-variate, 8 LOSO folds per experiment).
  - Baselines (graph_mode=none) + TGNNs × {predefined_coarse, predefined_fine,
    adaptive, predefined_correlation_mean, adaptive_correlation_mean,
    adaptive_nosvd_correlation_mean}, over 5 seeds.
  - Each run_experiment call runs all 8 LOSO folds internally and writes the
    aggregated top-level results.json.
  - already_done() checks that aggregated results.json (not per-fold).

Usage (on the DSRI pod):
    cd /workspace/persistent/Thesis_Master
    nohup python3 -u scripts/run_phase2_threaded.py --device cuda --parallel 3 \
        >> logs/phase2_sweep_gpu.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase2"

BASELINES   = ["lstm", "tcn", "transformer", "cnn_bigru"]
TGNN_MODELS = ["gconv_lstm", "gconv_gru", "a3tgcn", "dcrnn", "hybrid_tgnn"]
GRAPH_MODES = [
    "predefined_coarse", "predefined_fine", "adaptive",
    "predefined_correlation_mean", "adaptive_correlation_mean",
    "adaptive_nosvd_correlation_mean",
]
# Scheduling order only: these graph_modes run across all TGNN models
# before the rest of GRAPH_MODES, so this family finishes first.
PRIORITY_GRAPH_MODES = [
    "predefined_correlation_mean", "adaptive_correlation_mean",
    "adaptive_nosvd_correlation_mean",
]
SEEDS       = [42, 123, 456, 789, 1024]


def already_done(dataset: str, model: str, graph_mode: str, seed: int) -> bool:
    d = RESULTS / f"{dataset}_{model}_{graph_mode}_seed{seed}"
    return (d / "results.json").exists()


def make_cmd(dataset, model, graph_mode, seed, device):
    return [sys.executable, "-m", "scripts.run_experiment",
            "--dataset", dataset, "--model", model,
            "--graph_mode", graph_mode, "--seed", str(seed),
            "--phase", "2", "--device", device]


def build_todo(models_baseline=BASELINES, models_tgnn=TGNN_MODELS,
               graph_modes=GRAPH_MODES, seeds=SEEDS):
    ordered_modes = [gm for gm in PRIORITY_GRAPH_MODES if gm in graph_modes] + \
        [gm for gm in graph_modes if gm not in PRIORITY_GRAPH_MODES]
    todo = []
    for m in models_baseline:
        for s in seeds:
            todo.append(("DSA", m, "none", s))
    # graph_mode outer loop so priority modes complete across all models
    # before moving on to the rest.
    for gm in ordered_modes:
        for m in models_tgnn:
            for s in seeds:
                todo.append(("DSA", m, gm, s))
    return todo


def run_one(args):
    dataset, model, graph_mode, seed, device, n_threads = args
    if already_done(dataset, model, graph_mode, seed):
        return dataset, model, graph_mode, seed, 0, 0.0, True
    t0 = time.time()
    cmd = make_cmd(dataset, model, graph_mode, seed, device)
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(n_threads)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False,
                            timeout=43200, env=env)  # 12 hour max (8 folds)
    elapsed = time.time() - t0
    return dataset, model, graph_mode, seed, result.returncode, elapsed, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   default="cuda")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these TGNN models (baselines skipped if set)")
    ap.add_argument("--graph_modes", nargs="+", default=None,
                    help="restrict to these graph modes")
    args = ap.parse_args()

    if args.models:
        plan = build_todo(models_baseline=[], models_tgnn=args.models,
                          graph_modes=args.graph_modes or GRAPH_MODES)
    else:
        plan = build_todo(graph_modes=args.graph_modes or GRAPH_MODES)

    todo = [t for t in plan if not already_done(*t)]

    print("=== Threaded Phase-2 sweep (DSA / LOSO) ===")
    print(f"Planned:  {len(plan)}")
    print(f"Already done: {len(plan) - len(todo)}")
    print(f"To run:   {len(todo)}")
    print(f"Device:   {args.device}    Workers (threads): {args.parallel}")
    print()

    if not todo:
        print("Nothing to do.")
        return

    run_args = [(ds, m, gm, s, args.device, max(1, os.cpu_count() // args.parallel))
                for ds, m, gm, s in todo]
    done_count = 0
    fail_count = 0
    total = len(run_args)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, a): a for a in run_args}
        for fut in as_completed(futures):
            done_count += 1
            try:
                ds, m, gm, s, rc, elapsed, skipped = fut.result()
                status = "skip" if skipped else ("ok" if rc == 0 else f"fail(rc={rc})")
                if rc != 0 and not skipped:
                    fail_count += 1
                print(f"  [{done_count}/{total}] {status:12s} "
                      f"{ds}/{m}/{gm}/seed{s}  ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                fail_count += 1
                a = futures[fut]
                print(f"  [{done_count}/{total}] ERROR {a}: {e}", flush=True)

    print()
    print(f"=== Done: {done_count-fail_count} ok, {fail_count} failed ===")


if __name__ == "__main__":
    main()
