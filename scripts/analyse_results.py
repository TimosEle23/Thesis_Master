"""
Analyse experiment results and generate thesis-ready outputs.

Generates:
  - Comparison tables (LaTeX + CSV)
  - Statistical tests (Wilcoxon, bootstrap CIs)
  - Performance bar plots / heatmaps
  - Training curves
  - Confusion matrices
  - Learned adjacency visualisations
  - Computational cost summary

Usage:
    python -m scripts.analyse_results
    python -m scripts.analyse_results --results_dir results/ --output_dir results/analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logger
from src.evaluation.analysis import (
    load_experiment_results,
    build_comparison_table,
    to_latex_table,
    pairwise_wilcoxon,
    bootstrap_ci,
    run_statistical_analysis,
)

logger = logging.getLogger("thesis")

# ── Plot style ──────────────────────────────────────────────────


def set_thesis_style():
    """Set a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_palette("colorblind")


# ── Performance comparison bar charts ─────────────────────────


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing model accuracies per dataset."""
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        grouped = ds_df.groupby(["model", "graph_mode"])["accuracy"]
        means = grouped.mean().reset_index()
        stds = grouped.std().reset_index()
        means["std"] = stds["accuracy"].fillna(0)

        # Combine model + graph_mode label
        means["label"] = means.apply(
            lambda r: f"{r['model']}\n({r['graph_mode']})" if r["graph_mode"] != "none"
            else r["model"], axis=1
        )

        fig, ax = plt.subplots(figsize=(max(8, len(means) * 0.8), 5))
        colors = []
        for _, row in means.iterrows():
            if row["graph_mode"] == "none":
                colors.append("#4C72B0")
            elif row["graph_mode"] == "predefined" or row["graph_mode"].startswith("predefined"):
                colors.append("#55A868")
            else:
                colors.append("#C44E52")

        bars = ax.bar(
            range(len(means)), means["accuracy"],
            yerr=means["std"], capsize=3,
            color=colors, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means["label"], rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} – Model Comparison")
        ax.set_ylim(0, 1.05)

        # Value annotations
        for bar, val in zip(bars, means["accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / f"accuracy_{ds}.pdf")
        fig.savefig(output_dir / f"accuracy_{ds}.png")
        plt.close(fig)
        logger.info(f"Saved accuracy plot: accuracy_{ds}.pdf")


def plot_f1_comparison(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart for Macro-F1 and Weighted-F1."""
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        grouped = ds_df.groupby(["model", "graph_mode"])[["macro_f1", "weighted_f1"]]
        means = grouped.mean().reset_index()

        means["label"] = means.apply(
            lambda r: f"{r['model']}\n({r['graph_mode']})" if r["graph_mode"] != "none"
            else r["model"], axis=1
        )

        fig, ax = plt.subplots(figsize=(max(10, len(means) * 1.2), 5))
        x = np.arange(len(means))
        w = 0.35

        ax.bar(x - w/2, means["macro_f1"], w, label="Macro-F1", color="#4C72B0", alpha=0.85)
        ax.bar(x + w/2, means["weighted_f1"], w, label="Weighted-F1", color="#55A868", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(means["label"], rotation=45, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{ds} – F1 Score Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"f1_{ds}.pdf")
        fig.savefig(output_dir / f"f1_{ds}.png")
        plt.close(fig)


# ── Confusion matrices ────────────────────────────────────────


def plot_confusion_matrices(results_dir: Path, output_dir: Path):
    """Plot confusion matrices and save classification reports from saved labels."""
    cm_dir = output_dir / "confusion_matrices"
    cr_dir = output_dir / "classification_reports"
    cm_dir.mkdir(exist_ok=True)
    cr_dir.mkdir(exist_ok=True)

    for exp_dir in sorted(results_dir.glob("**/predictions.npy")):
        y_pred = np.load(exp_dir)
        true_labels_path = exp_dir.parent / "true_labels.npy"
        
        if not true_labels_path.exists():
            continue
            
        y_true = np.load(true_labels_path)
        name = exp_dir.parent.name

        # Calculate using sklearn
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        
        # Save Classification Report
        report = classification_report(y_true, y_pred)
        with open(cr_dir / f"report_{name}.txt", "w") as f:
            f.write(f"Classification Report for: {name}\n")
            f.write("="*50 + "\n")
            f.write(report)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.5), max(5, n_classes * 0.4)))

        # Normalise per row (per true class)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(
            cm_norm, annot=cm, fmt="d",
            cmap="Blues", ax=ax,
            xticklabels=range(n_classes),
            yticklabels=range(n_classes),
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {name}")

        fig.tight_layout()
        fig.savefig(cm_dir / f"cm_{name}.pdf")
        plt.close(fig)

    logger.info(f"Confusion matrices saved to {cm_dir}")
    logger.info(f"Classification reports saved to {cr_dir}")


# ── Training curves ───────────────────────────────────────────


def plot_training_curves(results_dir: Path, output_dir: Path):
    """Plot training/validation loss and accuracy curves."""
    curves_dir = output_dir / "training_curves"
    curves_dir.mkdir(exist_ok=True)

    for history_file in sorted(results_dir.glob("**/history.json")):
        with open(history_file) as f:
            history = json.load(f)

        name = history_file.parent.name
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
        ax1.plot(epochs, history["val_loss"], label="Validation", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{name} – Loss")
        ax1.legend()

        # Accuracy
        ax2.plot(epochs, history["train_acc"], label="Train", linewidth=1.5)
        ax2.plot(epochs, history["val_acc"], label="Validation", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{name} – Accuracy")
        ax2.legend()
        ax2.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(curves_dir / f"curve_{name}.pdf")
        plt.close(fig)

    logger.info(f"Training curves saved to {curves_dir}")


# ── Computational cost ────────────────────────────────────────


def plot_cost_comparison(df: pd.DataFrame, output_dir: Path):
    """Scatter: accuracy vs #parameters, coloured by model type."""
    if "n_parameters" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        is_tgnn = model in ["gconv_lstm", "gconv_gru", "a3tgcn", "dcrnn"]
        marker = "^" if is_tgnn else "o"
        ax.scatter(
            sub["n_parameters"], sub["accuracy"],
            label=model, marker=marker, s=60, alpha=0.8,
        )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Model Size")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "cost_vs_accuracy.pdf")
    fig.savefig(output_dir / "cost_vs_accuracy.png")
    plt.close(fig)
    logger.info("Saved cost vs accuracy plot")


# ── RQ-specific comparison: predefined vs adaptive ───────────


def plot_graph_mode_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare predefined vs adaptive graph modes (RQ2)."""
    tgnn_df = df[df["model"].isin(["gconv_lstm", "gconv_gru", "a3tgcn", "dcrnn"])]
    if tgnn_df.empty:
        return

    for ds in tgnn_df["dataset"].unique():
        ds_df = tgnn_df[tgnn_df["dataset"] == ds]
        pivot = ds_df.groupby(["model", "graph_mode"])["accuracy"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        models = pivot["model"].unique()
        x = np.arange(len(models))
        gm_list = sorted(pivot["graph_mode"].unique())
        width = 0.8 / len(gm_list)

        for i, gm in enumerate(gm_list):
            sub = pivot[pivot["graph_mode"] == gm]
            vals = []
            errs = []
            for m in models:
                row = sub[sub["model"] == m]
                vals.append(row["mean"].values[0] if len(row) > 0 else 0)
                errs.append(row["std"].values[0] if len(row) > 0 else 0)

            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=gm, capsize=3, alpha=0.85)

        ax.set_xticks(x + width * (len(gm_list) - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} – Predefined vs Adaptive (RQ2)")
        ax.legend()
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(output_dir / f"rq2_graph_modes_{ds}.pdf")
        fig.savefig(output_dir / f"rq2_graph_modes_{ds}.png")
        plt.close(fig)

    logger.info("Saved RQ2 graph mode comparison plots")


# ── Heatmap: model × dataset ─────────────────────────────────


def plot_performance_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of accuracy across models and datasets."""
    # Create a combined label
    df_copy = df.copy()
    df_copy["config"] = df_copy.apply(
        lambda r: f"{r['model']}_{r['graph_mode']}" if r["graph_mode"] != "none"
        else r["model"], axis=1
    )
    pivot = df_copy.pivot_table(values="accuracy", index="config", columns="dataset", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(6, len(pivot) * 0.4)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        ax=ax, linewidths=0.5, vmin=0, vmax=1,
    )
    ax.set_title("Accuracy Heatmap: Model × Dataset")
    ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_heatmap.pdf")
    fig.savefig(output_dir / "accuracy_heatmap.png")
    plt.close(fig)
    logger.info("Saved accuracy heatmap")


# ── Main analysis pipeline ────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyse experiment results")
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "analysis"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_dir=str(output_dir))
    set_thesis_style()

    logger.info(f"Loading results from {results_dir}")

    # 1. Load all results into DataFrame
    df = load_experiment_results(str(results_dir))
    if df.empty:
        logger.error("No results found. Run experiments first.")
        return

    logger.info(f"Loaded {len(df)} result entries")
    logger.info(f"Datasets: {df['dataset'].unique().tolist()}")
    logger.info(f"Models: {df['model'].unique().tolist()}")

    # Save raw results table
    df.to_csv(output_dir / "all_results.csv", index=False)

    # 2. Comparison tables
    logger.info("\n--- Comparison Tables ---")
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        table = build_comparison_table(ds_df, group_cols=["model", "graph_mode"])
        table.to_csv(output_dir / f"table_{ds}.csv", index=False)

        latex = to_latex_table(
            table,
            caption=f"Classification performance on {ds}",
            label=f"tab:{ds.lower()}_results",
        )
        (output_dir / f"table_{ds}.tex").write_text(latex)
        logger.info(f"  Table saved: table_{ds}.csv / .tex")

    # 3. Statistical analysis
    logger.info("\n--- Statistical Analysis ---")
    stats = run_statistical_analysis(str(results_dir), str(output_dir))

    # 4. Visualisations
    logger.info("\n--- Generating Plots ---")
    plot_accuracy_comparison(df, output_dir)
    plot_f1_comparison(df, output_dir)
    plot_confusion_matrices(results_dir, output_dir)
    plot_training_curves(results_dir, output_dir)
    plot_cost_comparison(df, output_dir)
    plot_graph_mode_comparison(df, output_dir)
    plot_performance_heatmap(df, output_dir)

    # 5. Summary report
    logger.info("\n--- Summary ---")
    summary_lines = ["# Experiment Analysis Summary\n"]

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        best_row = ds_df.loc[ds_df["accuracy"].idxmax()]
        summary_lines.append(f"## {ds}")
        summary_lines.append(f"- Best model: {best_row['model']} ({best_row['graph_mode']})")
        summary_lines.append(f"- Best accuracy: {best_row['accuracy']:.4f}")
        summary_lines.append(
            f"- Baselines: {ds_df[ds_df['model'].isin(['lstm','tcn','transformer'])]['accuracy'].mean():.4f} mean"
        )
        tgnn_acc = ds_df[ds_df['model'].isin(['gconv_lstm','gconv_gru','a3tgcn','dcrnn'])]['accuracy']
        if len(tgnn_acc) > 0:
            summary_lines.append(f"- TGNNs: {tgnn_acc.mean():.4f} mean")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    (output_dir / "summary.md").write_text(summary_text)
    logger.info(f"\nAnalysis complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
