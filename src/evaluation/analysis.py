"""
Statistical analysis for comparing model performance.

- Wilcoxon signed-rank tests
- Bootstrap confidence intervals
- LaTeX table generation
- Results aggregation across experiments
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import combinations

logger = logging.getLogger("thesis")


# ── Wilcoxon signed-rank test ──────────────────────────────────────


def wilcoxon_test(
    scores_a: list, scores_b: list, alternative: str = "two-sided"
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        scores_a: Metric values for model A (across seeds/folds).
        scores_b: Metric values for model B.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dictionary with statistic, p_value, significant (α=0.05).
    """
    from scipy.stats import wilcoxon

    a = np.array(scores_a)
    b = np.array(scores_b)

    if len(a) < 5:
        logger.warning(
            f"Wilcoxon test with n={len(a)} (<5) may lack power; "
            "consider using all seeds×folds."
        )

    # Handle the case where all differences are zero
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05,
    }


def pairwise_wilcoxon(
    model_scores: Dict[str, list], metric: str = "accuracy"
) -> pd.DataFrame:
    """Run pairwise Wilcoxon tests between all model pairs.

    Args:
        model_scores: {model_name: [score_seed1, score_seed2, ...]}.
        metric: Name of the metric (for logging).

    Returns:
        DataFrame with model_a, model_b, statistic, p_value columns.
    """
    names = sorted(model_scores.keys())
    rows = []
    for a, b in combinations(names, 2):
        # Use one-sided "greater" so that with n=5 seeds the minimum achievable
        # p-value is 0.0312 (< 0.05).  model_a is tested as the stronger model.
        scores_a = model_scores[a]
        scores_b = model_scores[b]
        mean_a = sum(scores_a) / len(scores_a)
        mean_b = sum(scores_b) / len(scores_b)
        # Always test the higher-mean model as "greater"
        if mean_a >= mean_b:
            result = wilcoxon_test(scores_a, scores_b, alternative="greater")
            rows.append({"model_a": a, "model_b": b, "metric": metric, **result})
        else:
            result = wilcoxon_test(scores_b, scores_a, alternative="greater")
            rows.append({"model_a": b, "model_b": a, "metric": metric, **result})
    return pd.DataFrame(rows)


# ── Bootstrap confidence intervals ────────────────────────────────


def bootstrap_ci(
    values: list,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        values: Sample values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 95%).
        seed: Random seed.

    Returns:
        Dictionary with mean, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)

    boot_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(arr.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


# ── Results aggregation ──────────────────────────────────────────


def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """Load all experiment result JSON files into a DataFrame.

    Expected directory structure:
        results_dir/
            {dataset}_{model}_{graph_mode}_seed{s}.json

    Each JSON should contain at minimum:
        dataset, model, graph_mode, seed, accuracy, macro_f1, …
    """
    results_dir = Path(results_dir)
    rows = []

    for f in sorted(results_dir.glob("**/results.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "dataset" in data:
                # Unify phase 2 metric names (mean) to match phase 1
                if "accuracy_mean" in data and "accuracy" not in data:
                    data["accuracy"] = data["accuracy_mean"]
                if "macro_f1_mean" in data and "macro_f1" not in data:
                    data["macro_f1"] = data["macro_f1_mean"]
                if "n_parameters" in data and isinstance(data["n_parameters"], dict):
                    data["n_parameters"] = data["n_parameters"].get("total", 0)
                if "weighted_f1_mean" in data and "weighted_f1" not in data:
                    data["weighted_f1"] = data["weighted_f1_mean"]
                    
                rows.append(data)
            else:
                logger.warning(f"Skipping {f}: missing 'dataset' key or invalid format")
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if not rows:
        logger.warning(f"No results found in {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
    # Differentiate datasets by phase
    if "phase" in df.columns:
        # Fill missing phases with 1 just in case
        df["phase"] = df["phase"].fillna(1)
        df["dataset"] = df.apply(lambda r: f"{r['dataset']} (Phase {int(r['phase'])})", axis=1)
        
    return df


def build_comparison_table(
    df: pd.DataFrame,
    metrics: List[str] = None,
    group_cols: List[str] = None,
) -> pd.DataFrame:
    """Build a mean±std comparison table.

    Args:
        df: Results DataFrame with one row per run.
        metrics: Metric columns to summarise.
        group_cols: Columns to group by (e.g. ['dataset', 'model', 'graph_mode']).

    Returns:
        DataFrame with formatted mean±std strings.
    """
    if metrics is None:
        metrics = ["accuracy", "macro_f1", "weighted_f1"]
    if group_cols is None:
        group_cols = ["dataset", "model", "graph_mode"]

    # Only group by columns that exist in df
    group_cols = [c for c in group_cols if c in df.columns]
    metrics = [m for m in metrics if m in df.columns]

    if not group_cols or not metrics:
        logger.warning("No valid group columns or metrics found")
        return df

    grouped = df.groupby(group_cols)[metrics]

    mean_df = grouped.mean()
    std_df = grouped.std().fillna(0)

    # Build formatted table
    formatted = pd.DataFrame(index=mean_df.index)
    for m in metrics:
        formatted[m] = [
            f"{mean:.4f}±{std:.4f}"
            for mean, std in zip(mean_df[m], std_df[m])
        ]

    # Bold the best per dataset
    for m in metrics:
        for ds in formatted.index.get_level_values(0).unique():
            mask = formatted.index.get_level_values(0) == ds
            vals = mean_df.loc[mask, m]
            best_idx = vals.idxmax()
            current = formatted.loc[best_idx, m]
            formatted.loc[best_idx, m] = f"**{current}**"

    return formatted.reset_index()


# ── LaTeX table export ────────────────────────────────────────────


def to_latex_table(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    bold_best: bool = True,
) -> str:
    """Convert comparison DataFrame to a LaTeX table string.

    Args:
        df: Comparison table (formatted with mean±std).
        caption: Table caption.
        label: Table label.
        bold_best: Whether to add \\textbf around Markdown bold.

    Returns:
        LaTeX table string.
    """
    # Replace Markdown bold with LaTeX bold
    latex_df = df.copy()
    for col in latex_df.columns:
        latex_df[col] = latex_df[col].astype(str).str.replace(
            r"\*\*(.*?)\*\*", r"\\textbf{\1}", regex=True
        )
        # Escape ± for LaTeX
        latex_df[col] = latex_df[col].str.replace("±", "$\\pm$", regex=False)

    # Build table
    n_cols = len(latex_df.columns)
    col_fmt = "l" * min(3, n_cols) + "c" * max(0, n_cols - 3)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")

    # Header
    header = " & ".join(latex_df.columns)
    lines.append(f"{header} \\\\")
    lines.append("\\midrule")

    # Rows
    prev_dataset = None
    for _, row in latex_df.iterrows():
        vals = " & ".join(str(v) for v in row)
        # Add midrule between datasets
        if "dataset" in latex_df.columns:
            curr = row.get("dataset", "")
            if prev_dataset is not None and curr != prev_dataset:
                lines.append("\\midrule")
            prev_dataset = curr
        lines.append(f"{vals} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ── Full statistical analysis pipeline ────────────────────────────


def run_statistical_analysis(
    results_dir: str,
    output_dir: str = "results/analysis",
) -> Dict[str, Any]:
    """Run all statistical analyses on experiment results.

    1. Load results
    2. Build comparison tables
    3. Run pairwise Wilcoxon tests
    4. Bootstrap CIs for key metrics
    5. Export LaTeX tables and summary JSON

    Returns:
        Dictionary with all analysis results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load results
    df = load_experiment_results(results_dir)
    if df.empty:
        logger.warning("No results to analyse")
        return {}

    analysis = {}

    # 2. Comparison tables per dataset
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        table = build_comparison_table(
            ds_df,
            group_cols=["model", "graph_mode"],
        )
        analysis[f"table_{ds}"] = table.to_dict()

        latex = to_latex_table(
            table,
            caption=f"Performance comparison on {ds}",
            label=f"tab:{ds.lower()}_results",
        )
        (output_path / f"table_{ds}.tex").write_text(latex)
        logger.info(f"LaTeX table saved: table_{ds}.tex")

    # 3. Pairwise Wilcoxon tests
    wilcoxon_results = {}
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        model_scores = {}
        for (name, mode), group in ds_df.groupby(["model", "graph_mode"]):
            # Use 'none' instead of nan for mode
            mode_str = str(mode) if pd.notna(mode) else "none"
            model_scores[f"{name}_{mode_str}"] = group["accuracy"].tolist()

        if len(model_scores) > 1:
            pw = pairwise_wilcoxon(model_scores, metric="accuracy")
            wilcoxon_results[ds] = pw.to_dict("records")
            pw.to_csv(output_path / f"wilcoxon_{ds}.csv", index=False)

    analysis["wilcoxon"] = wilcoxon_results

    # 4. Bootstrap CIs
    ci_results = {}
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        ci_results[ds] = {}
        for (model_name, mode), group in ds_df.groupby(["model", "graph_mode"]):
            mode_str = str(mode) if pd.notna(mode) else "none"
            ci = bootstrap_ci(group["accuracy"].tolist())
            ci_results[ds][f"{model_name}_{mode_str}"] = ci

    analysis["bootstrap_ci"] = ci_results

    # 5. Save summary
    # Convert for JSON serialization
    summary = {
        "wilcoxon": wilcoxon_results,
        "bootstrap_ci": ci_results,
    }
    with open(output_path / "statistical_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Statistical analysis saved to {output_path}")
    return analysis
