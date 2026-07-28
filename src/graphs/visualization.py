"""
Graph visualization utilities for comparing graph construction strategies.

Provides adjacency heatmaps, network topology plots, and structural
similarity metrics for comparing predefined and adaptive graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import torch
from typing import Dict, Optional, Tuple


def _to_numpy(adj) -> np.ndarray:
    """Convert adj_matrix (Tensor or ndarray) to numpy float32."""
    if isinstance(adj, torch.Tensor):
        return adj.detach().cpu().numpy().astype(np.float32)
    return np.asarray(adj, dtype=np.float32)


def plot_adjacency_comparison(
    graphs: Dict[str, dict],
    title: str = "Graph Strategy Comparison",
    cmap: str = "Blues",
    figsize: Tuple[float, float] = None,
    save_path: str = None,
) -> plt.Figure:
    """Plot adjacency matrices side by side as heatmaps.

    Args:
        graphs: {strategy_name: graph_dict} — each dict must have 'adj_matrix'.
        title: Overall figure title.
        cmap: Matplotlib colormap.
        figsize: (width, height). Auto-computed if None.
        save_path: If set, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    n = len(graphs)
    if figsize is None:
        figsize = (4.5 * n, 4.5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, g) in zip(axes, graphs.items()):
        adj = _to_numpy(g["adj_matrix"])
        vmax = adj.max() if adj.max() > 0 else 1.0
        im = ax.imshow(adj, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

        node_names = g.get("node_names", None)
        n_nodes = adj.shape[0]
        if node_names and len(node_names) == n_nodes and n_nodes <= 20:
            ax.set_xticks(range(n_nodes))
            ax.set_yticks(range(n_nodes))
            ax.set_xticklabels(node_names, rotation=90, fontsize=7)
            ax.set_yticklabels(node_names, fontsize=7)
        else:
            ax.set_xlabel("Node index", fontsize=8)
            ax.set_ylabel("Node index", fontsize=8)

        n_edges = int((adj > 1e-8).sum() - n_nodes)  # exclude diagonal
        desc = g.get("description", "")
        ax.set_title(f"{name}\n{desc}\n({n_edges} edges)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_network_comparison(
    graphs: Dict[str, dict],
    node_names: list = None,
    title: str = "Graph Topology Comparison",
    figsize: Tuple[float, float] = None,
    edge_weight_scale: float = 3.0,
    save_path: str = None,
) -> plt.Figure:
    """Plot graph topologies as network diagrams side by side.

    Requires networkx. Uses spring layout with a fixed seed for reproducibility.

    Args:
        graphs: {strategy_name: graph_dict}.
        node_names: Shared node labels (overrides per-graph node_names if given).
        title: Overall figure title.
        figsize: Auto-computed if None.
        edge_weight_scale: Multiplier for edge line widths.
        save_path: If set, save figure to this path.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for network plots: pip install networkx")

    n = len(graphs)
    if figsize is None:
        figsize = (5 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, g) in zip(axes, graphs.items()):
        adj = _to_numpy(g["adj_matrix"])
        n_nodes = adj.shape[0]

        adj_no_diag = adj.copy()
        np.fill_diagonal(adj_no_diag, 0)

        G = nx.from_numpy_array(adj_no_diag, create_using=nx.Graph())

        # Resolve node labels
        labels_source = node_names or g.get("node_names", None)
        if labels_source and len(labels_source) == n_nodes:
            G = nx.relabel_nodes(G, {i: labels_source[i] for i in range(n_nodes)})

        pos = nx.spring_layout(G, seed=42, weight="weight")

        edge_widths = [
            G[u][v].get("weight", 1.0) * edge_weight_scale
            for u, v in G.edges()
        ]
        node_degrees = dict(G.degree(weight="weight"))
        node_sizes = [200 + 300 * node_degrees.get(nd, 0) / max(node_degrees.values(), default=1)
                      for nd in G.nodes()]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color="steelblue", alpha=0.85)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color="gray", alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

        n_edges = G.number_of_edges()
        density = nx.density(G)
        ax.set_title(
            f"{name}\n{g.get('description', '')}\n"
            f"({n_edges} edges, density={density:.2f})",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dtw_distance_matrix(
    graph_dict: dict,
    node_names: list = None,
    title: str = "DTW Distance Matrix",
    save_path: str = None,
) -> plt.Figure:
    """Plot the raw DTW distance matrix returned by build_dtw_graph.

    Args:
        graph_dict: Output of build_dtw_graph (must contain 'dtw_distance_matrix').
        node_names: Channel labels.
        title: Plot title.
        save_path: Optional save path.
    """
    if "dtw_distance_matrix" not in graph_dict:
        raise ValueError("graph_dict must come from build_dtw_graph (contains 'dtw_distance_matrix').")

    D = _to_numpy(graph_dict["dtw_distance_matrix"])
    S = _to_numpy(graph_dict["dtw_similarity_matrix"])
    n_nodes = D.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mat, label, cmap in [
        (ax1, D, "DTW Distance", "Reds"),
        (ax2, S, "Gaussian Similarity", "Blues"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if node_names and len(node_names) == n_nodes and n_nodes <= 20:
            ax.set_xticks(range(n_nodes))
            ax.set_yticks(range(n_nodes))
            ax.set_xticklabels(node_names, rotation=90, fontsize=7)
            ax.set_yticklabels(node_names, fontsize=7)
        else:
            ax.set_xlabel("Channel")
            ax.set_ylabel("Channel")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_graph_similarity(
    graphs: Dict[str, dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute structural statistics and pairwise Jaccard similarity.

    Args:
        graphs: {strategy_name: graph_dict}.

    Returns:
        stats_df: Per-strategy stats (nodes, edges, density, mean_weight).
        jaccard_df: (n_strategies × n_strategies) Jaccard similarity DataFrame.
            Jaccard(A, B) = |edge_set(A) ∩ edge_set(B)| / |edge_set(A) ∪ edge_set(B)|
    """
    names = list(graphs.keys())

    rows = []
    binary_edges = {}
    for name, g in graphs.items():
        adj = _to_numpy(g["adj_matrix"])
        adj_no_diag = adj.copy()
        np.fill_diagonal(adj_no_diag, 0)
        n_nodes = adj.shape[0]
        max_edges = n_nodes * (n_nodes - 1)
        edge_mask = adj_no_diag > 1e-8
        n_edges = edge_mask.sum()
        binary_edges[name] = edge_mask.flatten()
        rows.append({
            "strategy": name,
            "n_nodes": n_nodes,
            "n_edges": int(n_edges),
            "density": float(n_edges / max_edges) if max_edges > 0 else 0.0,
            "mean_weight": float(adj_no_diag[edge_mask].mean()) if n_edges > 0 else 0.0,
            "description": g.get("description", ""),
        })
    stats_df = pd.DataFrame(rows).set_index("strategy")

    n = len(names)
    jaccard = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            b1, b2 = binary_edges[n1], binary_edges[n2]
            inter = (b1 & b2).sum()
            union = (b1 | b2).sum()
            jaccard[i, j] = float(inter) / float(union) if union > 0 else 0.0

    jaccard_df = pd.DataFrame(jaccard, index=names, columns=names)
    return stats_df, jaccard_df


def plot_graph_similarity_heatmap(
    graphs: Dict[str, dict],
    title: str = "Pairwise Graph Jaccard Similarity",
    cmap: str = "YlOrRd",
    save_path: str = None,
) -> plt.Figure:
    """Plot a heatmap of pairwise Jaccard similarity between graph strategies.

    Useful for understanding how topologically different each graph is —
    low Jaccard means the two strategies produce very different edge sets.

    Args:
        graphs: {strategy_name: graph_dict}.
        title: Plot title.
        cmap: Colormap (high contrast recommended, e.g. 'YlOrRd').
        save_path: Optional save path.
    """
    _, jaccard_df = compute_graph_similarity(graphs)
    n = len(jaccard_df)
    figsize = max(5, n * 1.2)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    im = ax.imshow(jaccard_df.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(jaccard_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(jaccard_df.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Jaccard similarity")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jaccard_df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if jaccard_df.values[i, j] < 0.7 else "white")

    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_accuracy_vs_graph(
    results: Dict[str, float],
    graphs: Dict[str, dict],
    metric_name: str = "Test Accuracy",
    title: str = "Accuracy by Graph Strategy",
    save_path: str = None,
) -> plt.Figure:
    """Bar chart comparing model accuracy across graph construction strategies.

    Args:
        results: {strategy_name: accuracy_value}.
        graphs: {strategy_name: graph_dict} — used to annotate bar with edge count.
        metric_name: Y-axis label.
        title: Plot title.
        save_path: Optional save path.
    """
    strategies = list(results.keys())
    values = [results[s] for s in strategies]
    edge_counts = []
    for s in strategies:
        if s in graphs:
            adj = _to_numpy(graphs[s]["adj_matrix"])
            np.fill_diagonal(adj, 0)
            edge_counts.append(int((adj > 1e-8).sum()))
        else:
            edge_counts.append(None)

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 1.5), 5))
    bars = ax.bar(range(len(strategies)), values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, min(1.0, max(values) * 1.15))

    for bar, val, ec in zip(bars, values, edge_counts):
        label = f"{val:.3f}"
        if ec is not None:
            label += f"\n({ec} edges)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
