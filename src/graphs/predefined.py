"""
Predefined graph construction for TGNN models.

Constructs adjacency matrices based on domain knowledge:
  - BasicMotions: Sensor-axis graph (6 nodes: AccXYZ + GyroXYZ)
  - Epilepsy: Fully connected (3 nodes: AccXYZ)
  - DSA coarse: Anatomical body-segment graph (5 nodes)
  - DSA fine: Channel-level graph (45 nodes) with intra/inter-unit edges

This module directly addresses RQ2 by providing the "predefined domain-knowledge"
graph construction strategy.
"""

import numpy as np
import torch
import logging

logger = logging.getLogger("thesis")


def build_predefined_graph(
    dataset_name: str,
    granularity: str = "default",
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Build a predefined adjacency graph for a dataset.
    
    Args:
        dataset_name: One of 'BasicMotions', 'Epilepsy', 'DSA'.
        granularity: For DSA: 'coarse' (5 nodes) or 'fine' (45 nodes).
        self_loops: Whether to add self-loops.
        normalise: Whether to apply symmetric normalisation.
        
    Returns:
        Dict with:
            edge_index: (2, n_edges) LongTensor
            edge_weight: (n_edges,) FloatTensor
            adj_matrix: (n_nodes, n_nodes) FloatTensor
            num_nodes: int
            node_features_dim: int
            node_names: list of str
            description: str
    """
    if dataset_name == "BasicMotions":
        return _build_basic_motions_graph(self_loops, normalise)
    elif dataset_name == "Epilepsy":
        return _build_epilepsy_graph(self_loops, normalise)
    elif dataset_name == "DSA":
        if granularity == "coarse":
            return _build_dsa_coarse_graph(self_loops, normalise)
        elif granularity == "biomechanical":
            return _build_dsa_biomechanical_graph(self_loops, normalise)
        elif granularity == "fine":
            return _build_dsa_fine_graph(self_loops, normalise)
        else:
            return _build_dsa_coarse_graph(self_loops, normalise)
    else:
        raise ValueError(f"No predefined graph for dataset: {dataset_name}")


def _build_basic_motions_graph(self_loops: bool, normalise: bool) -> dict:
    """BasicMotions: 6 nodes (AccX, AccY, AccZ, GyroX, GyroY, GyroZ).
    
    Graph structure:
      - Intra-type: All Acc axes connected, all Gyro axes connected
      - Cross-type: Same axis connected (AccX-GyroX, AccY-GyroY, AccZ-GyroZ)
    """
    num_nodes = 6
    node_names = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    
    # Undirected edges (both directions)
    edges = [
        # Intra-accelerometer
        (0, 1), (0, 2), (1, 2),
        # Intra-gyroscope
        (3, 4), (3, 5), (4, 5),
        # Cross-type same axis
        (0, 3), (1, 4), (2, 5),
    ]
    
    adj = _edges_to_adj(edges, num_nodes, self_loops, normalise)
    edge_index, edge_weight = _adj_to_edge_index(adj)
    
    logger.info(
        f"BasicMotions predefined graph: {num_nodes} nodes, "
        f"{edge_index.shape[1]} edges"
    )
    
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj,
        "num_nodes": num_nodes,
        "node_features_dim": 1,
        "node_names": node_names,
        "description": "Sensor-axis graph: intra-type + cross-type same-axis edges",
    }


def _build_epilepsy_graph(self_loops: bool, normalise: bool) -> dict:
    """Epilepsy: 3 nodes (AccX, AccY, AccZ), fully connected.
    
    Since all axes come from the same accelerometer sensor,
    all pairs are connected.
    """
    num_nodes = 3
    node_names = ["AccX", "AccY", "AccZ"]
    
    edges = [(0, 1), (0, 2), (1, 2)]
    
    adj = _edges_to_adj(edges, num_nodes, self_loops, normalise)
    edge_index, edge_weight = _adj_to_edge_index(adj)
    
    logger.info(
        f"Epilepsy predefined graph: {num_nodes} nodes, "
        f"{edge_index.shape[1]} edges"
    )
    
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj,
        "num_nodes": num_nodes,
        "node_features_dim": 1,
        "node_names": node_names,
        "description": "Fully connected 3-axis accelerometer graph",
    }


def _build_dsa_coarse_graph(self_loops: bool, normalise: bool) -> dict:
    """DSA coarse: 5 nodes (body segments), anatomical adjacency.
    
    Nodes: Torso(0), RightArm(1), LeftArm(2), RightLeg(3), LeftLeg(4)
    
    Edges encode anatomical adjacency:
      - Torso connects to all limbs (central hub through shoulder/hip joints)
      - Bilateral symmetry: RightArm-LeftArm, RightLeg-LeftLeg
    
    Each node has 9 features (acc_xyz + gyro_xyz + mag_xyz).
    """
    num_nodes = 5
    node_names = ["Torso", "RightArm", "LeftArm", "RightLeg", "LeftLeg"]
    
    edges = [
        (0, 1),  # Torso - RightArm (right shoulder)
        (0, 2),  # Torso - LeftArm (left shoulder)
        (0, 3),  # Torso - RightLeg (right hip)
        (0, 4),  # Torso - LeftLeg (left hip)
        (1, 2),  # RightArm - LeftArm (bilateral symmetry)
        (3, 4),  # RightLeg - LeftLeg (bilateral symmetry)
    ]
    
    adj = _edges_to_adj(edges, num_nodes, self_loops, normalise)
    edge_index, edge_weight = _adj_to_edge_index(adj)
    
    logger.info(
        f"DSA coarse predefined graph: {num_nodes} nodes, "
        f"{edge_index.shape[1]} edges"
    )
    
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj,
        "num_nodes": num_nodes,
        "node_features_dim": 9,
        "node_names": node_names,
        "sensor_groups": {
            "torso": list(range(0, 9)),
            "right_arm": list(range(9, 18)),
            "left_arm": list(range(18, 27)),
            "right_leg": list(range(27, 36)),
            "left_leg": list(range(36, 45)),
        },
        "description": "Anatomical body-segment graph, 5 nodes with 9 features each",
    }


def _build_dsa_biomechanical_graph(self_loops: bool, normalise: bool) -> dict:
    """DSA biomechanical: 5 nodes, richer anatomical connectivity.

    Nodes: Torso(0), RightArm(1), LeftArm(2), RightLeg(3), LeftLeg(4)

    Edges follow the Auckland EEG-ADG paper's Interconnected / Analogous /
    Lateral relation scheme adapted to body-segment kinematics:
      - Torso hub: connects to all four limbs  (Interconnected)
      - Arm bilateral: RightArm ↔ LeftArm     (Analogous)
      - Leg bilateral: RightLeg ↔ LeftLeg     (Analogous)
      - Same-side (Lateral): RightArm ↔ RightLeg, LeftArm ↔ LeftLeg

    Same-side ipsilateral links capture gait coordination (arm swing matches
    opposite leg but same-side chains carry correlated inertial loads).
    """
    num_nodes = 5
    node_names = ["Torso", "RightArm", "LeftArm", "RightLeg", "LeftLeg"]

    edges = [
        # Torso hub (Interconnected)
        (0, 1),  # Torso - RightArm
        (0, 2),  # Torso - LeftArm
        (0, 3),  # Torso - RightLeg
        (0, 4),  # Torso - LeftLeg
        # Bilateral (Analogous)
        (1, 2),  # RightArm - LeftArm
        (3, 4),  # RightLeg - LeftLeg
        # Same-side ipsilateral (Lateral)
        (1, 3),  # RightArm - RightLeg
        (2, 4),  # LeftArm  - LeftLeg
    ]

    adj = _edges_to_adj(edges, num_nodes, self_loops, normalise)
    edge_index, edge_weight = _adj_to_edge_index(adj)

    logger.info(
        f"DSA biomechanical graph: {num_nodes} nodes, "
        f"{edge_index.shape[1]} edges (hub+bilateral+lateral)"
    )

    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj,
        "num_nodes": num_nodes,
        "node_features_dim": 9,
        "node_names": node_names,
        "sensor_groups": {
            "torso": list(range(0, 9)),
            "right_arm": list(range(9, 18)),
            "left_arm": list(range(18, 27)),
            "right_leg": list(range(27, 36)),
            "left_leg": list(range(36, 45)),
        },
        "description": "Biomechanical body-segment graph: hub + bilateral + lateral ipsilateral edges",
    }


def _build_dsa_fine_graph(self_loops: bool, normalise: bool) -> dict:
    """DSA fine: 45 nodes (individual sensor channels).
    
    Edges:
      1. Intra-unit: All 9 channels within same sensor unit are fully connected
      2. Inter-unit: Matching channels between anatomically adjacent units
      
    Unit layout (each has 9 channels at offsets 0-8):
      Torso: 0-8, RightArm: 9-17, LeftArm: 18-26, RightLeg: 27-35, LeftLeg: 36-44
      
    Anatomical adjacency between units:
      Torso-RightArm, Torso-LeftArm, Torso-RightLeg, Torso-LeftLeg,
      RightArm-LeftArm, RightLeg-LeftLeg
    """
    num_nodes = 45
    unit_offsets = [0, 9, 18, 27, 36]
    channels_per_unit = 9
    
    # Node names
    unit_names = ["T", "RA", "LA", "RL", "LL"]
    channel_types = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                     "mag_x", "mag_y", "mag_z"]
    node_names = [f"{u}_{c}" for u in unit_names for c in channel_types]
    
    # Anatomical adjacency between units
    unit_adjacency = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Torso to all limbs
        (1, 2),  # Arms bilateral
        (3, 4),  # Legs bilateral
    ]
    
    edges = []
    
    # 1. Intra-unit: fully connected within each unit
    for offset in unit_offsets:
        for i in range(channels_per_unit):
            for j in range(i + 1, channels_per_unit):
                edges.append((offset + i, offset + j))
    
    # 2. Inter-unit: matching channels between adjacent units
    for u1, u2 in unit_adjacency:
        o1, o2 = unit_offsets[u1], unit_offsets[u2]
        for ch in range(channels_per_unit):
            edges.append((o1 + ch, o2 + ch))
    
    adj = _edges_to_adj(edges, num_nodes, self_loops, normalise)
    edge_index, edge_weight = _adj_to_edge_index(adj)
    
    logger.info(
        f"DSA fine predefined graph: {num_nodes} nodes, "
        f"{edge_index.shape[1]} edges"
    )
    
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj,
        "num_nodes": num_nodes,
        "node_features_dim": 1,
        "node_names": node_names,
        "description": "Channel-level graph with intra-unit + inter-unit anatomical edges",
    }


# ---- Helper functions ----

def _edges_to_adj(
    edges: list,
    num_nodes: int,
    self_loops: bool = True,
    normalise: bool = True,
) -> torch.FloatTensor:
    """Convert edge list to adjacency matrix.
    
    Args:
        edges: List of (i, j) tuples (undirected).
        num_nodes: Number of nodes.
        self_loops: Add self-loops.
        normalise: Apply symmetric normalisation: D^{-1/2} A D^{-1/2}
        
    Returns:
        FloatTensor adjacency matrix of shape (num_nodes, num_nodes).
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # Undirected
    
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    
    if normalise:
        adj = _symmetric_normalise(adj)
    
    return torch.FloatTensor(adj)


def _symmetric_normalise(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}.

    Standard GCN-style normalisation from Kipf & Welling (2017).
    """
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, degree ** (-0.5), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def _symmetric_normalise_signed(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation for signed adjacency matrices.

    Uses |degree| for the normalisation denominator so that negative edges
    don't cancel positive ones in the row sum.  Sign is preserved on the
    normalised weights.
    """
    abs_degree = np.abs(adj).sum(axis=1)
    degree_inv_sqrt = np.where(abs_degree > 0, abs_degree ** (-0.5), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def _adj_to_edge_index(adj: torch.FloatTensor) -> tuple:
    """Convert adjacency matrix to edge_index and edge_weight.

    Handles signed adjacency (negative edge weights from signed-correlation
    graphs) by thresholding on |weight| rather than weight > 0.

    Returns:
        edge_index: LongTensor of shape (2, n_edges)
        edge_weight: FloatTensor of shape (n_edges,)
    """
    indices = (adj.abs() > 1e-8).nonzero(as_tuple=False)
    edge_index = indices.t().contiguous().long()
    edge_weight = adj[indices[:, 0], indices[:, 1]]
    return edge_index, edge_weight


def build_correlation_graph(
    X_train: np.ndarray,
    threshold: float = 0.3,
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Build graph from SIGNED Pearson correlation of training data.

    Edges are included whenever |r| >= threshold. Edge weights retain the
    SIGNED value of r so that anti-correlated channels (e.g. legs swinging
    in opposition) produce negative edges — real relational structure that
    abs(r) would destroy.

    Args:
        X_train: (n_samples, seq_len, n_channels)
        threshold: Minimum |Pearson r| for an edge (applied to |r|, not r).

    Returns:
        Same format as build_predefined_graph.
    """
    n_channels = X_train.shape[2]

    flat = X_train.reshape(-1, n_channels)
    corr = np.corrcoef(flat.T).astype(np.float32)  # signed Pearson r, no abs

    # Include edge wherever |r| is strong enough; keep signed weight
    mask = (np.abs(corr) >= threshold).astype(np.float32)
    adj = corr * mask  # signed weights; zeros below threshold

    if self_loops:
        np.fill_diagonal(adj, 1.0)

    # Normalise only the magnitude; sign is preserved for GCN to use
    if normalise:
        adj = _symmetric_normalise_signed(adj)

    adj_tensor = torch.FloatTensor(adj)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)

    n_pos = int((edge_weight > 0).sum().item())
    n_neg = int((edge_weight < 0).sum().item())
    logger.info(
        f"Signed correlation graph: {n_channels} nodes, "
        f"{edge_index.shape[1]} edges (+{n_pos}/-{n_neg}) (|r|>={threshold})"
    )

    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": n_channels,
        "node_features_dim": 1,
        "description": f"Signed-correlation graph (|r|>={threshold})",
    }


# =============================================================================
# DTW-based graph construction
# =============================================================================

def _dtw_distance(s1: np.ndarray, s2: np.ndarray, window: int = None) -> float:
    """DTW distance between two 1D time series.

    Args:
        s1, s2: 1D arrays of equal or different length.
        window: Sakoe-Chiba band width (None = unconstrained standard DTW).

    Returns:
        Scalar DTW distance (square-root of cumulative squared cost).
    """
    n, m = len(s1), len(s2)
    local_cost = (s1[:, None] - s2[None, :]) ** 2  # (n, m), vectorised

    dp = np.full((n, m), np.inf, dtype=np.float64)
    dp[0, 0] = local_cost[0, 0]
    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + local_cost[i, 0]
    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + local_cost[0, j]

    for i in range(1, n):
        j_lo = max(1, i - window) if window is not None else 1
        j_hi = min(m - 1, i + window) if window is not None else m - 1
        for j in range(j_lo, j_hi + 1):
            dp[i, j] = local_cost[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(np.sqrt(dp[n - 1, m - 1]))


def _dtw_distance_matrix(prototypes: np.ndarray, window: int = None) -> np.ndarray:
    """Compute pairwise DTW distance matrix between channel prototypes.

    Args:
        prototypes: (n_channels, seq_len) — one representative time series per channel.
        window: Sakoe-Chiba band width (None = unconstrained).

    Returns:
        (n_channels, n_channels) float32 symmetric distance matrix, diagonal = 0.
    """
    n = prototypes.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance(prototypes[i], prototypes[j], window)
            D[i, j] = D[j, i] = d
    return D


def _dist_to_similarity(D: np.ndarray) -> np.ndarray:
    """Convert distance matrix to similarity via Gaussian kernel.

    sigma = mean of non-zero off-diagonal distances (bandwidth heuristic).
    similarity = exp(-d^2 / (2 * sigma^2))
    """
    n = D.shape[0]
    off_diag = D[np.triu_indices(n, k=1)]
    sigma = off_diag.mean() if off_diag.size > 0 and off_diag.mean() > 0 else 1.0
    return np.exp(-(D ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def build_dtw_graph(
    X_train: np.ndarray,
    threshold: float = None,
    k: int = None,
    window: int = None,
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Build graph from DTW distance between per-channel mean prototypes.

    Edges represent temporal similarity between sensor channels. Two variants:
      - DTW1 (standard): window=None, unconstrained warping path.
      - DTW2 (windowed): window=int, Sakoe-Chiba band constraint.

    Sparsification: provide exactly one of `threshold` or `k`.
      - threshold: keep edges where similarity >= threshold (0–1).
      - k: keep only k nearest neighbours per node (mutual KNN → undirected).

    Args:
        X_train: (n_samples, seq_len, n_channels)
        threshold: Minimum Gaussian similarity for an edge (used if k is None).
        k: Number of nearest neighbours per node (mutual KNN).
        window: Sakoe-Chiba band for DTW2 variant (None → DTW1).
        self_loops: Add self-loops to adjacency.
        normalise: Apply symmetric D^{-1/2} A D^{-1/2} normalisation.

    Returns:
        Same dict format as build_predefined_graph.
    """
    if threshold is None and k is None:
        raise ValueError("Provide either `threshold` or `k` to sparsify the DTW graph.")

    n_channels = X_train.shape[2]

    # Per-channel mean prototype: average over samples → (n_channels, seq_len)
    prototypes = X_train.mean(axis=0).T  # (n_channels, seq_len)

    # Normalise each prototype to zero-mean unit-variance for scale-invariant DTW
    mu = prototypes.mean(axis=1, keepdims=True)
    std = prototypes.std(axis=1, keepdims=True).clip(min=1e-8)
    prototypes = (prototypes - mu) / std

    variant = "windowed-DTW" if window is not None else "DTW"
    logger.info(f"Computing {variant} distance matrix for {n_channels} channels...")
    D = _dtw_distance_matrix(prototypes, window=window)

    # Convert distances to similarities
    S = _dist_to_similarity(D)

    if k is not None:
        # Mutual KNN: edge (i,j) if j in k-NN(i) OR i in k-NN(j)
        adj = np.zeros((n_channels, n_channels), dtype=np.float32)
        for i in range(n_channels):
            row = S[i].copy()
            row[i] = -np.inf  # exclude self
            nn_idx = np.argsort(row)[::-1][:k]
            for j in nn_idx:
                adj[i, j] = S[i, j]
                adj[j, i] = S[j, i]  # mutual → undirected
        strategy_desc = f"{variant}-KNN (k={k})"
    else:
        adj = np.where(S >= threshold, S, 0.0).astype(np.float32)
        strategy_desc = f"{variant}-threshold (t={threshold})"

    if self_loops:
        np.fill_diagonal(adj, 1.0)
    if normalise:
        adj = _symmetric_normalise(adj)

    adj_tensor = torch.FloatTensor(adj)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)

    logger.info(
        f"{strategy_desc}: {n_channels} nodes, {edge_index.shape[1]} edges"
    )
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": n_channels,
        "node_features_dim": 1,
        "dtw_distance_matrix": torch.FloatTensor(D),
        "dtw_similarity_matrix": torch.FloatTensor(S),
        "description": strategy_desc,
    }


# =============================================================================
# KNN graph construction
# =============================================================================

def build_knn_graph(
    X_train: np.ndarray,
    k: int,
    metric: str = "correlation",
    window: int = None,
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Build a mutual K-nearest-neighbour graph between channels.

    Args:
        X_train: (n_samples, seq_len, n_channels)
        k: Number of nearest neighbours per node.
        metric: 'correlation' (Pearson) or 'dtw'.
        window: DTW Sakoe-Chiba band (only used when metric='dtw').
        self_loops, normalise: Standard adjacency options.

    Returns:
        Same dict format as build_predefined_graph.
    """
    n_channels = X_train.shape[2]

    if metric == "correlation":
        flat = X_train.reshape(-1, n_channels)
        corr_signed = np.corrcoef(flat.T).astype(np.float32)
        # Rank neighbours by |r| (strongest relationship in either direction),
        # but keep the signed weight on the selected edges.
        corr_abs = np.abs(corr_signed)
        np.fill_diagonal(corr_abs, 0.0)
        S = corr_abs          # for neighbour selection
        S_signed = corr_signed  # for edge weights
        metric_desc = "Pearson signed-correlation (|r| for KNN selection)"
    elif metric == "dtw":
        prototypes = X_train.mean(axis=0).T
        mu = prototypes.mean(axis=1, keepdims=True)
        std = prototypes.std(axis=1, keepdims=True).clip(min=1e-8)
        prototypes = (prototypes - mu) / std
        D = _dtw_distance_matrix(prototypes, window=window)
        S = _dist_to_similarity(D)
        np.fill_diagonal(S, 0.0)
        metric_desc = f"DTW similarity (window={window})"
    else:
        raise ValueError(f"Unknown KNN metric: {metric}. Use 'correlation' or 'dtw'.")

    # Mutual KNN — select neighbours by S (|r| or similarity), assign
    # signed weight from S_signed (for correlation) or S (for DTW).
    use_signed = metric == "correlation"
    adj = np.zeros((n_channels, n_channels), dtype=np.float32)
    for i in range(n_channels):
        nn_idx = np.argsort(S[i])[::-1][:k]
        for j in nn_idx:
            w_ij = S_signed[i, j] if use_signed else S[i, j]
            w_ji = S_signed[j, i] if use_signed else S[j, i]
            adj[i, j] = w_ij
            adj[j, i] = w_ji

    if self_loops:
        np.fill_diagonal(adj, 1.0)
    if normalise:
        adj = (_symmetric_normalise_signed(adj) if use_signed
               else _symmetric_normalise(adj))

    adj_tensor = torch.FloatTensor(adj)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)

    description = f"KNN (k={k}, metric={metric_desc})"
    logger.info(f"{description}: {n_channels} nodes, {edge_index.shape[1]} edges")
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": n_channels,
        "node_features_dim": 1,
        "description": description,
    }


# =============================================================================
# Unified dispatcher
# =============================================================================

def build_graph_by_strategy(
    strategy: str,
    X_train: np.ndarray = None,
    dataset_name: str = None,
    granularity: str = "default",
    self_loops: bool = True,
    normalise: bool = True,
    # Correlation
    corr_threshold: float = 0.3,
    # DTW
    dtw_threshold: float = None,
    dtw_k: int = None,
    dtw_window: int = None,
    # KNN
    knn_k: int = None,
    knn_metric: str = "correlation",
) -> dict:
    """Unified dispatcher for all predefined graph construction strategies.

    Strategies:
      'domain'          — anatomical / sensor-axis domain knowledge graph
                          (granularity='coarse'|'biomechanical'|'fine' for DSA)
      'correlation'     — SIGNED Pearson r threshold graph
      'dtw'             — DTW1: unconstrained DTW similarity graph
      'dtw_windowed'    — DTW2: Sakoe-Chiba windowed DTW similarity graph
      'knn_correlation' — mutual KNN with signed Pearson correlation metric
      'knn_dtw'         — mutual KNN with DTW similarity metric

    For data-driven strategies (all except 'domain'), X_train is required.
    For 'domain', dataset_name is required.

    DSA-specific: pass granularity='biomechanical' with strategy='domain' to get
    the richer anatomical graph (hub + bilateral + lateral ipsilateral edges).
    """
    if strategy == "domain":
        return build_predefined_graph(dataset_name, granularity, self_loops, normalise)
    elif strategy == "correlation":
        return build_correlation_graph(X_train, corr_threshold, self_loops, normalise)
    elif strategy == "dtw":
        return build_dtw_graph(X_train, threshold=dtw_threshold, k=dtw_k,
                               window=None, self_loops=self_loops, normalise=normalise)
    elif strategy == "dtw_windowed":
        return build_dtw_graph(X_train, threshold=dtw_threshold, k=dtw_k,
                               window=dtw_window, self_loops=self_loops, normalise=normalise)
    elif strategy == "knn_correlation":
        return build_knn_graph(X_train, k=knn_k, metric="correlation",
                               self_loops=self_loops, normalise=normalise)
    elif strategy == "knn_dtw":
        return build_knn_graph(X_train, k=knn_k, metric="dtw", window=dtw_window,
                               self_loops=self_loops, normalise=normalise)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose: domain, correlation, dtw, dtw_windowed, knn_correlation, knn_dtw."
        )


# =============================================================================
# Random-predefined graph (null-model baseline for supervisor feedback #4)
# =============================================================================

def build_random_predefined_graph(
    dataset_name: str,
    seed: int,
    granularity: str = "default",
    self_loops: bool = True,
    normalise: bool = True,
    X_train_channels: int = None,
) -> dict:
    """Bernoulli null-model: same #edges as domain graph but at random pairs.

    Reference (edge count): the anatomical / sensor-axis predefined graph
    for this dataset.  We sample the same number of undirected edges
    uniformly at random over the n(n-1)/2 possible pairs and assign
    weight 1.0.  This tests whether the anatomical STRUCTURE is special,
    or whether any similarly-dense random graph would perform comparably.

    Args:
        dataset_name: 'BasicMotions' | 'Epilepsy' | 'DSA'.
        seed: RNG seed for the draw (deterministic).
        granularity: For DSA, 'coarse' or 'fine'.
        X_train_channels: Optional override of node count.

    Returns:
        Same dict format as build_predefined_graph.
    """
    # Reference domain graph -> extract edge count
    ref = build_predefined_graph(dataset_name, granularity=granularity,
                                  self_loops=False, normalise=False)
    ref_adj = ref["adj_matrix"].numpy() if hasattr(ref["adj_matrix"], "numpy") else ref["adj_matrix"]
    num_nodes = int(ref["num_nodes"])
    ref_upper = np.triu(ref_adj, k=1)
    n_edges = int((ref_upper > 0).sum())

    rng = np.random.RandomState(int(seed))
    all_pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    idx = rng.choice(len(all_pairs), size=min(n_edges, len(all_pairs)), replace=False)
    edges = [all_pairs[k] for k in idx]

    adj_tensor = _edges_to_adj(edges, num_nodes, self_loops=self_loops, normalise=normalise)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)

    node_names = ref.get("node_names", [f"n{i}" for i in range(num_nodes)])
    logger.info(
        f"Random-predef graph ({dataset_name}, seed={seed}): "
        f"{num_nodes} nodes, {n_edges} edges (matched to domain graph)"
    )
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": num_nodes,
        "node_features_dim": ref.get("node_features_dim", 1),
        "node_names": node_names,
        "description": f"Random-predefined (seed={seed}, matched to {dataset_name} domain edge count)",
    }


# =============================================================================
# Correlation-with-topN and no-threshold variants (feedback #3)
# =============================================================================

def build_correlation_graph_topN(
    X_train: np.ndarray,
    fraction: float = 0.5,
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Signed correlation graph keeping top `fraction` of |r| off-diagonal edges.

    Data-adaptive alternative to a fixed |r|>=0.3 cutoff.  Threshold is set to
    the (1-fraction) percentile of |r| over off-diagonal entries, so the
    number of retained edges is n(n-1)/2 * fraction (per direction, symmetric).

    Args:
        X_train: (n_samples, seq_len, n_channels)
        fraction: proportion of off-diagonal edges to keep (e.g. 0.5 = top half).

    Returns dict as build_correlation_graph.
    """
    n_channels = X_train.shape[2]
    flat = X_train.reshape(-1, n_channels)
    corr = np.corrcoef(flat.T).astype(np.float32)

    off_mask = ~np.eye(n_channels, dtype=bool)
    abs_r = np.abs(corr[off_mask])
    if abs_r.size == 0:
        thr = 0.0
    else:
        thr = float(np.quantile(abs_r, 1.0 - float(fraction)))

    mask = (np.abs(corr) >= thr).astype(np.float32)
    adj = corr * mask
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    if normalise:
        adj = _symmetric_normalise_signed(adj)

    adj_tensor = torch.FloatTensor(adj)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)
    logger.info(
        f"Correlation top-{int(fraction*100)}% graph: {n_channels} nodes, "
        f"{edge_index.shape[1]} edges (threshold |r|>={thr:.3f})"
    )
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": n_channels,
        "node_features_dim": 1,
        "description": f"Signed-correlation top-{int(fraction*100)}% (|r|>={thr:.3f})",
    }


def build_correlation_graph_nothresh(
    X_train: np.ndarray,
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Signed correlation graph with ZERO threshold (all off-diag edges kept).

    Tests the pure-ablation: keep every signed r_{ij} as an edge weight.
    """
    return build_correlation_graph(X_train, threshold=0.0,
                                    self_loops=self_loops, normalise=normalise)


def build_correlation_graph_datadriven(
    X_train: np.ndarray,
    stat: str = "mean",
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Signed correlation graph with a DATA-DRIVEN |r| threshold.

    Instead of a hard-coded cut-off (0.3), the threshold is a statistic of the
    off-diagonal |r| distribution itself, so it adapts to each dataset/fold:

      - stat="mean"     -> threshold = mean(|r|_offdiag)
      - stat="meanstd"  -> threshold = mean(|r|_offdiag) + std(|r|_offdiag)

    This answers the feedback that the 0.3 cut-off is arbitrary: the retained
    edges are exactly the pairs whose |r| exceeds the average (or average+std)
    coupling strength observed in the data.

    Args:
        X_train: (n_samples, seq_len, n_channels)
        stat: "mean" or "meanstd".

    Returns dict as build_correlation_graph.
    """
    if stat not in ("mean", "meanstd"):
        raise ValueError(f"stat must be 'mean' or 'meanstd', got {stat!r}")

    n_channels = X_train.shape[2]
    flat = X_train.reshape(-1, n_channels)
    corr = np.corrcoef(flat.T).astype(np.float32)

    off_mask = ~np.eye(n_channels, dtype=bool)
    abs_r = np.abs(corr[off_mask])
    if abs_r.size == 0:
        thr = 0.0
    elif stat == "mean":
        thr = float(abs_r.mean())
    else:  # meanstd
        thr = float(abs_r.mean() + abs_r.std())

    mask = (np.abs(corr) >= thr).astype(np.float32)
    adj = corr * mask
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    if normalise:
        adj = _symmetric_normalise_signed(adj)

    adj_tensor = torch.FloatTensor(adj)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)
    logger.info(
        f"Correlation data-driven ({stat}) graph: {n_channels} nodes, "
        f"{edge_index.shape[1]} edges (threshold |r|>={thr:.3f})"
    )
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": n_channels,
        "node_features_dim": 1,
        "description": f"Signed-correlation data-driven {stat} (|r|>={thr:.3f})",
    }
