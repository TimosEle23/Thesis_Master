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


def _adj_to_edge_index(adj: torch.FloatTensor) -> tuple:
    """Convert adjacency matrix to edge_index and edge_weight.
    
    Returns:
        edge_index: LongTensor of shape (2, n_edges)
        edge_weight: FloatTensor of shape (n_edges,)
    """
    # Find non-zero entries
    indices = (adj > 1e-8).nonzero(as_tuple=False)
    edge_index = indices.t().contiguous().long()
    edge_weight = adj[indices[:, 0], indices[:, 1]]
    
    return edge_index, edge_weight


def build_correlation_graph(
    X_train: np.ndarray,
    threshold: float = 0.5,
    self_loops: bool = True,
    normalise: bool = True,
) -> dict:
    """Build graph from Pearson correlation of training data.
    
    Alternative predefined strategy where edges are determined by
    statistical correlation rather than domain knowledge.
    
    Args:
        X_train: (n_samples, seq_len, n_channels)
        threshold: Minimum absolute correlation for an edge.
        
    Returns:
        Same format as build_predefined_graph.
    """
    n_channels = X_train.shape[2]
    
    # Compute correlation across all samples and time steps
    flat = X_train.reshape(-1, n_channels)
    corr = np.corrcoef(flat.T)
    corr = np.abs(corr)  # Use absolute correlation
    
    # Threshold to create binary adjacency
    adj = (corr >= threshold).astype(np.float32)
    
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    
    if normalise:
        adj = _symmetric_normalise(adj)
    
    adj_tensor = torch.FloatTensor(adj)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)
    
    logger.info(
        f"Correlation graph: {n_channels} nodes, "
        f"{edge_index.shape[1]} edges (threshold={threshold})"
    )
    
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "adj_matrix": adj_tensor,
        "num_nodes": n_channels,
        "node_features_dim": 1,
        "description": f"Correlation graph (threshold={threshold})",
    }
