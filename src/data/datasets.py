"""
PyTorch Dataset classes for multivariate time series classification.

Two dataset types:
  - MTSDataset: Standard (N, T, C) format for LSTM, TCN, Transformer baselines
  - GraphMTSDataset: Graph-structured format for TGNN models with edge_index
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class MTSDataset(Dataset):
    """Standard multivariate time series dataset for sequence models.
    
    Returns tensors of shape (seq_len, n_channels) and integer labels.
    For TCN, the training pipeline transposes to (n_channels, seq_len).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: np.ndarray of shape (n_samples, seq_len, n_channels)
            y: np.ndarray of shape (n_samples,) with integer labels
        """
        assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GraphMTSDataset(Dataset):
    """Graph-structured multivariate time series dataset for TGNN models.
    
    Each sample is a temporal graph signal where:
      - Nodes represent sensor variables (or sensor units)
      - Node features at each time step form the signal
      - Edges represent predefined or learned relationships
      
    There are two representation modes:
      1. Channel-as-node: Each channel is a node with scalar features over time
         X shape: (n_samples, seq_len, n_nodes) where n_nodes = n_channels
         
      2. Unit-as-node: Each sensor unit is a node with multi-dim features
         X shape: (n_samples, seq_len, n_nodes, node_feat_dim)
         e.g., for DSA coarse: (9120, 125, 5, 9)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray = None,
        node_mode: str = "channel",
        sensor_groups: dict = None,
    ):
        """
        Args:
            X: Input data, shape depends on node_mode.
            y: Labels, shape (n_samples,).
            edge_index: Edge index array of shape (2, n_edges).
            edge_weight: Optional edge weights of shape (n_edges,).
            node_mode: 'channel' (each channel = node) or 'unit' (sensor unit = node).
            sensor_groups: Dict mapping unit names to channel indices (for unit mode).
        """
        self.y = torch.LongTensor(y)
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_weight = torch.FloatTensor(edge_weight) if edge_weight is not None else None
        self.node_mode = node_mode
        
        if node_mode == "unit" and sensor_groups is not None:
            # Group channels into sensor units
            # X: (N, T, C) → (N, T, n_units, features_per_unit)
            self.X = self._group_channels(X, sensor_groups)
        else:
            # Channel-as-node mode
            # X: (N, T, C) → each channel is a node with 1 feature
            self.X = torch.FloatTensor(X)
    
    def _group_channels(self, X: np.ndarray, sensor_groups: dict) -> torch.Tensor:
        """Group channels by sensor unit.
        
        Args:
            X: (n_samples, seq_len, n_channels)
            sensor_groups: Dict mapping unit name to list of channel indices.
            
        Returns:
            Tensor of shape (n_samples, seq_len, n_units, features_per_unit)
        """
        unit_data = []
        for unit_name, indices in sensor_groups.items():
            unit_data.append(X[:, :, indices])
        
        # Stack: (n_samples, seq_len, n_units, features_per_unit)
        grouped = np.stack(unit_data, axis=2)
        return torch.FloatTensor(grouped)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {
            "x": self.X[idx],              # (T, N_nodes) or (T, N_nodes, F)
            "y": self.y[idx],
            "edge_index": self.edge_index,  # (2, E) - shared across samples
        }
        if self.edge_weight is not None:
            sample["edge_weight"] = self.edge_weight
        return sample


def graph_collate_fn(batch):
    """Custom collate function for GraphMTSDataset.
    
    Stacks graph signals into a batch while keeping edge_index shared.
    
    Returns:
        Dict with:
            x: (batch_size, seq_len, n_nodes, [feat_dim])
            y: (batch_size,)
            edge_index: (2, n_edges) - shared
            edge_weight: (n_edges,) - shared, optional
    """
    x = torch.stack([item["x"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    edge_index = batch[0]["edge_index"]  # Same for all samples
    
    result = {"x": x, "y": y, "edge_index": edge_index}
    
    if "edge_weight" in batch[0]:
        result["edge_weight"] = batch[0]["edge_weight"]
    
    return result
