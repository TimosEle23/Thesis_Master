"""
Adaptive (learned) graph construction for TGNN models.

The adjacency matrix is learned end-to-end during training via node
embeddings. This is the data-driven counterpart to the predefined
domain-knowledge graphs, and is central to RQ2.

Methods:
  1. Learned adjacency: Two learnable node embedding matrices E1, E2
     A = softmax(ReLU(E1 @ E2^T))
  
  2. Attention-based: Adjacency weights derived from attention over
     node features at each time step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveGraphLearner(nn.Module):
    """Learns a soft adjacency matrix from node embeddings.
    
    Implements the approach from Wu et al. (2019) "Graph WaveNet":
    A = SoftMax(ReLU(E1 @ E2^T))
    
    The learned adjacency can be asymmetric (directed) or forced symmetric.
    Sparsity is enforced via thresholding or top-k.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 16,
        symmetric: bool = True,
        sparsity_threshold: float = 0.1,
        init: str = "random",
    ):
        """
        Args:
            num_nodes: Number of graph nodes.
            embed_dim: Dimension of node embeddings.
            symmetric: If True, force A = (A + A^T) / 2.
            sparsity_threshold: Edges below this weight are zeroed.
            init: Initialization strategy: 'random', 'identity', 'xavier'.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.symmetric = symmetric
        self.sparsity_threshold = sparsity_threshold

        # Two learnable embedding matrices
        self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))

        self._init_embeddings(init)

    def _init_embeddings(self, init: str):
        """Initialize node embeddings."""
        if init == "random":
            nn.init.xavier_uniform_(self.E1)
            nn.init.xavier_uniform_(self.E2)
        elif init == "identity":
            # Start close to identity graph (self-loops dominant)
            nn.init.eye_(self.E1[:min(self.num_nodes, self.embed_dim),
                                  :min(self.num_nodes, self.embed_dim)])
            nn.init.eye_(self.E2[:min(self.num_nodes, self.embed_dim),
                                  :min(self.num_nodes, self.embed_dim)])
        elif init == "xavier":
            nn.init.xavier_normal_(self.E1)
            nn.init.xavier_normal_(self.E2)

    def forward(self) -> tuple:
        """Compute the learned adjacency matrix.
        
        Returns:
            edge_index: LongTensor (2, n_edges) 
            edge_weight: FloatTensor (n_edges,)
            adj_matrix: FloatTensor (num_nodes, num_nodes) - full soft adjacency
        """
        # Compute adjacency: A = softmax(relu(E1 @ E2^T))
        adj = torch.relu(self.E1 @ self.E2.t())
        
        # Row-wise softmax for normalisation
        adj = F.softmax(adj, dim=1)
        
        # Force symmetry if required
        if self.symmetric:
            adj = (adj + adj.t()) / 2.0
        
        # Add self-loops
        adj = adj + torch.eye(self.num_nodes, device=adj.device)
        
        # Re-normalise after adding self-loops
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Sparsify
        adj_sparse = adj.clone()
        adj_sparse[adj_sparse < self.sparsity_threshold] = 0.0
        
        # Convert to edge_index format
        edge_index, edge_weight = self._adj_to_edge_index(adj_sparse)
        
        return edge_index, edge_weight, adj

    def _adj_to_edge_index(self, adj: torch.Tensor) -> tuple:
        """Convert adjacency matrix to COO edge_index and weights."""
        indices = (adj > 0).nonzero(as_tuple=False)
        edge_index = indices.t().contiguous().long()
        edge_weight = adj[indices[:, 0], indices[:, 1]]
        return edge_index, edge_weight

    def get_adjacency(self) -> torch.Tensor:
        """Get the current adjacency matrix (detached, for visualization)."""
        with torch.no_grad():
            _, _, adj = self.forward()
        return adj.detach().cpu()

    def regularization_loss(self, reg_type: str = "sparsity") -> torch.Tensor:
        """Compute regularization loss on the adjacency.
        
        Args:
            reg_type: 'sparsity' (L1) or 'smoothness' (Frobenius norm).
            
        Returns:
            Scalar regularization loss.
        """
        _, _, adj = self.forward()
        
        if reg_type == "sparsity":
            # L1 regularization encourages sparse adjacency
            return adj.abs().mean()
        elif reg_type == "smoothness":
            # Frobenius norm of A - I encourages staying close to identity
            identity = torch.eye(self.num_nodes, device=adj.device)
            return torch.norm(adj - identity, p='fro')
        else:
            return torch.tensor(0.0, device=adj.device)


class AttentionGraphLearner(nn.Module):
    """Learns adjacency from node features via attention at each time step.
    
    Unlike the static AdaptiveGraphLearner, this produces a time-varying
    adjacency where edge weights depend on the current node features.
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 32,
        num_heads: int = 1,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Linear(node_feat_dim, hidden_dim * num_heads)
        self.key = nn.Linear(node_feat_dim, hidden_dim * num_heads)
        self.scale = (hidden_dim) ** 0.5

    def forward(self, node_features: torch.Tensor) -> tuple:
        """Compute attention-based adjacency for given node features.
        
        Args:
            node_features: (batch, num_nodes, feat_dim)
            
        Returns:
            edge_index: (2, n_edges) from thresholded adjacency
            edge_weight: (n_edges,)
            adj_matrix: (batch, num_nodes, num_nodes)
        """
        Q = self.query(node_features)  # (B, N, H*heads)
        K = self.key(node_features)

        # Compute attention scores
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, N, N)
        adj = F.softmax(attn, dim=-1)

        # Average over batch for a consistent graph structure
        adj_mean = adj.mean(dim=0)
        
        # Threshold
        adj_sparse = adj_mean.clone()
        adj_sparse[adj_sparse < 0.05] = 0.0
        
        indices = (adj_sparse > 0).nonzero(as_tuple=False)
        edge_index = indices.t().contiguous().long()
        edge_weight = adj_sparse[indices[:, 0], indices[:, 1]]

        return edge_index, edge_weight, adj
