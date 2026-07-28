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
    Sparsity is enforced via thresholding.

    When `init_matrix` is supplied, E1 and E2 are initialised via truncated
    SVD so that E1 @ E2.T ≈ init_matrix.  This seeds the adaptive graph from
    a predefined one (binary, correlation, DTW, kNN) and tracks drift during
    training via `adjacency_drift()`.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 16,
        symmetric: bool = True,
        sparsity_threshold: float = 0.1,
        init: str = "random",
        init_matrix: torch.Tensor = None,
        mode: str = "factored",
    ):
        """
        Args:
            num_nodes: Number of graph nodes.
            embed_dim: Dimension of node embeddings (ignored when mode='elementwise').
            symmetric: If True, force A = (A + A^T) / 2.
            sparsity_threshold: Edges below this weight are zeroed.
            init: Initialization strategy: 'random', 'identity', 'xavier',
                  or 'from_matrix' (requires init_matrix).
            init_matrix: (num_nodes, num_nodes) tensor to seed embeddings from.
                         Used when init='from_matrix'.
            mode: 'factored' — A = softmax(ReLU(E1@E2^T)) via low-rank embeddings
                              (Graph WaveNet; supports SVD seeding via init_matrix).
                  'elementwise' — A is a direct N×N nn.Parameter initialised to
                              init_matrix (via log(A_norm)). No factorisation, so
                              training can move any entry independently. Used to
                              test whether the SVD low-rank bottleneck limits how
                              far the learned graph can drift from the predefined
                              seed (feedback #10).
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.symmetric = symmetric
        self.sparsity_threshold = sparsity_threshold
        self.mode = mode

        if mode not in ("factored", "elementwise"):
            raise ValueError(f"Unknown AdaptiveGraphLearner mode: {mode!r}")

        if mode == "factored":
            self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
            self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))
            if init_matrix is not None:
                self._init_from_matrix(init_matrix)
            else:
                self._init_embeddings(init)
        else:
            self.A_raw = nn.Parameter(torch.empty(num_nodes, num_nodes))
            self._init_elementwise(init_matrix, init)

        # Capture the processed initial adjacency (after ReLU/softmax/normalize)
        # so drift measures how much training moves the graph, not how much
        # the SVD approximation differs from the raw predefined matrix.
        with torch.no_grad():
            _, _, A0 = self.forward()
        self.register_buffer("_A_init", A0.detach().clone())

    def _init_elementwise(self, init_matrix: torch.Tensor, init: str):
        """Initialise raw N×N logits so softmax(ReLU(A_raw)) ~ init_matrix."""
        if init_matrix is not None:
            A = init_matrix.detach().cpu().float()
            row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
            A_norm = A / row_sum
            A_log = torch.log(A_norm.clamp(min=1e-8))
            with torch.no_grad():
                self.A_raw.copy_(A_log)
        elif init == "identity":
            with torch.no_grad():
                self.A_raw.copy_(torch.eye(self.num_nodes))
        else:
            nn.init.xavier_uniform_(self.A_raw)

    def _init_embeddings(self, init: str):
        """Initialize node embeddings (random / identity / xavier)."""
        if init == "random":
            nn.init.xavier_uniform_(self.E1)
            nn.init.xavier_uniform_(self.E2)
        elif init == "identity":
            nn.init.eye_(self.E1[:min(self.num_nodes, self.embed_dim),
                                  :min(self.num_nodes, self.embed_dim)])
            nn.init.eye_(self.E2[:min(self.num_nodes, self.embed_dim),
                                  :min(self.num_nodes, self.embed_dim)])
        elif init == "xavier":
            nn.init.xavier_normal_(self.E1)
            nn.init.xavier_normal_(self.E2)

    def _init_from_matrix(self, A: torch.Tensor):
        """Seed E1, E2 via truncated SVD so that E1 @ E2.T ≈ A.

        Uses the top-`embed_dim` singular components of A.  After this init,
        the forward pass reproduces A closely before any gradient updates,
        letting us measure how far training drifts from the predefined graph.
        """
        A_np = A.detach().cpu().float().numpy()
        k = min(self.embed_dim, self.num_nodes)
        try:
            from numpy.linalg import svd
            U, s, Vt = svd(A_np, full_matrices=False)
            # Take top-k components
            U_k = U[:, :k]        # (N, k)
            s_k = s[:k]           # (k,)
            Vt_k = Vt[:k, :]     # (k, N)

            # Absorb sqrt(s) into both factors: E1 @ E2.T = (U*sqrt(s)) @ (V*sqrt(s)).T
            sqrt_s = np.sqrt(np.maximum(s_k, 0.0))
            E1_init = U_k * sqrt_s[None, :]    # (N, k)
            E2_init = Vt_k.T * sqrt_s[None, :] # (N, k)

            # Pad to embed_dim if k < embed_dim
            if k < self.embed_dim:
                pad = self.embed_dim - k
                E1_init = np.concatenate([E1_init, np.zeros((self.num_nodes, pad), dtype=np.float32)], axis=1)
                E2_init = np.concatenate([E2_init, np.zeros((self.num_nodes, pad), dtype=np.float32)], axis=1)

            with torch.no_grad():
                self.E1.copy_(torch.FloatTensor(E1_init))
                self.E2.copy_(torch.FloatTensor(E2_init))
        except Exception as e:
            # Fall back to random init if SVD fails
            nn.init.xavier_uniform_(self.E1)
            nn.init.xavier_uniform_(self.E2)

    def forward(self) -> tuple:
        """Compute the learned adjacency matrix.

        Returns:
            edge_index: LongTensor (2, n_edges)
            edge_weight: FloatTensor (n_edges,)
            adj_matrix: FloatTensor (num_nodes, num_nodes) - full soft adjacency
        """
        if self.mode == "factored":
            adj = torch.relu(self.E1 @ self.E2.t())
        else:
            adj = torch.relu(self.A_raw)
        
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

    def adjacency_drift(self) -> dict:
        """Frobenius-norm drift of learned A from its initial state.

        Compares A_current (after training) to A_init (snapshot taken
        immediately after parameter initialisation in __init__).

        Returns:
          frob_norm: ||A_current - A_init||_F
          relative_frob: frob_norm / ||A_init||_F
          A_current: (N, N) tensor (detached, CPU)
        """
        A_current = self.get_adjacency()
        A_init = self._A_init.cpu()
        diff = A_current - A_init
        frob = float(torch.norm(diff, p="fro").item())
        denom = float(torch.norm(A_init, p="fro").item())
        rel = frob / denom if denom > 1e-8 else 0.0
        return {
            "frob_norm": frob,
            "relative_frob": rel,
            "A_current": A_current,
        }

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
