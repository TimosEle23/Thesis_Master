"""
Hybrid TGNN classifier: predefined adjacency + learnable residual graph.

This model combines the strengths of predefined domain-knowledge graphs (RQ2)
with a learnable graph correction. The effective adjacency is:

    A_eff = alpha * A_predefined + (1 - alpha) * A_learned

where alpha is a learnable scalar controlling the blend. A GRU-based temporal
core processes node features at each time step using graph convolutions over
A_eff. This directly addresses RQ2 by testing whether combining predefined
and adaptive graph strategies outperforms either alone.

Architecture:
  Input (B, T, N, F) → project to H → for each t: GraphConv(GRU) → temporal
  attention pooling → node mean pooling → classifier head → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.tgnn import ChebConv, GConvGRUCell
from src.graphs.adaptive import AdaptiveGraphLearner


class HybridTGNNClassifier(nn.Module):
    """Hybrid TGNN with blended predefined + learned adjacency.

    Supports both 3D (B, T, C) and 4D (B, T, N, F) inputs.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        num_nodes: int,
        node_features_dim: int,
        hidden_dim: int = 64,
        K: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        adaptive_embed_dim: int = 16,
        adaptive_sparsity: float = 0.1,
        adaptive_mode: str = "factored",
        edge_index: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
        adj_matrix: torch.Tensor = None,
    ):
        super().__init__()
        self.model_name = "HybridTGNN"
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_channels = n_channels

        # Predefined adjacency
        if adj_matrix is not None:
            self.register_buffer("adj_predefined", adj_matrix)
        elif edge_index is not None:
            adj = torch.zeros(num_nodes, num_nodes)
            if edge_weight is not None:
                adj[edge_index[0], edge_index[1]] = edge_weight
            else:
                adj[edge_index[0], edge_index[1]] = 1.0
            self.register_buffer("adj_predefined", adj)
        else:
            self.register_buffer("adj_predefined", torch.eye(num_nodes))

        # Learnable graph residual
        self.adaptive_graph = AdaptiveGraphLearner(
            num_nodes=num_nodes,
            embed_dim=adaptive_embed_dim,
            symmetric=True,
            sparsity_threshold=adaptive_sparsity,
            mode=adaptive_mode,
        )

        # Learnable blend factor (initialised to favour predefined)
        self.alpha_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ≈ 0.88

        # Input projection
        self.input_proj = nn.Linear(node_features_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # GRU-based TGNN layers with graph conv
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(GConvGRUCell(hidden_dim, hidden_dim, K))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def _get_adjacency(self) -> torch.Tensor:
        """Blend predefined and learned adjacency matrices."""
        alpha = torch.sigmoid(self.alpha_logit)
        _, _, adj_learned = self.adaptive_graph()
        return alpha * self.adj_predefined + (1 - alpha) * adj_learned

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input to (B, T, N, F). Handles 3D and 4D inputs."""
        if x.ndim == 4:
            return x
        B, T, C = x.shape
        if self.node_features_dim == 1:
            return x.unsqueeze(-1)
        else:
            return x.view(B, T, self.num_nodes, self.node_features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_channels) or (batch, seq_len, n_nodes, feat).

        Returns:
            Logits of shape (batch, n_classes).
        """
        B, T = x.shape[0], x.shape[1]
        x_nodes = self._reshape_input(x)  # (B, T, N, F)

        adj = self._get_adjacency()

        # Project: (B, T, N, F) → (B, T, N, H)
        x_proj = self.input_proj(x_nodes)
        x_proj = self.input_norm(x_proj)

        hidden_states = [None] * self.num_layers
        all_temporal_outputs = []

        for t in range(T):
            layer_input = x_proj[:, t]  # (B, N, H)
            for i, cell in enumerate(self.cells):
                h = cell(layer_input, adj, hidden_states[i])
                hidden_states[i] = h
                layer_input = self.dropout(self.norms[i](h))
            all_temporal_outputs.append(layer_input)

        temporal_stack = torch.stack(all_temporal_outputs, dim=1)  # (B, T, N, H)
        node_pooled = temporal_stack.mean(dim=2)  # (B, T, H)

        attn_weights = self.temporal_attention(node_pooled)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        temporal_pooled = (attn_weights * node_pooled).sum(dim=1)  # (B, H)

        return self.classifier(temporal_pooled)

    def get_blend_alpha(self) -> float:
        """Return the current predefined/learned blend ratio."""
        return torch.sigmoid(self.alpha_logit).item()

    def get_adaptive_reg_loss(self) -> torch.Tensor:
        """Sparsity regularisation on the learned graph component."""
        return self.adaptive_graph.regularization_loss("sparsity")

    def get_learned_adjacency(self) -> Optional[torch.Tensor]:
        """Return the learned adjacency matrix."""
        return self.adaptive_graph.get_adjacency()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
