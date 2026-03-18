"""
Temporal Graph Neural Network (TGNN) architectures for MTS classification.

This module implements several TGNN variants that combine spatial graph
convolution with temporal sequence modelling. These are the primary models
under investigation for all three research questions.

Architectures:
  1. GConvLSTM: Chebyshev graph convolution + LSTM cells
  2. GConvGRU: Chebyshev graph convolution + GRU cells
  3. A3TGCN: Attention-based Temporal GCN
  4. DCRNN: Diffusion Convolutional Recurrent Neural Network

All architectures follow the pattern:
  For each time step t:
    1. Spatial: Graph convolution over nodes using adjacency
    2. Temporal: Recurrent update of hidden states

Graph construction (predefined vs adaptive) is handled externally and
passed to these models via edge_index/edge_weight, enabling fair comparison
across graph strategies (RQ2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from src.graphs.adaptive import AdaptiveGraphLearner


# =============================================================================
# Chebyshev Graph Convolution
# =============================================================================

class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution (Defferrard et al., 2016).
    
    Approximates spectral graph convolution using K-order Chebyshev polynomials.
    Avoids explicit eigendecomposition of the Laplacian.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 2):
        """
        Args:
            in_features: Input feature dimension per node.
            out_features: Output feature dimension per node.
            K: Order of Chebyshev polynomial (receptive field size).
        """
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        
        # One weight matrix per Chebyshev order
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, num_nodes, in_features).
            adj: Normalised adjacency matrix (num_nodes, num_nodes).
            
        Returns:
            Output features (batch, num_nodes, out_features).
        """
        # Compute scaled Laplacian for Chebyshev basis
        # L_tilde = 2 * L / lambda_max - I  (approximated as 2*A - I for normalised A)
        
        # Chebyshev recurrence: T_0(x) = I, T_1(x) = x, T_k(x) = 2x*T_{k-1} - T_{k-2}
        T_0 = x  # (B, N, F_in)
        out = T_0 @ self.weights[0]  # (B, N, F_out)

        if self.K > 1:
            T_1 = torch.matmul(adj, x)  # (B, N, F_in) via broadcast
            out = out + T_1 @ self.weights[1]

            for k in range(2, self.K):
                T_2 = 2 * torch.matmul(adj, T_1) - T_0
                out = out + T_2 @ self.weights[k]
                T_0, T_1 = T_1, T_2

        return out + self.bias


# =============================================================================
# Diffusion Convolution
# =============================================================================

class DiffusionConv(nn.Module):
    """Diffusion convolution (Li et al., 2018 - DCRNN).
    
    Models diffusion process on the graph using forward and backward
    random walks of K steps.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 2):
        super().__init__()
        self.K = K
        # Forward and backward diffusion
        self.weight_forward = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(K)
        ])
        self.weight_backward = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in list(self.weight_forward) + list(self.weight_backward):
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes) normalised adjacency
        """
        adj_t = adj.t()  # Backward transition matrix
        
        out = torch.zeros_like(x[:, :, :self.weight_forward[0].shape[1]])
        
        # Forward diffusion
        Pk = x
        for k in range(self.K):
            out = out + Pk @ self.weight_forward[k]
            Pk = torch.matmul(adj, Pk)
        
        # Backward diffusion
        Pk = x
        for k in range(self.K):
            out = out + Pk @ self.weight_backward[k]
            Pk = torch.matmul(adj_t, Pk)
        
        return out + self.bias


# =============================================================================
# Graph Convolutional LSTM Cell
# =============================================================================

class GConvLSTMCell(nn.Module):
    """LSTM cell with Chebyshev graph convolution replacing linear transforms.
    
    At each time step, the input-to-hidden and hidden-to-hidden transforms
    use graph convolution instead of standard matrix multiplication,
    enabling spatial message passing within the recurrent update.
    """

    def __init__(self, in_features: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Graph convolutions for the 4 LSTM gates: input, forget, cell, output
        # Input: concat(x_t, h_{t-1}) → 4 * hidden_dim
        self.graph_conv = ChebConv(
            in_features + hidden_dim, 4 * hidden_dim, K
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        h: torch.Tensor = None,
        c: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input at time t, shape (batch, num_nodes, in_features).
            adj: Adjacency matrix (num_nodes, num_nodes).
            h: Previous hidden state (batch, num_nodes, hidden_dim).
            c: Previous cell state (batch, num_nodes, hidden_dim).
            
        Returns:
            h_new: Updated hidden state.
            c_new: Updated cell state.
        """
        batch_size, num_nodes, _ = x.shape
        
        if h is None:
            h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        if c is None:
            c = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h], dim=2)  # (B, N, F_in + H)
        
        # Graph convolution over concatenated input
        gates = self.graph_conv(combined, adj)  # (B, N, 4H)
        
        # Split into 4 gates
        i, f, g, o = gates.chunk(4, dim=2)
        
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


# =============================================================================
# Graph Convolutional GRU Cell
# =============================================================================

class GConvGRUCell(nn.Module):
    """GRU cell with Chebyshev graph convolution."""

    def __init__(self, in_features: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Graph conv for reset and update gates
        self.conv_gates = ChebConv(
            in_features + hidden_dim, 2 * hidden_dim, K
        )
        # Graph conv for candidate hidden state
        self.conv_candidate = ChebConv(
            in_features + hidden_dim, hidden_dim, K
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        h: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes)
            h: (batch, num_nodes, hidden_dim)
            
        Returns:
            h_new: (batch, num_nodes, hidden_dim)
        """
        batch_size, num_nodes, _ = x.shape
        
        if h is None:
            h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        
        combined = torch.cat([x, h], dim=2)
        gates = self.conv_gates(combined, adj)
        r, z = gates.chunk(2, dim=2)
        r = torch.sigmoid(r)  # Reset gate
        z = torch.sigmoid(z)  # Update gate
        
        combined_r = torch.cat([x, r * h], dim=2)
        h_candidate = torch.tanh(self.conv_candidate(combined_r, adj))
        
        h_new = z * h + (1 - z) * h_candidate
        return h_new


# =============================================================================
# DCRNN Cell (Diffusion Convolutional Recurrent)
# =============================================================================

class DCRNNCell(nn.Module):
    """DCRNN cell using diffusion convolution within GRU structure."""

    def __init__(self, in_features: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.conv_gates = DiffusionConv(
            in_features + hidden_dim, 2 * hidden_dim, K
        )
        self.conv_candidate = DiffusionConv(
            in_features + hidden_dim, hidden_dim, K
        )

    def forward(self, x, adj, h=None):
        batch_size, num_nodes, _ = x.shape
        if h is None:
            h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        
        combined = torch.cat([x, h], dim=2)
        gates = self.conv_gates(combined, adj)
        r, z = gates.chunk(2, dim=2)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        
        combined_r = torch.cat([x, r * h], dim=2)
        h_candidate = torch.tanh(self.conv_candidate(combined_r, adj))
        
        h_new = z * h + (1 - z) * h_candidate
        return h_new


# =============================================================================
# A3TGCN: Attention Temporal Graph Convolutional Network
# =============================================================================

class A3TGCNCell(nn.Module):
    """A3TGCN cell: GConvGRU with temporal attention mechanism.
    
    Based on Bai et al. (2021). Adds an attention mechanism over the
    temporal dimension to weight the importance of different time steps.
    """

    def __init__(self, in_features: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gconv_gru = GConvGRUCell(in_features, hidden_dim, K)
        
        # Temporal attention
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj, h=None):
        """Same as GConvGRU forward."""
        return self.gconv_gru(x, adj, h)


# =============================================================================
# Full TGNN Classifier
# =============================================================================

class TGNNClassifier(nn.Module):
    """Temporal Graph Neural Network classifier for MTS.
    
    Combines spatial graph convolution with temporal recurrent processing.
    Supports multiple TGNN cell types and both predefined/adaptive graphs.
    
    Architecture:
      For each time step t:
        1. Graph conv over nodes at time t
        2. Recurrent state update
      Final: Temporal pooling → Node pooling → Classification
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
        cell_type: str = "gconv_lstm",
        graph_mode: str = "predefined",
        adaptive_embed_dim: int = 16,
        adaptive_sparsity: float = 0.1,
        edge_index: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
        adj_matrix: torch.Tensor = None,
    ):
        """
        Args:
            n_channels: Total number of input channels.
            n_classes: Number of output classes.
            num_nodes: Number of graph nodes.
            node_features_dim: Feature dimension per node (1 for channel-as-node,
                              9 for DSA coarse body-segment nodes).
            hidden_dim: Hidden dimension of recurrent cells.
            K: Chebyshev polynomial order.
            num_layers: Number of stacked TGNN layers.
            dropout: Dropout rate.
            cell_type: 'gconv_lstm', 'gconv_gru', 'a3tgcn', or 'dcrnn'.
            graph_mode: 'predefined' or 'adaptive'.
            adaptive_embed_dim: Node embedding dim for adaptive graph.
            adaptive_sparsity: Sparsity threshold for adaptive graph.
            edge_index: Predefined edge index (2, E).
            edge_weight: Predefined edge weights (E,).
            adj_matrix: Predefined adjacency matrix (N, N).
        """
        super().__init__()
        self.model_name = f"TGNN_{cell_type}"
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.graph_mode = graph_mode
        self.n_channels = n_channels

        # Store predefined graph
        if adj_matrix is not None:
            self.register_buffer("adj_matrix", adj_matrix)
        elif edge_index is not None:
            # Reconstruct adj_matrix from edge_index
            adj = torch.zeros(num_nodes, num_nodes)
            if edge_weight is not None:
                adj[edge_index[0], edge_index[1]] = edge_weight
            else:
                adj[edge_index[0], edge_index[1]] = 1.0
            self.register_buffer("adj_matrix", adj)
        else:
            self.register_buffer("adj_matrix", torch.eye(num_nodes))

        # Adaptive graph learner (for RQ2 comparison)
        self.adaptive_graph = None
        if graph_mode == "adaptive":
            self.adaptive_graph = AdaptiveGraphLearner(
                num_nodes=num_nodes,
                embed_dim=adaptive_embed_dim,
                symmetric=True,
                sparsity_threshold=adaptive_sparsity,
            )

        # Input projection: project node features to hidden_dim
        self.input_proj = nn.Linear(node_features_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # TGNN layers
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for layer in range(num_layers):
            in_dim = hidden_dim  # After projection
            cell = self._build_cell(cell_type, in_dim, hidden_dim, K)
            self.cells.append(cell)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Temporal attention for weighting time steps
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

    def _build_cell(self, cell_type, in_features, hidden_dim, K):
        """Factory for TGNN cells."""
        if cell_type == "gconv_lstm":
            return GConvLSTMCell(in_features, hidden_dim, K)
        elif cell_type == "gconv_gru":
            return GConvGRUCell(in_features, hidden_dim, K)
        elif cell_type == "a3tgcn":
            return A3TGCNCell(in_features, hidden_dim, K)
        elif cell_type == "dcrnn":
            return DCRNNCell(in_features, hidden_dim, K)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def _get_adjacency(self) -> torch.Tensor:
        """Get the adjacency matrix (predefined or learned)."""
        if self.graph_mode == "adaptive" and self.adaptive_graph is not None:
            _, _, adj = self.adaptive_graph()
            return adj
        return self.adj_matrix

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input from (B, T, C) to (B, T, N, F).
        
        For channel-as-node (F=1): (B, T, C) → (B, T, C, 1)
        For unit-as-node (e.g., F=9): (B, T, C) → (B, T, N, F)
        """
        B, T, C = x.shape
        
        if self.node_features_dim == 1:
            # Channel-as-node: each channel is a node with 1 feature
            return x.unsqueeze(-1)  # (B, T, N, 1) where N=C
        else:
            # Unit-as-node: reshape features per node
            # C should equal num_nodes * node_features_dim
            return x.view(B, T, self.num_nodes, self.node_features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels).
            
        Returns:
            Logits of shape (batch, n_classes).
        """
        B, T, C = x.shape
        
        # Reshape to node format: (B, T, N, F)
        x_nodes = self._reshape_input(x)
        
        # Get adjacency
        adj = self._get_adjacency()
        
        # Project node features: (B, T, N, F) → (B, T, N, H)
        x_proj = self.input_proj(x_nodes)
        x_proj = self.input_norm(x_proj)
        
        # Process through TGNN layers
        # For each layer, iterate over time steps
        hidden_states_per_layer = [None] * self.num_layers
        all_temporal_outputs = []  # Collect outputs at each time step
        
        for t in range(T):
            layer_input = x_proj[:, t]  # (B, N, H)
            
            for layer_idx, cell in enumerate(self.cells):
                h_prev = hidden_states_per_layer[layer_idx]
                
                if self.cell_type == "gconv_lstm":
                    if h_prev is None:
                        h, c = cell(layer_input, adj)
                    else:
                        h, c = cell(layer_input, adj, h_prev[0], h_prev[1])
                    hidden_states_per_layer[layer_idx] = (h, c)
                    layer_input = self.dropout(self.norms[layer_idx](h))
                else:
                    h = cell(layer_input, adj, h_prev)
                    hidden_states_per_layer[layer_idx] = h
                    layer_input = self.dropout(self.norms[layer_idx](h))
            
            # Output of last layer at time t: (B, N, H)
            all_temporal_outputs.append(layer_input)
        
        # Stack temporal outputs: (B, T, N, H)
        temporal_stack = torch.stack(all_temporal_outputs, dim=1)
        
        # Node pooling: mean over nodes → (B, T, H)
        node_pooled = temporal_stack.mean(dim=2)
        
        # Temporal attention pooling: weighted sum over time → (B, H)
        attn_weights = self.temporal_attention(node_pooled)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        temporal_pooled = (attn_weights * node_pooled).sum(dim=1)  # (B, H)
        
        logits = self.classifier(temporal_pooled)
        return logits

    def get_adaptive_reg_loss(self) -> torch.Tensor:
        """Get regularization loss for adaptive graph (used during training)."""
        if self.adaptive_graph is not None:
            return self.adaptive_graph.regularization_loss("sparsity")
        return torch.tensor(0.0)

    def get_learned_adjacency(self) -> Optional[torch.Tensor]:
        """Get the learned adjacency matrix for visualization."""
        if self.adaptive_graph is not None:
            return self.adaptive_graph.get_adjacency()
        return None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
