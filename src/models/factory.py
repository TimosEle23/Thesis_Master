"""
Model factory: builds any model from a configuration dictionary.

Centralises model construction so the experiment runner only needs
to specify model name and dataset metadata.
"""

import torch
import logging
from typing import Optional

from src.models.lstm import LSTMClassifier
from src.models.tcn import TCNClassifier
from src.models.transformer import TransformerClassifier
from src.models.tgnn import TGNNClassifier

logger = logging.getLogger("thesis")

# All registered model names
BASELINE_MODELS = ["lstm", "tcn", "transformer"]
TGNN_MODELS = ["gconv_lstm", "gconv_gru", "a3tgcn", "dcrnn"]
ALL_MODELS = BASELINE_MODELS + TGNN_MODELS


def build_model(
    model_name: str,
    n_channels: int,
    n_classes: int,
    seq_len: int,
    model_config: dict,
    graph_config: Optional[dict] = None,
    graph_mode: str = "predefined",
) -> torch.nn.Module:
    """Build a model from configuration.
    
    Args:
        model_name: One of ALL_MODELS.
        n_channels: Number of input channels/variables.
        n_classes: Number of output classes.
        seq_len: Sequence length.
        model_config: Model-specific hyperparameters.
        graph_config: Graph construction output (for TGNNs).
            Must contain: num_nodes, node_features_dim, edge_index, edge_weight, adj_matrix
        graph_mode: 'predefined' or 'adaptive' (for TGNNs).
        
    Returns:
        Instantiated model.
    """
    logger.info(f"Building model: {model_name} (channels={n_channels}, classes={n_classes})")

    if model_name == "lstm":
        model = LSTMClassifier(
            n_channels=n_channels,
            n_classes=n_classes,
            hidden_dim=model_config.get("hidden_dim", 128),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.3),
            bidirectional=model_config.get("bidirectional", False),
        )

    elif model_name == "tcn":
        model = TCNClassifier(
            n_channels=n_channels,
            n_classes=n_classes,
            num_channels=model_config.get("num_channels", [64, 128, 128]),
            kernel_size=model_config.get("kernel_size", 3),
            dropout=model_config.get("dropout", 0.2),
        )

    elif model_name == "transformer":
        model = TransformerClassifier(
            n_channels=n_channels,
            n_classes=n_classes,
            d_model=model_config.get("d_model", 128),
            nhead=model_config.get("nhead", 8),
            num_layers=model_config.get("num_layers", 3),
            dim_feedforward=model_config.get("dim_feedforward", 256),
            dropout=model_config.get("dropout", 0.1),
            positional_encoding=model_config.get("positional_encoding", "sinusoidal"),
            max_seq_len=seq_len + 10,
        )

    elif model_name in TGNN_MODELS:
        if graph_config is None:
            raise ValueError(f"TGNN model '{model_name}' requires graph_config")
        
        model = TGNNClassifier(
            n_channels=n_channels,
            n_classes=n_classes,
            num_nodes=graph_config["num_nodes"],
            node_features_dim=graph_config["node_features_dim"],
            hidden_dim=model_config.get("hidden_dim", 64),
            K=model_config.get("K", 2),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.3),
            cell_type=model_name,
            graph_mode=graph_mode,
            adaptive_embed_dim=model_config.get("adaptive_embed_dim", 16),
            adaptive_sparsity=model_config.get("adaptive_sparsity", 0.1),
            edge_index=graph_config.get("edge_index"),
            edge_weight=graph_config.get("edge_weight"),
            adj_matrix=graph_config.get("adj_matrix"),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {ALL_MODELS}")

    n_params = model.count_parameters()
    logger.info(f"  Model built: {n_params:,} trainable parameters")

    return model
