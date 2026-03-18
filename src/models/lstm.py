"""
LSTM baseline classifier for multivariate time series.

A standard LSTM that processes the full multivariate input as a vector at each
time step. This baseline treats variables as an unstructured vector, relying on
the LSTM's internal capacity to implicitly learn inter-variable dependencies.

Architecture:
  Input (batch, seq_len, n_channels) → LSTM layers → last hidden state → FC → classes
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM-based multivariate time series classifier.
    
    Serves as a baseline for RQ1 (comparison with TGNNs) and
    RQ3 (generalization across dataset complexity).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Args:
            n_channels: Number of input variables/channels.
            n_classes: Number of output classes.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        self.model_name = "LSTM"
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels).
            
        Returns:
            Logits of shape (batch, n_classes).
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        
        # Use last time step output
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            last_out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_out = h_n[-1]  # (batch, hidden_dim)
        
        last_out = self.layer_norm(last_out)
        last_out = self.dropout(last_out)
        
        logits = self.classifier(last_out)
        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
