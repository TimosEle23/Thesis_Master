"""
CNN-BiGRU hybrid classifier for multivariate time series.

Architecture:
  Input (batch, seq_len, n_channels) → Conv1d feature extraction → BiGRU
  temporal modelling → last hidden state → FC → classes

The CNN front-end extracts local patterns from each time step, while the
bidirectional GRU captures long-range temporal dependencies in both directions.
This is a strong hybrid baseline frequently used in wearable HAR literature
(Ordonez & Roggen, 2016; Zhao et al., 2018).
"""

import torch
import torch.nn as nn


class CNNBiGRUClassifier(nn.Module):
    """CNN + Bidirectional GRU classifier for MTS.

    Serves as a hybrid baseline for RQ1 and RQ3.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        cnn_channels: list = None,
        kernel_size: int = 3,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_channels: Number of input variables/channels.
            n_classes: Number of output classes.
            cnn_channels: List of output channels for Conv1d layers.
            kernel_size: Convolution kernel size.
            gru_hidden_dim: GRU hidden dimension.
            gru_num_layers: Number of stacked GRU layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.model_name = "CNN_BiGRU"

        if cnn_channels is None:
            cnn_channels = [64, 128]

        # CNN feature extractor (operates on channel dimension)
        cnn_layers = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(gru_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, gru_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_channels).

        Returns:
            Logits of shape (batch, n_classes).
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # (batch, cnn_out, seq_len)

        # Back to (batch, seq_len, features) for GRU
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)  # h_n: (2*num_layers, batch, hidden)

        # Concatenate last hidden states from both directions
        last_out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        last_out = self.layer_norm(last_out)
        last_out = self.dropout(last_out)

        logits = self.classifier(last_out)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
