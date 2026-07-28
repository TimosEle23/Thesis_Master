"""
Temporal Convolutional Network (TCN) baseline classifier.

Based on Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling".

Architecture:
  Input (batch, n_channels, seq_len) → Residual TCN blocks → Global Avg Pool → FC → classes

Each TCN block uses dilated causal convolutions with exponentially increasing
dilation factors, enabling a large receptive field with fewer parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import List


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-side padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove right-side padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block with two causal dilated convolutions.
    
    Implements weight normalisation and spatial dropout for regularisation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()

        self.padding1 = (kernel_size - 1) * dilation
        # weight_norm stabilises training and matches Bai et al. (2018)
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding1, dilation=dilation
        ))

        self.padding2 = (kernel_size - 1) * dilation
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding2, dilation=dilation
        ))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 convolution for residual connection if dimensions change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        """
        # First causal conv
        out = self.conv1(x)
        if self.padding1 > 0:
            out = out[:, :, :-self.padding1]
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second causal conv
        out = self.conv2(out)
        if self.padding2 > 0:
            out = out[:, :, :-self.padding2]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNClassifier(nn.Module):
    """TCN-based multivariate time series classifier.
    
    Serves as a baseline for RQ1 (comparison with TGNNs) and
    RQ3 (generalization across dataset complexity).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Args:
            n_channels: Number of input variables/channels.
            n_classes: Number of output classes.
            num_channels: List of channel sizes for each TCN block.
                         Dilation doubles at each layer: [1, 2, 4, ...].
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()
        self.model_name = "TCN"

        if num_channels is None:
            num_channels = [64, 128, 128]

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = n_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels).
            
        Returns:
            Logits of shape (batch, n_classes).
        """
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply TCN blocks
        out = self.network(x)  # (batch, last_channel, seq_len)
        
        # Global average pooling
        out = self.global_pool(out).squeeze(-1)  # (batch, last_channel)
        
        logits = self.classifier(out)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
