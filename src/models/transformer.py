"""
Transformer baseline classifier for multivariate time series.

Based on the encoder-only Transformer architecture with positional encoding.
Processes the full multivariate input as a token at each time step.

Architecture:
  Input (batch, seq_len, n_channels) → Linear projection → Positional Encoding →
  Transformer Encoder layers → [CLS] token or mean pooling → FC → classes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer-based multivariate time series classifier.
    
    Serves as a baseline for RQ1 (comparison with TGNNs) and
    RQ3 (generalization across dataset complexity).
    
    Uses a [CLS] token prepended to the sequence for classification.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
        max_seq_len: int = 500,
    ):
        """
        Args:
            n_channels: Number of input variables/channels.
            n_classes: Number of output classes.
            d_model: Transformer model dimension.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: FFN hidden dimension.
            dropout: Dropout rate.
            positional_encoding: 'sinusoidal' or 'learnable'.
            max_seq_len: Maximum supported sequence length.
        """
        super().__init__()
        self.model_name = "Transformer"
        self.d_model = d_model

        # Input projection: n_channels → d_model
        self.input_projection = nn.Linear(n_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model, max_len=max_seq_len + 1, dropout=dropout
            )
        else:
            self.pos_encoder = LearnablePositionalEncoding(
                d_model, max_len=max_seq_len + 1, dropout=dropout
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels).
            
        Returns:
            Logits of shape (batch, n_classes).
        """
        batch_size = x.size(0)

        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.input_norm(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, 1 + seq_len, d_model)

        # Use [CLS] token output for classification
        cls_output = x[:, 0]  # (batch, d_model)
        cls_output = self.output_norm(cls_output)

        logits = self.classifier(cls_output)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
