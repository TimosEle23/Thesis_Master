"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing.
    
    Helps prevent overconfident predictions on small datasets.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits (batch, n_classes).
            target: Class indices (batch,).
        """
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        
        # Create smoothed targets
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, target.unsqueeze(1), 1.0
        )
        targets_smooth = (
            (1 - self.smoothing) * targets_one_hot
            + self.smoothing / n_classes
        )
        
        loss = -(targets_smooth * log_probs).sum(dim=1)
        
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            loss = loss * weight[target]
        
        return loss.mean()


def get_loss_function(
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor = None,
) -> nn.Module:
    """Get loss function.
    
    Args:
        loss_type: 'cross_entropy' or 'label_smoothing'.
        label_smoothing: Smoothing parameter (0 = no smoothing).
        class_weights: Optional class weights for imbalanced datasets.
    """
    if label_smoothing > 0:
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=class_weights
        )
    else:
        return nn.CrossEntropyLoss(weight=class_weights)
