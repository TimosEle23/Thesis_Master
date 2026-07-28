import torch
import torch.nn as nn
from src.models.tcn import TCNClassifier
model = TCNClassifier(n_channels=3, n_classes=2)
model = model.to('cpu')
print("TCN initialized and moved to CPU successfully!")
