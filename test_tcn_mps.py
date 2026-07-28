import torch
from src.models.tcn import TCNClassifier
import torch.nn.utils.parametrizations as param
model = TCNClassifier(3, 2)
if torch.backends.mps.is_available():
    model = model.to('mps')
    print("Moved to mps")
