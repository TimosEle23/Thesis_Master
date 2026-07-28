import torch
from src.models.tcn import TCNClassifier
model = TCNClassifier(3, 2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
print("Model moved to device.")
# Now let's try deepcopy just in case
import copy
m2 = copy.deepcopy(model)
m2 = m2.to('cpu')
