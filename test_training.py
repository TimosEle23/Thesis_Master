"""Quick mini-training test: trains LSTM for 5 epochs on BasicMotions."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.download import download_all_datasets, ensure_dirs
from src.data.preprocessing import preprocess_dataset
from src.data.splits import get_data_splits
from src.data.datasets import MTSDataset, GraphMTSDataset, graph_collate_fn
from src.graphs.predefined import build_predefined_graph
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics
import torch

set_seed(42)
device = get_device("cpu")
print(f"Device: {device}")

# Data
ensure_dirs()
download_all_datasets()
data = preprocess_dataset("BasicMotions")
splits = get_data_splits("BasicMotions", data, seed=42)

n_ch = data["metadata"]["n_channels"]
n_cls = data["metadata"]["n_classes"]
seq_len = data["metadata"]["seq_len"]

# --- Test 1: LSTM baseline ---
print("\n--- LSTM mini-training ---")
model = build_model("lstm", n_ch, n_cls, seq_len, {"hidden_dim": 32, "num_layers": 1})
ds_train = MTSDataset(splits["X_train"], splits["y_train"])
ds_val = MTSDataset(splits["X_val"], splits["y_val"])
ds_test = MTSDataset(splits["X_test"], splits["y_test"])

cfg = {"epochs": 5, "patience": 10, "batch_size": 8, "learning_rate": 0.001,
       "weight_decay": 0.0001, "scheduler": "cosine", "gradient_clip_val": 1.0,
       "label_smoothing": 0.0}

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=8)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=8)

trainer = Trainer(model, device, cfg, save_dir="results/test", experiment_name="lstm_test")
result = trainer.train(train_loader, val_loader)
print(f"  Training done: best_epoch={result['best_epoch']}, val_acc={result['best_val_acc']:.4f}")

metrics = trainer.evaluate(test_loader)
print(f"  Test accuracy: {metrics['accuracy']:.4f}, macro_f1: {metrics['macro_f1']:.4f}")

# --- Test 2: GConvGRU with predefined graph ---
print("\n--- GConvGRU mini-training ---")
gc = build_predefined_graph("BasicMotions")
import numpy as np
ei = gc["edge_index"].numpy() if isinstance(gc["edge_index"], torch.Tensor) else gc["edge_index"]
ew = gc["edge_weight"].numpy() if isinstance(gc["edge_weight"], torch.Tensor) else gc["edge_weight"]

model2 = build_model("gconv_gru", n_ch, n_cls, seq_len,
                      {"hidden_dim": 16, "num_layers": 1, "K": 2}, gc, "predefined")

g_train = GraphMTSDataset(splits["X_train"], splits["y_train"], ei, ew)
g_val = GraphMTSDataset(splits["X_val"], splits["y_val"], ei, ew)
g_test = GraphMTSDataset(splits["X_test"], splits["y_test"], ei, ew)

g_train_loader = torch.utils.data.DataLoader(g_train, batch_size=8, collate_fn=graph_collate_fn)
g_val_loader = torch.utils.data.DataLoader(g_val, batch_size=8, collate_fn=graph_collate_fn)
g_test_loader = torch.utils.data.DataLoader(g_test, batch_size=8, collate_fn=graph_collate_fn)

trainer2 = Trainer(model2, device, cfg, save_dir="results/test", experiment_name="gconvgru_test")
result2 = trainer2.train(g_train_loader, g_val_loader)
print(f"  Training done: best_epoch={result2['best_epoch']}, val_acc={result2['best_val_acc']:.4f}")

metrics2 = trainer2.evaluate(g_test_loader)
print(f"  Test accuracy: {metrics2['accuracy']:.4f}, macro_f1: {metrics2['macro_f1']:.4f}")

print("\n=== MINI-TRAINING TEST PASSED ===")
