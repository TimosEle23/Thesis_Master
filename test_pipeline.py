"""Quick sanity check: import all modules and run a minimal training loop."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --- 1. Imports ---
print("Testing imports...")
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging_utils import setup_logger
print("  Utils: OK")

from src.data.download import download_all_datasets, ensure_dirs
print("  Download: OK")

from src.data.preprocessing import preprocess_dataset
print("  Preprocessing: OK")

from src.data.splits import get_data_splits
print("  Splits: OK")

from src.data.datasets import MTSDataset, GraphMTSDataset, graph_collate_fn
print("  Datasets: OK")

from src.graphs.predefined import build_predefined_graph
from src.graphs.adaptive import AdaptiveGraphLearner
print("  Graphs: OK")

from src.models.factory import build_model, ALL_MODELS
print(f"  Models: OK ({ALL_MODELS})")

from src.training.losses import get_loss_function
from src.training.trainer import Trainer
print("  Training: OK")

from src.evaluation.metrics import compute_metrics
from src.evaluation.analysis import run_statistical_analysis
print("  Evaluation: OK")

print("\nAll imports successful!\n")

# --- 2. Data loading test ---
print("Testing data pipeline...")
ensure_dirs()
download_all_datasets()
print("  Download/parse: OK")

data_bm = preprocess_dataset("BasicMotions")
print(f"  BasicMotions: train={data_bm['X_train'].shape}, test={data_bm['X_test'].shape}")

splits = get_data_splits("BasicMotions", data_bm, seed=42)
print(f"  Splits: train={splits['X_train'].shape}, val={splits['X_val'].shape}, test={splits['X_test'].shape}")

# --- 3. Graph test ---
print("\nTesting graph construction...")
g = build_predefined_graph("BasicMotions")
print(f"  BasicMotions graph: {g['num_nodes']} nodes, {g['edge_index'].shape[1]} edges")

g_epi = build_predefined_graph("Epilepsy")
print(f"  Epilepsy graph: {g_epi['num_nodes']} nodes, {g_epi['edge_index'].shape[1]} edges")

g_dsa = build_predefined_graph("DSA", granularity="coarse")
print(f"  DSA coarse graph: {g_dsa['num_nodes']} nodes, {g_dsa['edge_index'].shape[1]} edges")

# --- 4. Model build test ---
print("\nTesting model construction...")
import torch

n_channels = data_bm["metadata"]["n_channels"]
n_classes = data_bm["metadata"]["n_classes"]
seq_len = data_bm["metadata"]["seq_len"]

for model_name in ["lstm", "tcn", "transformer"]:
    m = build_model(model_name, n_channels, n_classes, seq_len, {})
    print(f"  {model_name}: {m.count_parameters()} params")

for model_name in ["gconv_lstm", "gconv_gru", "a3tgcn", "dcrnn"]:
    gc = build_predefined_graph("BasicMotions")
    m = build_model(model_name, n_channels, n_classes, seq_len, {}, gc, "predefined")
    print(f"  {model_name} (predefined): {m.count_parameters()} params")

# --- 5. Quick forward pass ---
print("\nTesting forward passes...")
device = get_device("cpu")

# Baseline
ds = MTSDataset(splits["X_train"], splits["y_train"])
loader = torch.utils.data.DataLoader(ds, batch_size=4)
x, y = next(iter(loader))

for model_name in ["lstm", "tcn", "transformer"]:
    m = build_model(model_name, n_channels, n_classes, seq_len, {}).to(device)
    out = m(x.to(device))
    print(f"  {model_name}: input={x.shape} -> output={out.shape}")

# TGNN
import numpy as np
gc = build_predefined_graph("BasicMotions")
ei = gc["edge_index"].numpy() if isinstance(gc["edge_index"], torch.Tensor) else gc["edge_index"]
ew = gc["edge_weight"].numpy() if isinstance(gc["edge_weight"], torch.Tensor) else gc["edge_weight"]
graph_ds = GraphMTSDataset(splits["X_train"], splits["y_train"], ei, ew)
graph_loader = torch.utils.data.DataLoader(graph_ds, batch_size=4, collate_fn=graph_collate_fn)
batch = next(iter(graph_loader))

for model_name in ["gconv_lstm", "gconv_gru"]:
    m = build_model(model_name, n_channels, n_classes, seq_len, {}, gc, "predefined").to(device)
    out = m(batch["x"].to(device))
    print(f"  {model_name}: input={batch['x'].shape} -> output={out.shape}")

print("\n=== ALL TESTS PASSED ===")
