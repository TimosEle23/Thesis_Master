#!/usr/bin/env bash
# =============================================================================
# vastai_run.sh — run the remaining Phase 2 DSA experiments on a Vast.ai box.
#
# Runs predefined_fine + the correlation-mean family (predefined / adaptive-SVD /
# adaptive-noSVD) — i.e. "predef fine that's left + the rest". Leaves the plain
# `adaptive` runs to Colab so the two don't duplicate work.
#
# Uses run_phase2_threaded with --parallel: because these recurrent TGNNs are
# launch-bound (~15-20% GPU each), running several at once fills the idle A100
# and multiplies throughput. torch.compile stays ON (amortised over 8 folds per
# experiment). Resumable: any experiment whose results.json already exists is
# skipped, so upload the resume bundle first (see USAGE) to avoid redoing the
# ~13 already-finished fine runs.
#
# USAGE (in the Vast.ai Jupyter terminal):
#   git clone --depth=1 --branch phase2-colab https://github.com/TimosEle23/Thesis_Master.git
#   cd Thesis_Master
#   # (optional but recommended) upload thesis_p2_resume_results.zip here, then:
#   #   unzip -o thesis_p2_resume_results.zip        # -> results/phase2/DSA_*/results.json
#   PARALLEL=6 bash scripts/vastai_run.sh
#
# Tune PARALLEL (default 6). More = faster while GPU/CPU allow; drop it if the
# box has few CPUs or you hit OOM.
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs
PARALLEL="${PARALLEL:-6}"

echo "==================================================================="
echo " Vast.ai Phase 2 runner — fine + correlation-mean, parallel=$PARALLEL"
echo "==================================================================="

echo "[1/4] Installing Python deps (image already has torch+cuda)..."
pip install -q torch_geometric omegaconf aeon jsonargparse pyyaml \
    scikit-learn seaborn scipy six decorator networkx tqdm 2>&1 | tail -2
# torch_geometric_temporal depends on torch-sparse / torch-scatter, whose wheels fail
# to build against this torch. Install it WITHOUT them and stub torch_sparse — the models
# used here (GConvLSTM/GConvGRU/DCRNN/A3TGCN/hybrid) never touch EvolveGCN/SparseTensor.
pip install -q --no-deps torch_geometric_temporal 2>&1 | tail -1
python3 - <<'PYSTUB'
import site, os
p = os.path.join(site.getsitepackages()[0], "torch_sparse")
os.makedirs(p, exist_ok=True)
with open(os.path.join(p, "__init__.py"), "w") as f:
    f.write("try:\n    from torch_geometric.typing import SparseTensor\n"
            "except Exception:\n    class SparseTensor:\n"
            "        def __init__(self, *a, **k):\n            raise RuntimeError('torch_sparse stub')\n"
            "__version__ = '0.0.0-stub'\n")
print("  stubbed torch_sparse")
PYSTUB
echo "  verifying imports..."
python3 -c "import torch, torch_geometric, torch_geometric_temporal, sklearn, omegaconf, aeon; \
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU, DCRNN, A3TGCN; \
print('  deps OK — torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "  ERROR: dependency import failed — aborting (not launching a broken sweep)."; exit 1; }

echo "[2/4] Preparing DSA data (download from UCI + preprocess, ~5-10 min once)..."
python3 - <<'PY'
import os, zipfile, urllib.request, shutil
from pathlib import Path
if os.path.exists("data/processed/DSA/X.npy"):
    print("  DSA already processed."); raise SystemExit
url = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"
zp = "/tmp/dsa.zip"
if not os.path.exists(zp):
    print("  downloading DSA (~170 MB)..."); urllib.request.urlretrieve(url, zp)
ex = "/tmp/dsa_ex"
with zipfile.ZipFile(zp) as z: z.extractall(ex)
root = None
for r, d, _ in os.walk(ex):
    if any(x.startswith("a0") for x in d): root = r; break
if root is None: raise SystemExit("could not find a0x activity folders in the UCI zip")
if os.path.exists("UCI"): shutil.rmtree("UCI")
shutil.copytree(root, "UCI")
import sys; sys.path.insert(0, ".")
from src.data.preprocessing import preprocess_dataset
data = preprocess_dataset("DSA", force=True)
print(f"  DSA ready: X={data['X'].shape}")
PY

echo "[3/4] Already-complete experiments present (will be skipped):"
find results/phase2 -name results.json 2>/dev/null | wc -l | xargs echo "  results.json on disk:"

echo "[4/4] Launching sweep — predefined_fine FIRST, then correlation-mean (parallel=$PARALLEL)..."
# FORCE_NUM_WORKERS=0: cheap in-memory data -> main-process loading avoids worker
# oversubscription when many experiments run at once.
# Two chained calls so the unfinished predefined_fine runs complete before the
# larger correlation-mean bonus family (run_phase2_threaded's own priority list
# would otherwise put fine last).
nohup bash -c "
  env FORCE_NUM_WORKERS=0 python3 -u scripts/run_phase2_threaded.py \
      --device cuda --parallel $PARALLEL --graph_modes predefined_fine
  env FORCE_NUM_WORKERS=0 python3 -u scripts/run_phase2_threaded.py \
      --device cuda --parallel $PARALLEL \
      --graph_modes predefined_correlation_mean adaptive_correlation_mean adaptive_nosvd_correlation_mean
" >> logs/vastai_phase2.log 2>&1 &
echo "  launched PID $! — fine first, then correlation. monitor: tail -f logs/vastai_phase2.log"
echo ""
echo "Results land in results/phase2/. Download that folder (Jupyter file browser"
echo "or scp) when done — the results.json + fold_*_results.json are what the"
echo "analysis needs; best_model.pt are optional (large)."
