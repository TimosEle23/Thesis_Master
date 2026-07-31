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
pip install -q torch_geometric torch_geometric_temporal omegaconf aeon \
    jsonargparse pyyaml scikit-learn seaborn 2>&1 | tail -2

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

echo "[4/4] Launching fine + correlation-mean sweep (parallel=$PARALLEL)..."
# FORCE_NUM_WORKERS=0: cheap in-memory data -> main-process loading avoids worker
# oversubscription when many experiments run at once.
nohup env FORCE_NUM_WORKERS=0 python3 -u scripts/run_phase2_threaded.py \
    --device cuda --parallel "$PARALLEL" \
    --graph_modes predefined_fine predefined_correlation_mean \
                  adaptive_correlation_mean adaptive_nosvd_correlation_mean \
    >> logs/vastai_phase2.log 2>&1 &
echo "  launched PID $! — monitor with:  tail -f logs/vastai_phase2.log"
echo ""
echo "Results land in results/phase2/. Download that folder (Jupyter file browser"
echo "or scp) when done — the results.json + fold_*_results.json are what the"
echo "analysis needs; best_model.pt are optional (large)."
