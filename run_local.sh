#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Run experiments locally using conda base environment
#
# Avoids iCloud Docker volume issues. Uses conda for PyTorch/PyG.
#
# Usage:
#   ./run_local.sh phase1                    # Run all Phase 1
#   ./run_local.sh phase2                    # Run all Phase 2
#   ./run_local.sh all                       # Run both phases
#   ./run_local.sh analyse                   # Generate analysis
#   ./run_local.sh single "--dataset DSA --model tcn --graph_mode none --seed 42 --phase 2"
#   ./run_local.sh dry                       # Preview experiment plan
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

# Detect device: prefer MPS on Apple Silicon, then CUDA, else CPU
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi
echo "Using device: $DEVICE"

case "${1:-help}" in
    phase1)
        python3 -m scripts.run_all --phase 1 --device "$DEVICE"
        ;;
    phase2)
        python3 -m scripts.run_all --phase 2 --device "$DEVICE"
        ;;
    all)
        python3 -m scripts.run_all --device "$DEVICE"
        ;;
    analyse)
        python3 -m scripts.analyse_results
        ;;
    single)
        shift
        echo "Running: python3 -m scripts.run_experiment $* --device $DEVICE"
        python3 -m scripts.run_experiment $* --device "$DEVICE"
        ;;
    dry)
        python3 -m scripts.run_all --dry_run
        ;;
    help|*)
        echo "Usage: $0 {phase1|phase2|all|analyse|single|dry|help}"
        echo ""
        echo "  phase1     Run all Phase 1 experiments"
        echo "  phase2     Run all Phase 2 experiments"
        echo "  all        Run both phases"
        echo "  analyse    Generate analysis tables & plots"
        echo "  single     Run single experiment (pass args in quotes)"
        echo "  dry        Preview experiment plan"
        echo ""
        echo "Detected device: $DEVICE"
        ;;
esac
