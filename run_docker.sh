#!/usr/bin/env bash
# =============================================================================
# run_docker.sh — Run experiments in Docker with iCloud-safe data handling
#
# On macOS with iCloud Drive, Docker volume mounts can hit file-locking issues.
# This script copies data into the container for reliable I/O, then copies
# results back to the host.
#
# Usage:
#   ./run_docker.sh phase1           # Run all Phase 1 experiments
#   ./run_docker.sh phase2           # Run all Phase 2 experiments
#   ./run_docker.sh all              # Run both phases
#   ./run_docker.sh analyse          # Generate analysis tables & plots
#   ./run_docker.sh shell            # Interactive shell in container
#   ./run_docker.sh single "ARGS"    # Run a single experiment with custom args
#
# Example:
#   ./run_docker.sh single "--dataset DSA --model tcn --graph_mode none --seed 42 --phase 2 --device cpu"
# =============================================================================
set -euo pipefail

IMAGE="thesis-tgnn"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build image if needed
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image..."
    docker compose build
fi

run_in_container() {
    local CMD="$1"
    docker run --rm \
        -v "${PROJECT_DIR}/src:/app/src:ro" \
        -v "${PROJECT_DIR}/scripts:/app/scripts:ro" \
        -v "${PROJECT_DIR}/configs:/app/configs:ro" \
        -v "${PROJECT_DIR}/data:/app/data" \
        -v "${PROJECT_DIR}/results:/app/results" \
        -v "${PROJECT_DIR}/UCI:/app/UCI:ro" \
        -v "${PROJECT_DIR}/BasicMotions:/app/BasicMotions:ro" \
        -v "${PROJECT_DIR}/Epilepsy:/app/Epilepsy:ro" \
        -e PYTHONPATH=/app \
        -w /app \
        "$IMAGE" \
        bash -c "$CMD"
}

case "${1:-help}" in
    phase1)
        echo "Running Phase 1 (BasicMotions + Epilepsy)..."
        run_in_container "python -m scripts.run_all --phase 1 --device cpu"
        ;;
    phase2)
        echo "Running Phase 2 (DSA LOSO)..."
        run_in_container "python -m scripts.run_all --phase 2 --device cpu"
        ;;
    all)
        echo "Running all experiments..."
        run_in_container "python -m scripts.run_all --device cpu"
        ;;
    analyse)
        echo "Running analysis..."
        run_in_container "python -m scripts.analyse_results"
        ;;
    shell)
        echo "Starting interactive shell..."
        docker run --rm -it \
            -v "${PROJECT_DIR}/src:/app/src" \
            -v "${PROJECT_DIR}/scripts:/app/scripts" \
            -v "${PROJECT_DIR}/configs:/app/configs" \
            -v "${PROJECT_DIR}/data:/app/data" \
            -v "${PROJECT_DIR}/results:/app/results" \
            -v "${PROJECT_DIR}/UCI:/app/UCI" \
            -v "${PROJECT_DIR}/BasicMotions:/app/BasicMotions" \
            -v "${PROJECT_DIR}/Epilepsy:/app/Epilepsy" \
            -e PYTHONPATH=/app \
            -w /app \
            "$IMAGE" \
            bash
        ;;
    single)
        shift
        echo "Running: python -m scripts.run_experiment $*"
        run_in_container "python -m scripts.run_experiment $*"
        ;;
    dry)
        echo "Dry run — preview all experiments..."
        run_in_container "python -m scripts.run_all --dry_run"
        ;;
    help|*)
        echo "Usage: $0 {phase1|phase2|all|analyse|shell|single|dry|help}"
        echo ""
        echo "  phase1    Run all Phase 1 experiments"
        echo "  phase2    Run all Phase 2 experiments"
        echo "  all       Run both phases"
        echo "  analyse   Generate analysis tables & plots"
        echo "  shell     Interactive container shell"
        echo "  single    Run single experiment (pass args in quotes)"
        echo "  dry       Preview experiment plan"
        ;;
esac
