#!/usr/bin/env bash
# ============================================================
# prepare_colab.sh
# Creates thesis_colab.zip for one-time upload to Google Drive.
# Run from ANY directory:   bash scripts/prepare_colab.sh
# ============================================================

set -e

# Go to project root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ZIP="$PROJECT_ROOT/thesis_colab.zip"

echo "=================================================="
echo "  Thesis Master — Colab Upload Prep"
echo "  Project root: $PROJECT_ROOT"
echo "=================================================="
echo ""

# Remove stale zip if it exists
[ -f "$OUTPUT_ZIP" ] && rm "$OUTPUT_ZIP" && echo "Removed old thesis_colab.zip"

echo "Packing files..."
echo "(This may take ~15 seconds for the 196 MB DSA data)"
echo ""

zip -r "$OUTPUT_ZIP" \
    src/ \
    scripts/ \
    configs/ \
    data/processed/DSA/ \
    data/raw/BasicMotions/ \
    data/raw/Epilepsy/ \
    -x "*.pyc" \
    -x "*/__pycache__/*" \
    -x "*/.DS_Store" \
    -x "*.ipynb_checkpoints/*" \
    2>&1 | grep -E "^(adding|error)" | tail -20

# Also add the one complete Phase 2 result so run_all.py skips it
if [ -d "results/phase2/DSA_lstm_none_seed42" ]; then
    echo ""
    echo "Including already-complete Phase 2 result (DSA_lstm_none_seed42)..."
    zip -r "$OUTPUT_ZIP" results/phase2/DSA_lstm_none_seed42/ \
        -x "*.pyc" -x "*/__pycache__/*" 2>&1 | grep "^adding" | wc -l | xargs -I{} echo "  {} files added"
fi

echo ""
echo "=================================================="
echo "  DONE!"
echo "  Output: thesis_colab.zip"
printf "  Size:   "; du -sh "$OUTPUT_ZIP" | cut -f1
echo ""
echo "  Next steps:"
echo "  1. Upload thesis_colab.zip to Google Drive"
echo "     (drag it into drive.google.com, or use Drive desktop app)"
echo "  2. Open notebooks/colab_phase2.ipynb in Google Colab"
echo "     (File → Open notebook → Google Drive)"
echo "  3. Follow the cells top to bottom"
echo "=================================================="
