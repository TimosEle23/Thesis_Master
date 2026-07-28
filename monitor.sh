#!/bin/bash
# Monitor Phase 2 experiment progress
LOG="/tmp/thesis_data/phase2_full.log"
RESULTS="/tmp/thesis_data/results/phase2"

echo "=========================================="
echo "  Phase 2 Experiment Monitor"
echo "=========================================="

# Process status
PID=$(pgrep -f "run_all.*phase 2" 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Status: RUNNING (PID $PID)"
    ps -p "$PID" -o pcpu,pmem,time 2>/dev/null | tail -1
else
    echo "Status: NOT RUNNING"
fi

echo ""
echo "=== Completed experiments ==="
COMPLETE=$(find "$RESULTS" -name "results.json" 2>/dev/null | wc -l | tr -d ' ')
echo "  Complete: $COMPLETE / 75"

echo ""
echo "=== Completed list ==="
find "$RESULTS" -name "results.json" -exec sh -c 'echo "  $(dirname {} | xargs basename): $(python3 -c "import json; d=json.load(open(\"{}\")); print(f\"acc={d.get(\"accuracy_mean\", d.get(\"accuracy\", \"?\"))} f1={d.get(\"macro_f1_mean\", d.get(\"macro_f1\", \"?\"))}\")" 2>/dev/null)' \; 2>/dev/null | sort

echo ""
echo "=== Active fold dirs ==="
for d in "$RESULTS"/DSA_*_fold*; do
    [ -d "$d" ] && echo "  $(basename "$d")"
done 2>/dev/null

echo ""
echo "=== Errors ==="
if [ -f "$LOG" ]; then
    ERR=$(grep -c "ERROR\|FAILED\|cannot be opened\|RuntimeError" "$LOG" 2>/dev/null)
    echo "  Error count: $ERR"
    [ "$ERR" -gt 0 ] && grep "ERROR\|FAILED" "$LOG" | head -5
fi

echo ""
echo "=== Last 5 log lines ==="
[ -f "$LOG" ] && tail -5 "$LOG"
