#!/bin/bash
# collect_visualizations.sh — Gather all benchmark visualizations into one directory.
#
# Finds the most recent output for each benchmark category (e2e, ablation, burst,
# stair, pareto, fidelity) and runs the corresponding visualization script, saving
# all plots into a single timestamped results directory.
#
# Usage:
#   ./collect_visualizations.sh                     # auto-detect all available results
#   ./collect_visualizations.sh --output-dir <path>  # specify output directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--output-dir <path>]"
            echo "  Collects visualizations from the most recent benchmark results."
            echo "  Default output: results_<timestamp>/ under benchmarks/energy/"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

# Find the most recently modified timestamped directory under a parent.
latest_dir() {
    local parent="$1"
    # shellcheck disable=SC2012
    ls -dt "$parent"/*/ 2>/dev/null | head -1
}

echo ""
echo "====== Collecting Visualizations ======"
echo "Output: $OUTPUT_DIR"

COUNT=0

# E2E
E2E_DIR=$(latest_dir "$SCRIPT_DIR/end_to_end")
if [[ -n "$E2E_DIR" ]]; then
    echo "  e2e        ← $E2E_DIR"
    mkdir -p "$OUTPUT_DIR/e2e"
    python3 "$SCRIPT_DIR/visualization/visualize_multiple_traces.py" \
        "$E2E_DIR" --output-dir "$OUTPUT_DIR/e2e" || true
    ((COUNT++)) || true
fi

# Ablation
ABLATION_DIR=$(latest_dir "$SCRIPT_DIR/sensitivity_tests/ablation")
if [[ -n "$ABLATION_DIR" ]]; then
    echo "  ablation   ← $ABLATION_DIR"
    mkdir -p "$OUTPUT_DIR/ablation"
    python3 "$SCRIPT_DIR/visualization/visualize_multiple_traces.py" \
        "$ABLATION_DIR" --output-dir "$OUTPUT_DIR/ablation" || true
    ((COUNT++)) || true
fi

# Burst
BURST_DIR=$(latest_dir "$SCRIPT_DIR/sensitivity_tests/burst")
if [[ -n "$BURST_DIR" ]]; then
    echo "  burst      ← $BURST_DIR"
    mkdir -p "$OUTPUT_DIR/burst"
    python3 "$SCRIPT_DIR/visualization/visualize_timeline.py" \
        "$BURST_DIR" --window-size 3.0 --save "$OUTPUT_DIR/burst/timeline.png" || true
    ((COUNT++)) || true
fi

# Stair
STAIR_DIR=$(latest_dir "$SCRIPT_DIR/sensitivity_tests/stair")
if [[ -n "$STAIR_DIR" ]]; then
    echo "  stair      ← $STAIR_DIR"
    mkdir -p "$OUTPUT_DIR/stair"
    python3 "$SCRIPT_DIR/visualization/visualize_timeline.py" \
        "$STAIR_DIR" --window-size 3.0 --save "$OUTPUT_DIR/stair/timeline.png" || true
    ((COUNT++)) || true
fi

# Pareto
PARETO_DIR=$(latest_dir "$SCRIPT_DIR/sensitivity_tests/pareto")
if [[ -n "$PARETO_DIR" ]]; then
    echo "  pareto     ← $PARETO_DIR"
    mkdir -p "$OUTPUT_DIR/pareto"
    python3 "$SCRIPT_DIR/visualization/visualize_multiple_traces.py" \
        "$PARETO_DIR" --output-dir "$OUTPUT_DIR/pareto" || true
    ((COUNT++)) || true
fi

# Fidelity
FIDELITY_DIR=$(latest_dir "$SCRIPT_DIR/sensitivity_tests/fidelity")
if [[ -n "$FIDELITY_DIR" ]]; then
    echo "  fidelity   ← $FIDELITY_DIR"
    mkdir -p "$OUTPUT_DIR/fidelity"
    python3 "$SCRIPT_DIR/visualization/visualize_fidelity.py" \
        "$FIDELITY_DIR" --output-dir "$OUTPUT_DIR/fidelity" || true
    ((COUNT++)) || true
fi

if [[ $COUNT -eq 0 ]]; then
    echo "  (no benchmark results found)"
    rmdir "$OUTPUT_DIR" 2>/dev/null || true
else
    echo ""
    echo "Collected visualizations from $COUNT benchmark(s) into:"
    echo "  $OUTPUT_DIR"
fi

echo "====== Done: Visualizations ======"
