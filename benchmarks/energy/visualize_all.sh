#!/bin/bash

# Automatically finds the most recent experiment directories and runs the
# appropriate visualization script for each experiment type.
#
# Usage:
#   ./visualize_all.sh [options]
#
# Options (all optional — auto-detected when omitted):
#   --end-to-end <dir>   Directory from run_e2e.sh
#   --ablation <dir>     Directory from run_ablation.sh
#   --burst <dir>        Directory from run_burst.sh
#   --stair <dir>        Directory from run_stair.sh
#   --fidelity <dir>     Directory from run_fidelity.sh
#   --burst-window <s>   Window size for burst timeline (default: 5.0)
#   --stair-window <s>   Window size for stair timeline (default: 3.0)
#   -h|--help

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIZ_DIR="$SCRIPT_DIR/visualization"

END_TO_END_DIR=""
ABLATION_DIR=""
BURST_DIR=""
STAIR_DIR=""
FIDELITY_DIR=""
BURST_WINDOW="5.0"
STAIR_WINDOW="3.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --end-to-end)   END_TO_END_DIR="$2"; shift 2 ;;
        --ablation)     ABLATION_DIR="$2";   shift 2 ;;
        --burst)        BURST_DIR="$2";      shift 2 ;;
        --stair)        STAIR_DIR="$2";      shift 2 ;;
        --fidelity)     FIDELITY_DIR="$2";   shift 2 ;;
        --burst-window) BURST_WINDOW="$2";   shift 2 ;;
        --stair-window) STAIR_WINDOW="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--end-to-end <dir>] [--ablation <dir>] [--burst <dir>] [--stair <dir>] [--fidelity <dir>]"
            echo "       [--burst-window <s>] [--stair-window <s>]"
            echo ""
            echo "Without arguments, automatically finds the most recent result directory for each"
            echo "experiment type and runs the corresponding visualization script."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Return the most recently created subdirectory under $1 (uses timestamp-based name sort)
find_latest_dir() {
    local parent="$1"
    [ -d "$parent" ] || return
    find "$parent" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1
}

# Auto-detect if not specified
[ -z "$END_TO_END_DIR" ] && END_TO_END_DIR="$(find_latest_dir "$SCRIPT_DIR/end_to_end")"
[ -z "$ABLATION_DIR"   ] && ABLATION_DIR="$(find_latest_dir "$SCRIPT_DIR/sensitivity_tests/ablation")"
[ -z "$BURST_DIR"      ] && BURST_DIR="$(find_latest_dir "$SCRIPT_DIR/sensitivity_tests/burst")"
[ -z "$STAIR_DIR"      ] && STAIR_DIR="$(find_latest_dir "$SCRIPT_DIR/sensitivity_tests/stair")"
[ -z "$FIDELITY_DIR"   ] && FIDELITY_DIR="$(find_latest_dir "$SCRIPT_DIR/sensitivity_tests/fidelity")"

# Strip trailing slashes
END_TO_END_DIR="${END_TO_END_DIR%/}"
ABLATION_DIR="${ABLATION_DIR%/}"
BURST_DIR="${BURST_DIR%/}"
STAIR_DIR="${STAIR_DIR%/}"
FIDELITY_DIR="${FIDELITY_DIR%/}"

echo "=============================================="
echo "Visualization — Auto-detected Directories"
echo "=============================================="
printf "  %-14s %s\n" "End-to-end:"  "${END_TO_END_DIR:-<not found>}"
printf "  %-14s %s\n" "Ablation:"    "${ABLATION_DIR:-<not found>}"
printf "  %-14s %s\n" "Burst:"       "${BURST_DIR:-<not found>}"
printf "  %-14s %s\n" "Stair:"       "${STAIR_DIR:-<not found>}"
printf "  %-14s %s\n" "Fidelity:"    "${FIDELITY_DIR:-<not found>}"
echo "=============================================="

RAN_ANY=0

run_step() {
    local label="$1"
    local dir="$2"
    shift 2   # remaining args are the python command + args

    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo ""
        echo "[SKIP] $label — no directory found"
        return
    fi

    echo ""
    echo "====== $label ======"
    echo "  dir: $dir"
    python3 "$@" "$dir"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "WARNING: $label failed (exit code $rc)"
    else
        echo "Done: $label"
        RAN_ANY=1
    fi
}

run_step "End-to-End (visualize_multiple_traces)" \
    "$END_TO_END_DIR" \
    "$VIZ_DIR/visualize_multiple_traces.py"

run_step "Ablation (visualize_multiple_traces)" \
    "$ABLATION_DIR" \
    "$VIZ_DIR/visualize_multiple_traces.py"

# Timeline scripts take --window-size before the positional dir arg, so we
# call python directly here to keep arg order correct.
if [ -n "$BURST_DIR" ] && [ -d "$BURST_DIR" ]; then
    echo ""
    echo "====== Burst (visualize_timeline) ======"
    echo "  dir: $BURST_DIR"
    python3 "$VIZ_DIR/visualize_timeline.py" "$BURST_DIR" --window-size "$BURST_WINDOW"
    [ $? -eq 0 ] && { echo "Done: Burst"; RAN_ANY=1; } || echo "WARNING: Burst visualization failed"
else
    echo ""
    echo "[SKIP] Burst (visualize_timeline) — no directory found"
fi

if [ -n "$STAIR_DIR" ] && [ -d "$STAIR_DIR" ]; then
    echo ""
    echo "====== Stair (visualize_timeline) ======"
    echo "  dir: $STAIR_DIR"
    python3 "$VIZ_DIR/visualize_timeline.py" "$STAIR_DIR" --window-size "$STAIR_WINDOW"
    [ $? -eq 0 ] && { echo "Done: Stair"; RAN_ANY=1; } || echo "WARNING: Stair visualization failed"
else
    echo ""
    echo "[SKIP] Stair (visualize_timeline) — no directory found"
fi

run_step "Fidelity (visualize_fidelity)" \
    "$FIDELITY_DIR" \
    "$VIZ_DIR/visualize_fidelity.py"

echo ""
if [ "$RAN_ANY" -eq 0 ]; then
    echo "No experiment directories found or all visualizations failed."
    echo "Run some benchmarks first (run_e2e.sh, run_ablation.sh, run_burst.sh, etc.)"
    exit 1
else
    echo "=============================================="
    echo "All visualizations complete."
    echo "=============================================="
fi
