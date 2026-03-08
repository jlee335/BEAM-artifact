#!/bin/bash

# Full self-contained pipeline for Pareto frontier analysis.
# Requires exactly 8 GPUs (tests tp4_pp2 and tp2_pp4 configurations).
#
# Usage:
#   ./run_pareto_pipeline.sh [--model <name>]
#
# Example:
#   ./run_pareto_pipeline.sh --model meta-llama/Llama-3.3-70B-Instruct

SCRIPT_DIR="$(dirname "$0")"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
REQUIRED_GPUS=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--model <name>]"
            echo ""
            echo "Runs the full Pareto frontier pipeline:"
            echo "  1) System requirement checks (HF_TOKEN, 8 GPUs, disk space)"
            echo "  2) Offline profiling for tp4_pp2 and tp2_pp4"
            echo "  3) DynamoLLM profiling for tp4_pp2 and tp2_pp4"
            echo "  4) Pareto experiment (run_pareto.sh)"
            echo "  5) Visualization (sensitivity_tests/visualize_pareto.py)"
            echo ""
            echo "Options:"
            echo "  --model <name>   Model name (default: meta-llama/Llama-3.3-70B-Instruct)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────
# System requirement checks
# ──────────────────────────────────────────────

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please export your Hugging Face token first:"
    echo "  export HF_TOKEN=<your_huggingface_token>"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -ne "$REQUIRED_GPUS" ]; then
    echo "Error: Pareto pipeline requires exactly ${REQUIRED_GPUS} GPUs."
    echo "  Detected: ${NUM_GPUS} GPU(s)"
    echo "  Configurations tested: tp4_pp2 and tp2_pp4 (each requires 8 GPUs)"
    exit 1
fi
echo "GPU check passed: ${NUM_GPUS} GPUs detected."

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "====== Step: $step_name ======"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: '$step_name' failed (exit code $rc). Aborting."
        exit $rc
    fi
    echo "====== Done: $step_name ======"
}

# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

echo ""
echo "=============================================="
echo "Pareto Frontier Pipeline"
echo "=============================================="
echo "Model:    $MODEL"
echo "Configs:  tp4_pp2, tp2_pp4"
echo "=============================================="

# Step 1 & 2: Offline profiling for each parallelism config
run_step "Offline Profiling (tp=4, pp=2)" \
    bash "$SCRIPT_DIR/run_offline_profile.sh" --model "$MODEL" --tp 4 --pp 2

run_step "Offline Profiling (tp=2, pp=4)" \
    bash "$SCRIPT_DIR/run_offline_profile.sh" --model "$MODEL" --tp 2 --pp 4

# Step 3 & 4: DynamoLLM profiling for each parallelism config
run_step "DynamoLLM Profiling (tp=4, pp=2)" \
    bash "$SCRIPT_DIR/run_dynamo_benchmark.sh" --model "$MODEL" --tp 4 --pp 2

run_step "DynamoLLM Profiling (tp=2, pp=4)" \
    bash "$SCRIPT_DIR/run_dynamo_benchmark.sh" --model "$MODEL" --tp 2 --pp 4

# Step 5: Pareto experiment
run_step "Pareto Experiment" \
    bash "$SCRIPT_DIR/run_pareto.sh" --model "$MODEL"

# Step 6: Visualize
echo ""
echo "====== Step: Visualization ======"
PARETO_DIR=$(ls -td "$SCRIPT_DIR/sensitivity_tests/pareto/pareto_analysis_"*/ 2>/dev/null | head -1)
if [ -z "$PARETO_DIR" ]; then
    echo "ERROR: No pareto output directory found under sensitivity_tests/pareto/. Aborting."
    exit 1
fi
echo "Visualizing: $PARETO_DIR"
python3 "$SCRIPT_DIR/sensitivity_tests/visualize_pareto.py" "$PARETO_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Visualization failed. Aborting."
    exit 1
fi
echo "====== Done: Visualization ======"

echo ""
echo "=============================================="
echo "Pareto pipeline complete!"
echo "Results:  $PARETO_DIR"
echo "=============================================="