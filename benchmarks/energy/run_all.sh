#!/bin/bash

# Full artifact evaluation pipeline.
# Runs all benchmarks in order:
#   1) Offline profiling
#   2) End-to-end ablation (Vanilla vs DynamoLLM vs S1+S2)
#   3) Ablation study (A: Vanilla, B2: DynamoLLM, B3: S1-DVFS-only, C: S1-only, D: S1+S2)
#   4) Burst pattern sensitivity study
#   5) Stair pattern sensitivity study
#   6) Pareto frontier analysis (uses its own multi-TP/PP configs)
#   7) Fidelity test
#
# Usage:
#   ./run_all.sh --model <name> --tp <size> --pp <size> --dataset-path <path>
#   ./run_all.sh --skip-profiling --model <name> --tp <size> --pp <size> --dataset-path <path>
#
# Example:
#   ./run_all.sh --model Qwen/Qwen2.5-14B --tp 1 --pp 4 \
#       --dataset-path /workspace/benchmarks/energy/datasets/requests_lang_m-small_day1_19h00m-19h03m_200s_3rps.csv
# /workspace/benchmarks/energy/datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv

# Ensure HF_TOKEN is set in your environment before running this script.
# e.g.: export HF_TOKEN=<your_huggingface_token>
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please export your Hugging Face token first:"
    echo "  export HF_TOKEN=<your_huggingface_token>"
    exit 1
fi

SCRIPT_DIR="$(dirname "$0")"

# Defaults (matching individual script defaults)
MODEL="Qwen/Qwen2.5-14B"
TP=1
PP=4
DATASET_PATH=""
SKIP_PROFILING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --skip-profiling) SKIP_PROFILING=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-profiling] --model <name> --tp <size> --pp <size> --dataset-path <path>"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET_PATH" ]; then
    echo "Error: --dataset-path is required"
    echo "Usage: $0 --model <name> --tp <size> --pp <size> --dataset-path <path>"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

echo "=============================================="
echo "Artifact Evaluation - Full Pipeline"
echo "=============================================="
echo "Model:        $MODEL"
echo "TP x PP:      ${TP} x ${PP}"
echo "Dataset:      $DATASET_PATH"
echo "Skip Profile: $SKIP_PROFILING"
echo "=============================================="

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

if [[ "$SKIP_PROFILING" == "true" ]]; then
    echo ""
    echo "====== Skipping profiling steps (--skip-profiling) ======"
    # Verify that pre-computed profiles exist
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    MODEL_CLEANED=$(echo "$MODEL" | sed 's/\//_/g')
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
    SYSTEM_CSV="$SCRIPT_DIR/offline_profile_results/dvfs_profile_${GPU_NAME}_${MODEL_CLEANED}_tp${TP}_pp${PP}_one.csv"
    DYNAMO_CSV="$SCRIPT_DIR/dynamollm_profiles/dynamo_dvfs_profile_${GPU_NAME}_${MODEL_SHORT}.csv"
    if [[ ! -f "$SYSTEM_CSV" ]]; then
        echo "WARNING: System profile not found: $SYSTEM_CSV"
        echo "  S1/S2 scheduling may not work. Run without --skip-profiling to generate it."
    fi
    if [[ ! -f "$DYNAMO_CSV" ]]; then
        echo "WARNING: DynamoLLM profile not found: $DYNAMO_CSV"
        echo "  DynamoLLM baseline may not work. Run without --skip-profiling to generate it."
    fi
else
    run_step "Offline Profiling (System Profiles)" \
        bash "$SCRIPT_DIR/run_offline_profile.sh" \
            --model "$MODEL" --tp "$TP" --pp "$PP"

    run_step "Offline Profiling (DynamoLLM Profiles)" \
        bash "$SCRIPT_DIR/run_dynamo_benchmark.sh" \
            --model "$MODEL" --tp "$TP" --pp "$PP"
fi

run_step "End-to-End Evaluation" \
    bash "$SCRIPT_DIR/run_e2e.sh" \
        --model "$MODEL" --tp "$TP" --pp "$PP" \
        --dataset-path "$DATASET_PATH"

run_step "Ablation Study" \
    bash "$SCRIPT_DIR/run_ablation.sh" \
        --model "$MODEL" --tp "$TP" --pp "$PP"

run_step "Burst Pattern Study" \
    bash "$SCRIPT_DIR/run_burst.sh" \
        --model "$MODEL" --tp "$TP" --pp "$PP"

run_step "Stair Pattern Study" \
    bash "$SCRIPT_DIR/run_stair.sh" \
        --model "$MODEL" --tp "$TP" --pp "$PP"

run_step "Pareto Frontier Analysis" \
    bash "$SCRIPT_DIR/run_pareto.sh" \
        --model "$MODEL"

run_step "Fidelity Test" \
    bash "$SCRIPT_DIR/run_fidelity.sh" \
        --model "$MODEL" --tp "$TP" --pp "$PP" \
        --dataset-path "$DATASET_PATH"

echo ""
echo "=============================================="
echo "All artifact evaluation steps complete!"
echo "=============================================="
