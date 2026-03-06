#!/bin/bash

# Ablation Study - Tests the following configurations:
# A) Vanilla vLLM (no S1, no S2)
# B2) DynamoLLM-style Scheduling (window-based DVFS only)
# D) S1 + S2 (ours)
#
# Uses single parallelism config and single SLO for controlled comparison

model_name="Qwen/Qwen2.5-14B"

SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}"

source "$(dirname "$0")/utils.sh"

# Default parallelism configuration (can be overridden)
parallelism_configs=(
    "1:4"
)

# SLO targets as tuples: (TBT, TTFT)
# Single SLO for ablation study
SLO_TUPLES=(
    "0.4:1.0"     # Single SLO for ablation
)

FIXED_CHUNK_SIZE=2048

# Parse command line arguments
DATASET_PATH=""
_MODEL="" _TP="" _PP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --model) _MODEL="$2"; shift 2 ;;
        --tp) _TP="$2"; shift 2 ;;
        --pp) _PP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
[[ -n "$_MODEL" ]] && model_name="$_MODEL"
if [[ -n "$_TP" || -n "$_PP" ]]; then
    _TP="${_TP:-${parallelism_configs[0]%%:*}}"
    _PP="${_PP:-${parallelism_configs[0]##*:}}"
    parallelism_configs=("${_TP}:${_PP}")
fi

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: --dataset-path argument is required"
    echo "Usage: $0 --dataset-path <path_to_dataset.csv>"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

echo "Using traced dataset: $DATASET_PATH"

BENCHMARK_EXTRA_ARGS="--dataset-name traced --dataset-path ${DATASET_PATH}"
VLLM_EXTRA_ARGS=""
EXPERIMENT_INFO=""

# Main function
main() {
    # Make directory w.r.t current date
    local current_date=$(date +%Y%m%d_%H%M%S)
    local dataset_basename=$(basename "$DATASET_PATH" .csv)
    local base_dir="end_to_end/${dataset_basename}_${current_date}"
    mkdir -p "$base_dir"
    
    echo "====================================="
    echo "Starting End-to-End Ablation Study"
    echo "====================================="
    echo "Model: ${model_name}"
    echo "Parallelism Config: ${parallelism_configs[0]}"
    echo "SLO Configuration: ${SLO_TUPLES[0]}"
    echo "Dataset: ${DATASET_PATH}"
    echo "Base Directory: ${base_dir}"
    echo "====================================="
    
    # Run all experiments
    run_experiment_a "$base_dir"
    run_experiment_b2 "$base_dir"
    run_experiment_d "$base_dir"
    
    echo "\n====================================="
    echo "End-to-End Ablation Study Complete!"
    echo "====================================="
    echo "Results saved in: ${base_dir}"
    
    # Create summary file
    echo "End-to-End Ablation Study Summary" > "$base_dir/ablation_summary.txt"
    echo "==================================" >> "$base_dir/ablation_summary.txt"
    echo "" >> "$base_dir/ablation_summary.txt"
    echo "Model: ${model_name}" >> "$base_dir/ablation_summary.txt"
    echo "Dataset: ${DATASET_PATH}" >> "$base_dir/ablation_summary.txt"
    echo "Parallelism Config: ${parallelism_configs[0]}" >> "$base_dir/ablation_summary.txt"
    echo "SLO Configuration: ${SLO_TUPLES[0]}" >> "$base_dir/ablation_summary.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$base_dir/ablation_summary.txt"
    echo "" >> "$base_dir/ablation_summary.txt"
    echo "Experiments:" >> "$base_dir/ablation_summary.txt"
    echo "  A) Vanilla vLLM" >> "$base_dir/ablation_summary.txt"
    echo "  B2) DynamoLLM-style Scheduling (window-based DVFS only)" >> "$base_dir/ablation_summary.txt"
    echo "  D) S1 + S2 (ours)" >> "$base_dir/ablation_summary.txt"
    echo "" >> "$base_dir/ablation_summary.txt"
    echo "Completed: $(date)" >> "$base_dir/ablation_summary.txt"
    
    # Slack notification
    if command -v curl >/dev/null 2>&1; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"✅ End-to-End Ablation Study Completed for ${model_name}\\nDataset: ${DATASET_PATH}\\nExperiments: A) Vanilla, B2) DynamoLLM, D) S1+S2\\nResults: ${base_dir}\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
}

main