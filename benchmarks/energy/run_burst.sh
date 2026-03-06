#!/bin/bash

# Burst Pattern Study - Tests system behavior under LLHLL request rate pattern
# (Low-Low-High-Low-Low) across the following configurations:
# A) Vanilla vLLM (no S1, no S2)
# B2) DynamoLLM-style Scheduling (window-based DVFS only)
# C) S1-only
# D) S1 + S2 (ours)
#
# Uses single parallelism config and single SLO for controlled comparison
# Request pattern: L-L-H-L-L (5 phases with 1 burst in the middle)

# model_name="meta-llama/Llama-3.3-70B-Instruct"
# model_name="RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
# model_name="Qwen/Qwen2.5-14B"
model_name="Qwen/Qwen2.5-32B"
# model_name="ByteResearch/Llama-3-8B-Instruct"

source "$(dirname "$0")/utils.sh"

# Default parallelism configuration (can be overridden)
parallelism_configs=(
    "1:4"
)

# Burst experiment configuration: LLHLL pattern
REQUEST_RATE_HIGH=20
REQUEST_RATE_LOW=4
REQUEST_RATE=1
RANDOM_INPUT_LEN=256
RANDOM_OUTPUT_LEN=64
RANDOM_RANGE_RATIO=0.5
BURSTINESS=1
NUM_PROMPTS=1000

# SLO targets as tuples: (TBT, TTFT)
# Single SLO for ablation study
SLO_TUPLES=(
    "0.2:1.0"     # Single SLO for ablation
)

# Automatically derive SLO names from SLO_TUPLES array
SLO_NAMES=()
for i in "${!SLO_TUPLES[@]}"; do
    # Convert index to letter (A, B, C, ...)
    letter=$(printf "\\$(printf '%03o' $((65 + i)))")
    SLO_NAMES+=("SLO_${letter}")
done

# Fixed clock frequency for experiment B
FIXED_CLOCK_FREQ=1230

FIXED_CHUNK_SIZE=2048

# Parse command line arguments
_MODEL="" _TP="" _PP=""
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Using random dataset for ablation study

echo "Using random dataset for ablation study: input_len=${RANDOM_INPUT_LEN}, output_len=${RANDOM_OUTPUT_LEN}"

BENCHMARK_EXTRA_ARGS="--dataset-name random --random-input-len ${RANDOM_INPUT_LEN} --random-output-len ${RANDOM_OUTPUT_LEN} --random-range-ratio ${RANDOM_RANGE_RATIO} --num-prompts ${NUM_PROMPTS} --burstiness ${BURSTINESS} --request-rate ${REQUEST_RATE} --request-rates ${REQUEST_RATE_LOW} ${REQUEST_RATE_LOW} ${REQUEST_RATE_HIGH} ${REQUEST_RATE_LOW} ${REQUEST_RATE_LOW} --profile --seed 42"

EXPERIMENT_INFO="Study Type: Burst Pattern (LLHLL)\nRequest Pattern: L-L-H-L-L\nLow Request Rate: ${REQUEST_RATE_LOW} req/s\nHigh Request Rate: ${REQUEST_RATE_HIGH} req/s"

# Main function
main() {
    # Make directory w.r.t current date
    local current_date=$(date +%Y%m%d_%H%M%S)
    local base_dir="sensitivity_tests/burst/burst_study_${current_date}"
    mkdir -p "$base_dir"
    
    echo "====================================="
    echo "Starting Burst Pattern Study (LLHLL)"
    echo "====================================="
    echo "Model: ${model_name}"
    echo "Parallelism Config: ${parallelism_configs[0]}"
    echo "SLO Configuration: ${SLO_TUPLES[0]}"
    echo "Burst Pattern: L-L-H-L-L"
    echo "Low Request Rate: ${REQUEST_RATE_LOW} req/s"
    echo "High Request Rate: ${REQUEST_RATE_HIGH} req/s"
    echo "Input Length: ${RANDOM_INPUT_LEN}"
    echo "Output Length: ${RANDOM_OUTPUT_LEN}"
    echo "Number of Prompts: ${NUM_PROMPTS}"
    echo "Base Directory: ${base_dir}"
    echo "====================================="
    
    # Run all five experiments
    run_experiment_a "$base_dir" # vLLM vanilla
    run_experiment_b2 "$base_dir" # DynamoLLM-style DVFS only
    run_experiment_d "$base_dir" # S1 + S2 (ours)
    
    echo "\n====================================="
    echo "Burst Pattern Study Complete!"
    echo "====================================="
    echo "Results saved in: ${base_dir}"
    
    # Create summary file
    echo "Burst Pattern Study Summary (LLHLL)" > "$base_dir/burst_summary.txt"
    echo "=====================================" >> "$base_dir/burst_summary.txt"
    echo "" >> "$base_dir/burst_summary.txt"
    echo "Model: ${model_name}" >> "$base_dir/burst_summary.txt"
    echo "Parallelism Config: ${parallelism_configs[0]}" >> "$base_dir/burst_summary.txt"
    echo "SLO Configuration: ${SLO_TUPLES[0]}" >> "$base_dir/burst_summary.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$base_dir/burst_summary.txt"
    echo "Burst Pattern: L-L-H-L-L" >> "$base_dir/burst_summary.txt"
    echo "Low Request Rate: ${REQUEST_RATE_LOW} req/s" >> "$base_dir/burst_summary.txt"
    echo "High Request Rate: ${REQUEST_RATE_HIGH} req/s" >> "$base_dir/burst_summary.txt"
    echo "Input Length: ${RANDOM_INPUT_LEN}" >> "$base_dir/burst_summary.txt"
    echo "Output Length: ${RANDOM_OUTPUT_LEN}" >> "$base_dir/burst_summary.txt"
    echo "Number of Prompts: ${NUM_PROMPTS}" >> "$base_dir/burst_summary.txt"
    echo "" >> "$base_dir/burst_summary.txt"
    echo "Experiments:" >> "$base_dir/burst_summary.txt"
    echo "  A) Vanilla vLLM" >> "$base_dir/burst_summary.txt"
    echo "  B2) DynamoLLM-style Scheduling (window-based DVFS only)" >> "$base_dir/burst_summary.txt"
    echo "  C) S1-only" >> "$base_dir/burst_summary.txt"
    echo "  D) S1 + S2 (ours)" >> "$base_dir/burst_summary.txt"
    echo "" >> "$base_dir/burst_summary.txt"
    echo "Completed: $(date)" >> "$base_dir/burst_summary.txt"
}

main