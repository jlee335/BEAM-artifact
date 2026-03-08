#!/bin/bash

# Fidelity Test - Tests S1 + S2 (our approach)
# Uses single parallelism config and single SLO for controlled comparison

model_name="meta-llama/Llama-3.3-70B-Instruct"

source "$(dirname "$0")/utils.sh"

# Default parallelism configuration (can be overridden)
parallelism_configs=(
    "2:4"
)

REQUEST_RATE=8
RANDOM_INPUT_LEN=300
RANDOM_OUTPUT_LEN=100
RANDOM_RANGE_RATIO=0.8
BURSTINESS=1
NUM_PROMPTS=400


# SLO targets as tuples: (TBT, TTFT)
SLO_TUPLES=(
    "0.2:1.0"
)

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

echo "Using random dataset: input_len=${RANDOM_INPUT_LEN}, output_len=${RANDOM_OUTPUT_LEN}"

BENCHMARK_EXTRA_ARGS="--dataset-name random --random-input-len ${RANDOM_INPUT_LEN} --random-output-len ${RANDOM_OUTPUT_LEN} --random-range-ratio ${RANDOM_RANGE_RATIO} --num-prompts ${NUM_PROMPTS} --burstiness ${BURSTINESS} --request-rate ${REQUEST_RATE} --profile --seed 42"
EXPERIMENT_INFO="Mode: Random Dataset"

VLLM_EXTRA_ARGS="--measure-fidelity"

# Main function
main() {
    # Make directory w.r.t current date
    local current_date=$(date +%Y%m%d_%H%M%S)
    
    local base_dir="sensitivity_tests/fidelity/fidelity_test_${current_date}"
    mkdir -p "$base_dir"
    
    echo "====================================="
    echo "Starting Fidelity Test"
    echo "====================================="
    echo "Model: ${model_name}"
    echo "Parallelism Config: ${parallelism_configs[0]}"
    echo "SLO Configuration: ${SLO_TUPLES[0]}"
    echo "Mode: Random Dataset"
    echo "Request Rate: ${REQUEST_RATE}"
    echo "Input Length: ${RANDOM_INPUT_LEN}"
    echo "Output Length: ${RANDOM_OUTPUT_LEN}"
    echo "Number of Prompts: ${NUM_PROMPTS}"
    echo "Base Directory: ${base_dir}"
    echo "====================================="
    
    run_experiment_d "$base_dir"
    
    echo "\n====================================="
    echo "Fidelity Test Complete!"
    echo "====================================="
    echo "Results saved in: ${base_dir}"
    
    # Create summary file
    echo "Fidelity Test Summary" > "$base_dir/fidelity_summary.txt"
    echo "=====================" >> "$base_dir/fidelity_summary.txt"
    echo "" >> "$base_dir/fidelity_summary.txt"
    echo "Model: ${model_name}" >> "$base_dir/fidelity_summary.txt"
    echo "Parallelism Config: ${parallelism_configs[0]}" >> "$base_dir/fidelity_summary.txt"
    echo "SLO Configuration: ${SLO_TUPLES[0]}" >> "$base_dir/fidelity_summary.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$base_dir/fidelity_summary.txt"
    
    echo "Mode: Random Dataset" >> "$base_dir/fidelity_summary.txt"
    echo "Request Rate: ${REQUEST_RATE}" >> "$base_dir/fidelity_summary.txt"
    echo "Input Length: ${RANDOM_INPUT_LEN}" >> "$base_dir/fidelity_summary.txt"
    echo "Output Length: ${RANDOM_OUTPUT_LEN}" >> "$base_dir/fidelity_summary.txt"
    echo "Number of Prompts: ${NUM_PROMPTS}" >> "$base_dir/fidelity_summary.txt"
    
    echo "" >> "$base_dir/fidelity_summary.txt"
    echo "Test: S1 + S2 (our approach)" >> "$base_dir/fidelity_summary.txt"
    echo "" >> "$base_dir/fidelity_summary.txt"
    echo "Completed: $(date)" >> "$base_dir/fidelity_summary.txt"
    
}

main