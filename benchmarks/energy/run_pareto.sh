#!/bin/bash

# Comprehensive dataset test script for traced datasets
# Usage: ./run_e2e.sh --dataset-path path/to/trace.csv
# 
# Uses actual timestamps and request counts from CSV file
# Tests 3 configurations: 
#   1) Fixed chunk 2048 + System-level GPU clock locking (1830 MHz) + TBT SLO 2.0s
#   2) DynamoLLM Energy-Aware Scheduling with explicit SLO tuples (TPOT_MEAN:TPOT_P90:TTFT_P90 in seconds)
#   3) Dynamic energy budget with explicit SLO tuples (same as dynamo) + TBT SLO derived from TPOT_MEAN

model_name="meta-llama/Llama-3.3-70B-Instruct"

source "$(dirname "$0")/utils.sh"

# Default parallelism configuration (can be overridden)
parallelism_configs=(
    "4:2"
    "2:4"
)

REQUEST_RATE=1
RANDOM_INPUT_LEN=1024
RANDOM_OUTPUT_LEN=64
RANDOM_RANGE_RATIO=0.4
BURSTINESS=1
NUM_PROMPTS=60



# SLO targets as tuples: (TBT, TTFT)
# Each tuple specifies the three SLO targets in seconds for DynamoLLM scheduling
# For Pareto analysis: varying TBT SLO to create energy/performance trade-off curve
    # "0.1:0.2:5.0"
    # "0.1:1.0"     # Very strict÷
SLO_TUPLES=(
    "0.06:1.0"     # Very strict
    "0.10:1.0"     # Relaxed
    "0.14:1.0"     # Very relaxed
    "0.18:1.0"     # Most relaxed
    "0.22:1.0"     # Most relaxed
)


# Automatically derive SLO names from SLO_TUPLES array
SLO_NAMES=()
for i in "${!SLO_TUPLES[@]}"; do
    # Convert index to letter (A, B, C, ...)
    letter=$(printf "\\$(printf '%03o' $((65 + i)))")
    SLO_NAMES+=("SLO_${letter}")
done

FIXED_CHUNK_SIZE=2048

# Parse command line arguments
_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) _MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
[[ -n "$_MODEL" ]] && model_name="$_MODEL"

# Using random dataset for Pareto analysis

echo "Using random dataset for Pareto analysis: input_len=${RANDOM_INPUT_LEN}, output_len=${RANDOM_OUTPUT_LEN}"

BENCHMARK_EXTRA_ARGS="--dataset-name random --random-input-len ${RANDOM_INPUT_LEN} --random-output-len ${RANDOM_OUTPUT_LEN} --random-range-ratio ${RANDOM_RANGE_RATIO} --num-prompts ${NUM_PROMPTS} --burstiness ${BURSTINESS} --request-rate ${REQUEST_RATE} --profile --seed 42"

run_benchmark_config_all_slos() {
    local base_dir="$1"
    local config_num_pp="$2"
    
    echo "Running benchmarks for all SLO configurations..."
    echo "Total SLO configurations: ${#SLO_TUPLES[@]}"
    
    # Iterate over all SLO tuples
    for i in "${!SLO_TUPLES[@]}"; do
        local slo_tuple="${SLO_TUPLES[$i]}"
        local slo_name="${SLO_NAMES[$i]}"
        
        echo "\n===== Running benchmark for ${slo_name}: ${slo_tuple} ====="
        
        # Parse SLO tuple: "TBT:TTFT"
        IFS=':' read -r tbt_slo ttft_slo <<< "$slo_tuple"
        
        # Create result directory for this SLO configuration
        local config_dir="${base_dir}/${slo_name}_${slo_tuple//:/_}"
        mkdir -p "$config_dir"
        
        # Create result directory for traced dataset
        local result_dir="${config_dir}/traced_dataset"
        mkdir -p "$result_dir"
        
        echo "SLO Configuration: TBT=${tbt_slo}s, TTFT=${ttft_slo}s"
        echo "Result directory: $result_dir"
        
        # Start GPU energy monitor
        echo "Starting GPU energy monitor for ${slo_name}..."
        python3 gpu_energy_monitor.py "$result_dir" &
        local logger_pid=$!
        
        # Run the benchmark with DynamoLLM scheduling
        echo "Running DynamoLLM benchmark with SLO tuple: ${slo_tuple}"
        run_benchmark_config "Dynamic" "$result_dir" "" "$tbt_slo" "$tbt_slo" "$ttft_slo" "" "$config_num_pp"
        
        # Stop energy monitor
        echo "Stopping GPU energy monitor for ${slo_name}..."
        kill -TERM $logger_pid 2>/dev/null
        wait $logger_pid 2>/dev/null
        
        # Save SLO configuration info
        echo "SLO Configuration: ${slo_tuple}" > "$config_dir/slo_info.txt"
        echo "TBT: ${tbt_slo}s" >> "$config_dir/slo_info.txt"
        echo "TTFT: ${ttft_slo}s" >> "$config_dir/slo_info.txt"
        echo "Test completed: $(date)" >> "$config_dir/slo_info.txt"
        
        echo "Completed benchmark for ${slo_name}"
    done
    
    echo "\n===== All SLO benchmarks completed ====="
}

# Configuration B: vLLM with --use-s1 and --use-s2 for different TP x PP configurations
run_beam_test() {
    local base_dir="$1"
    local chunk_size=$FIXED_CHUNK_SIZE
    
    echo "\n===== Running vLLM with --use-s1 and --use-s2 for different TP x PP configurations ====="
    echo "Testing parallelism configurations: ${parallelism_configs[*]}"
    
    # Iterate over all parallelism configurations
    for config in "${parallelism_configs[@]}"; do
        # Parse TP:PP from config
        IFS=':' read -r num_tp num_pp <<< "$config"
        
        echo "\n--- Testing TP=${num_tp}, PP=${num_pp} ---"
        
        # Create config directory for this parallelism configuration
        local config_dir="${base_dir}/s1_s2_tp${num_tp}_pp${num_pp}_chunk_${FIXED_CHUNK_SIZE}"
        mkdir -p "$config_dir"
        
        echo "Configuration: TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
        
        # Build server command with --use-s1 and --use-s2
        local server_cmd="HF_HOME=/workspace/.cache/huggingface/ \
            VLLM_TORCH_PROFILER_DIR=${config_dir} \
            VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000 \
            VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
            VLLM_ATTENTION_BACKEND=FLASHINFER \
            vllm serve ${model_name} \
            --dtype auto \
            --kv-cache-dtype auto \
            -pp ${num_pp} \
            -tp ${num_tp} \
            --distributed-executor-backend ray \
            --custom-profiler \
            --no-enable-prefix-caching \
            --max-model-len 10000 \
            --max-num-seqs 256 \
            --max-num-batched-tokens 10000 \
            --gpu-memory-utilization 0.9 \
            --use-s1 \
            --use-s2 \
            > \"$config_dir/server.log\" 2>&1 &"
        
        # Execute the server command
        eval "$server_cmd"
        local server_pid=$!
        
        wait_for_server_ready
        
        # Run all SLO configurations for this parallelism setting
        run_benchmark_config_all_slos "$config_dir" "$num_pp"
        
        echo "TP=${num_tp}, PP=${num_pp} run done. Killing server..."
        kill -TERM $server_pid 2>/dev/null
        wait $server_pid 2>/dev/null
        
        # Save configuration info
        echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" > "$config_dir/parallelism_info.txt"
        echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/parallelism_info.txt"
        echo "Test completed: $(date)" >> "$config_dir/parallelism_info.txt"
        
        echo "Completed benchmark for TP=${num_tp}, PP=${num_pp}"
    done
    
    echo "\n===== All parallelism configurations completed ====="
}

# Main function
main() {
    # Make directory w.r.t current date
    local current_date=$(date +%Y%m%d_%H%M%S)
    local base_dir="sensitivity_tests/pareto/pareto_analysis_${current_date}"
    mkdir -p "$base_dir"
    
    run_beam_test "$base_dir"
}

main