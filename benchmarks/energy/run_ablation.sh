#!/bin/bash

# Ablation Study - Tests the following configurations:
# A) Vanilla vLLM (no S1, no S2)
# B2) DynamoLLM-style Scheduling (window-based DVFS only)
# B3) S1 with DVFS only (no chunk optimization)
# C) S1-only (chunk + DVFS)
# D) S1 + S2 (ours)
#
# Uses single parallelism config and single SLO for controlled comparison

# # PATHs
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
export NSYS=/opt/nvidia/nsight-systems/2024.5.1/bin/
export PATH=$PATH:$NSYS
export HF_HOME=/workspace/.cache/huggingface/
# export TORCH_CUDA_ARCH_LIST="8.6+PTX"
set -m

# Defaults (can be overridden via CLI args)
model_name="Qwen/Qwen2.5-14B"
TP=1
PP=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) model_name="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [--tp <size>] [--pp <size>]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

model_name_cleaned=$(echo $model_name | sed 's/\//-/g')

# Check for required tools
if ! command -v bc &> /dev/null; then
    echo "Error: bc (calculator) is required for SLO calculations but not installed"
    exit 1
fi

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Parallelism configuration from args
parallelism_configs=(
    "${TP}:${PP}"
)

REQUEST_RATE=4
RANDOM_INPUT_LEN=512
RANDOM_OUTPUT_LEN=64
RANDOM_RANGE_RATIO=0.5
BURSTINESS=1
NUM_PROMPTS=180


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

# Using random dataset for ablation study

echo "Using random dataset for ablation study: input_len=${RANDOM_INPUT_LEN}, output_len=${RANDOM_OUTPUT_LEN}"

# GPU clock management functions
lock_all_gpus_clock() {
    local freq=$1
    echo "Locking all GPU clocks to ${freq} MHz..."
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        nvidia-smi -i $gpu -pm 1
        nvidia-smi -i $gpu -lgc ${freq}
    done
    echo "GPU clocks locked to ${freq} MHz"
}

release_all_gpus_clock() {
    echo "Releasing GPU clock locks..."
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        nvidia-smi -i $gpu -rgc
        nvidia-smi -i $gpu -pm 0
    done
    echo "GPU clock locks released"
}

# Safety cleanup function - ensures GPU clocks are released on script exit
cleanup_on_exit() {
    echo "Script exiting - ensuring GPU clock locks are released..."
    release_all_gpus_clock 2>/dev/null || true
}

# Set trap to release clocks on script exit
trap cleanup_on_exit EXIT


# Server management functions
wait_for_server_ready() {
    local max_retries=200
    local retry=0
    local url="http://localhost:8000/v1/models"
    echo "Waiting for vllm server to be ready..."
    until curl --silent --fail "$url" > /dev/null; do
        ((retry++))
        if [ $retry -ge $max_retries ]; then
            echo "Server did not become ready in time. Exiting."
            exit 1
        fi
        sleep 10
    done
    echo "Server is ready!"
}

run_benchmark_config() {
    local chunk_size="$1"
    local result_dir="$2"
    local dataset_path="$3"
    local tpot_slo="$4"
    local tbt_slo="$5"
    local ttft_slo="$6"
    local microbatch_size="$7"
    local config_num_pp="$8"
    echo "Running benchmark: chunk_size=${chunk_size}, tpot_slo=${tpot_slo}, tbt_slo=${tbt_slo}, ttft_slo=${ttft_slo}, dataset=${dataset_path}"

    # If chunk_size is Dynamic, put placeholder 1024
    if [ "$chunk_size" == "Dynamic" ]; then
        chunk_size=2048
    fi
    
    # Use provided num_pp or default to microbatch_size
    local mb_size="${config_num_pp:-${microbatch_size:-4}}"

    # Build base benchmark command
    local benchmark_cmd="PYTHONPATH=\"/workspace/disagg/energy-inf-v1-disagg/benchmarks:$PYTHONPATH\" python3 benchmark_script.py \
        --backend vllm \
        --model ${model_name} \
        --dataset-name random \
        --random-input-len ${RANDOM_INPUT_LEN} \
        --random-output-len ${RANDOM_OUTPUT_LEN} \
        --random-range-ratio ${RANDOM_RANGE_RATIO} \
        --num-prompts ${NUM_PROMPTS} \
        --burstiness ${BURSTINESS} \
        --request-rate ${REQUEST_RATE} \
        --profile \
        --seed 42 \
        --microbatch-size ${mb_size} \
        --tpot-slo ${tpot_slo} \
        --tbt-slo ${tbt_slo} \
        --ttft-slo ${ttft_slo} \
        --result-dir ${result_dir} \
        --save-detailed \
        --save-result"

    
    # Add specific SLO configurations for dynamo scheduling
    if [ -n "$tpot_slo" ] && [ -n "$tbt_slo" ] && [ -n "$ttft_slo" ]; then
        echo "Adding DynamoLLM SLO configuration: TTFT=${ttft_slo}s, TPOT=${tpot_slo}s, TBT=${tbt_slo}s"
        benchmark_cmd="${benchmark_cmd} --tpot-slo ${tpot_slo} --tbt-slo ${tbt_slo} --ttft-slo ${ttft_slo}"
        
        # Convert to milliseconds for goodput tracking (goodput tool may expect milliseconds)
        local ttft_slo_ms=$(echo "$ttft_slo * 1000" | bc)
        local tpot_slo_ms=$(echo "$tpot_slo * 1000" | bc)
        local tbt_slo_ms=$(echo "$tbt_slo * 1000" | bc)
        benchmark_cmd="${benchmark_cmd} --goodput ttft:${ttft_slo_ms} tpot:${tpot_slo_ms}"
    fi
    
    # Execute the benchmark command
    eval "$benchmark_cmd"
}

run_single_benchmark() {
    local result_dir="$1"
    local config_num_pp="$2"
    local config_dir="$3"
    
    echo "Running benchmark for single SLO configuration..."
    
    # Use the single SLO tuple
    local slo_tuple="${SLO_TUPLES[0]}"
    
    echo "\n===== Running benchmark for SLO: ${slo_tuple} ====="
    
    # Parse SLO tuple: "TBT:TTFT"
    IFS=':' read -r tbt_slo ttft_slo <<< "$slo_tuple"
    
    mkdir -p "$result_dir"
    
    echo "SLO Configuration: TBT=${tbt_slo}s, TTFT=${ttft_slo}s"
    echo "Result directory: $result_dir"
    
    # Start GPU energy monitor
    echo "Starting GPU energy monitor..."
    python3 gpu_energy_monitor.py "$result_dir" &
    local logger_pid=$!
    
    # Run the benchmark with DynamoLLM scheduling
    echo "Running DynamoLLM benchmark with SLO tuple: ${slo_tuple}"
    run_benchmark_config "Dynamic" "$result_dir" "" "$tbt_slo" "$tbt_slo" "$ttft_slo" "" "$config_num_pp"
    
    # Stop energy monitor
    echo "Stopping GPU energy monitor..."
    kill -TERM $logger_pid 2>/dev/null
    wait $logger_pid 2>/dev/null
    
    # Save SLO configuration info (at config_dir level, not in traced_dataset)
    if [ -n "$config_dir" ]; then
        echo "SLO Configuration: ${slo_tuple}" > "$config_dir/slo_info.txt"
        echo "TBT: ${tbt_slo}s" >> "$config_dir/slo_info.txt"
        echo "TTFT: ${ttft_slo}s" >> "$config_dir/slo_info.txt"
        echo "Test completed: $(date)" >> "$config_dir/slo_info.txt"
    fi
    
    echo "Completed benchmark"
}

# Experiment A: Vanilla vLLM (no S1, no S2)
run_experiment_a() {
    local base_dir="$1"
    
    echo "\n===== Experiment A: Vanilla vLLM ====="
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/A_vanilla_vllm_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: Vanilla vLLM, TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    release_all_gpus_clock
    # Build server command WITHOUT --use-s1 and --use-s2
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
        --max-num-batched-tokens 256 \
        --gpu-memory-utilization 0.9 \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir"
    
    echo "Experiment A done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Save configuration info (at config_dir level, not in traced_dataset)
    echo "Experiment: A - Vanilla vLLM" > "$config_dir/experiment_info.txt"
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed Experiment A"
}

# Experiment B: Vanilla vLLM with fixed clock (1230 MHz)
run_experiment_b() {
    local base_dir="$1"
    
    echo "\n===== Experiment B: Vanilla vLLM with Fixed Clock (${FIXED_CLOCK_FREQ} MHz) ====="
    
    # Lock GPU clocks
    lock_all_gpus_clock $FIXED_CLOCK_FREQ
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/B_vanilla_vllm_fixed_clock_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: Vanilla vLLM (Fixed Clock ${FIXED_CLOCK_FREQ} MHz), TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    
    # Build server command WITHOUT --use-s1 and --use-s2
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
        --max-num-batched-tokens 256 \
        --gpu-memory-utilization 0.9 \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir"
    
    echo "Experiment B done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Release GPU clocks
    release_all_gpus_clock
    
    # Save configuration info (at config_dir level, not in traced_dataset)
    echo "Experiment: B - Vanilla vLLM with Fixed Clock" > "$config_dir/experiment_info.txt"
    echo "Fixed Clock Frequency: ${FIXED_CLOCK_FREQ} MHz" >> "$config_dir/experiment_info.txt"
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed Experiment B"
}

# Experiment B2: DynamoLLM-style scheduling (window-based DVFS only)
run_experiment_b2() {
    local base_dir="$1"
    
    echo "\n===== Experiment B2: DynamoLLM-style Scheduling ====="
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/B2_dynamollm_dvfs_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: DynamoLLM-style DVFS, TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    release_all_gpus_clock
    
    # Build server command with --use-window-based-dvfs-only
    local server_cmd="HF_HOME=/workspace/.cache/huggingface/ \
        VLLM_TORCH_PROFILER_DIR=${config_dir} \
        VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000 \
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
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
        --max-num-batched-tokens 256 \
        --gpu-memory-utilization 0.9 \
        --use-window-based-dvfs-only \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir"
    
    echo "Experiment B2 done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Save configuration info (at config_dir level, not in traced_dataset)
    echo "Experiment: B2 - DynamoLLM-style Scheduling" > "$config_dir/experiment_info.txt"
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/experiment_info.txt"
    echo "Scheduling: Window-based DVFS only (DynamoLLM-like)" >> "$config_dir/experiment_info.txt"
    echo "DVFS Reschedule Interval: 5 seconds" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed Experiment B2"
}

# Experiment B3: S1 with DVFS only
run_experiment_b3() {
    local base_dir="$1"
    
    echo "\n===== Experiment B3: S1 with DVFS only ====="
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/B3_s1_dvfs_only_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: S1 with DVFS only, TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    release_all_gpus_clock
    
    # Build server command with --use-s1 and --use-s1-dvfs-only
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
        --max-num-batched-tokens 256 \
        --gpu-memory-utilization 0.9 \
        --use-s1 \
        --use-s1-dvfs-only \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir"
    
    echo "Experiment B3 done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Save configuration info (at config_dir level, not in traced_dataset)
    echo "Experiment: B3 - S1 with DVFS only" > "$config_dir/experiment_info.txt"
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: Fixed at 256 (DVFS only)" >> "$config_dir/experiment_info.txt"
    echo "Scheduling: S1 algorithm with DVFS optimization only" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed Experiment B3"
}

# Experiment C: S1-only
run_experiment_c() {
    local base_dir="$1"
    
    echo "\n===== Experiment C: S1-only ====="
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/C_s1_only_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: S1-only, TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    
    # Build server command with ONLY --use-s1
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
        --gpu-memory-utilization 0.9 \
        --use-s1 \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir"
    
    echo "Experiment C done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Save configuration info (at config_dir level, not in traced_dataset)
    echo "Experiment: C - S1-only" > "$config_dir/experiment_info.txt"
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed Experiment C"
}

# Experiment D: S1 + S2 (ours)
run_experiment_d() {
    local base_dir="$1"
    
    echo "\n===== Experiment D: S1 + S2 (ours) ====="
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/D_s1_s2_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: S1 + S2, TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    
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
        --gpu-memory-utilization 0.9 \
        --use-s1 \
        --use-s2 \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir"
    
    echo "Experiment D done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Save configuration info (at config_dir level, not in traced_dataset)
    echo "Experiment: D - S1 + S2 (ours)" > "$config_dir/experiment_info.txt"
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed Experiment D"
}

# Main function
main() {
    # Make directory w.r.t current date
    local current_date=$(date +%Y%m%d_%H%M%S)
    local base_dir="sensitivity_tests/ablation/ablation_study_${current_date}"
    mkdir -p "$base_dir"
    
    echo "====================================="
    echo "Starting Ablation Study"
    echo "====================================="
    echo "Model: ${model_name}"
    echo "Parallelism Config: ${parallelism_configs[0]}"
    echo "SLO Configuration: ${SLO_TUPLES[0]}"
    echo "Request Rate: ${REQUEST_RATE}"
    echo "Input Length: ${RANDOM_INPUT_LEN}"
    echo "Output Length: ${RANDOM_OUTPUT_LEN}"
    echo "Number of Prompts: ${NUM_PROMPTS}"
    echo "Base Directory: ${base_dir}"
    echo "====================================="
    
    # Run all experiments
    run_experiment_a "$base_dir"
    run_experiment_b3 "$base_dir"
    run_experiment_c "$base_dir"
    run_experiment_d "$base_dir"
    
    echo "\n====================================="
    echo "Ablation Study Complete!"
    echo "====================================="
    echo "Results saved in: ${base_dir}"
    
    # Create summary file
    echo "Ablation Study Summary" > "$base_dir/ablation_summary.txt"
    echo "======================" >> "$base_dir/ablation_summary.txt"
    echo "" >> "$base_dir/ablation_summary.txt"
    echo "Model: ${model_name}" >> "$base_dir/ablation_summary.txt"
    echo "Parallelism Config: ${parallelism_configs[0]}" >> "$base_dir/ablation_summary.txt"
    echo "SLO Configuration: ${SLO_TUPLES[0]}" >> "$base_dir/ablation_summary.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$base_dir/ablation_summary.txt"
    echo "Request Rate: ${REQUEST_RATE}" >> "$base_dir/ablation_summary.txt"
    echo "Input Length: ${RANDOM_INPUT_LEN}" >> "$base_dir/ablation_summary.txt"
    echo "Output Length: ${RANDOM_OUTPUT_LEN}" >> "$base_dir/ablation_summary.txt"
    echo "Number of Prompts: ${NUM_PROMPTS}" >> "$base_dir/ablation_summary.txt"
    echo "" >> "$base_dir/ablation_summary.txt"
    echo "Experiments:" >> "$base_dir/ablation_summary.txt"
    echo "  A) Vanilla vLLM" >> "$base_dir/ablation_summary.txt"
    echo "  B2) DynamoLLM-style Scheduling (window-based DVFS only)" >> "$base_dir/ablation_summary.txt"
    echo "  B3) S1 with DVFS only (no chunk optimization)" >> "$base_dir/ablation_summary.txt"
    echo "  C) S1-only (chunk + DVFS)" >> "$base_dir/ablation_summary.txt"
    echo "  D) S1 + S2 (ours)" >> "$base_dir/ablation_summary.txt"
    echo "" >> "$base_dir/ablation_summary.txt"
    echo "Completed: $(date)" >> "$base_dir/ablation_summary.txt"
}

main