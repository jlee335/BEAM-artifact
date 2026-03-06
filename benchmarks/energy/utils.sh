# PATHs
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
export NSYS=/opt/nvidia/nsight-systems/2024.5.1/bin/
export PATH=$PATH:$NSYS
export HF_HOME=/workspace/.cache/huggingface/
set -m

# Check for required tools
if ! command -v bc &> /dev/null; then
    echo "Error: bc (calculator) is required for SLO calculations but not installed"
    exit 1
fi

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Common utility variables
model_name_cleaned=$(echo $model_name | sed 's/\//-/g')

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
    echo "Running benchmark: chunk_size=${chunk_size}, tpot_slo=${tpot_slo}, tbt_slo=${tbt_slo}, ttft_slo=${ttft_slo}"

    # If chunk_size is Dynamic, put placeholder 2048 (default) unless overridden
    if [ "$chunk_size" == "Dynamic" ]; then
        chunk_size=${FIXED_CHUNK_SIZE:-2048}
    fi
    
    # Use provided num_pp or default to microbatch_size
    local mb_size="${config_num_pp:-${microbatch_size:-4}}"

    # Build base benchmark command
    local benchmark_cmd="PYTHONPATH=\"/workspace/disagg/energy-inf-v1-disagg/benchmarks:$PYTHONPATH\" python3 benchmark_script.py \
        --backend vllm \
        --model ${model_name} \
        --chunk-size ${chunk_size} \
        --microbatch-size ${mb_size} \
        --tpot-slo ${tpot_slo} \
        --tbt-slo ${tbt_slo} \
        --ttft-slo ${ttft_slo} \
        --result-dir ${result_dir} \
        ${BENCHMARK_EXTRA_ARGS} \
        --save-detailed \
        --save-result \
        --profile"

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
    local chunk_size="${4:-Dynamic}"

    echo "Running benchmark for single SLO configuration..."

    # Use the single SLO tuple
    local slo_tuple="${SLO_TUPLES[0]}"

    echo -e "\n===== Running benchmark for SLO: ${slo_tuple} ====="

    # Parse SLO tuple: "TBT:TTFT"
    IFS=':' read -r tbt_slo ttft_slo <<< "$slo_tuple"

    mkdir -p "$result_dir"

    echo "SLO Configuration: TBT=${tbt_slo}s, TTFT=${ttft_slo}s"
    echo "Result directory: $result_dir"

    # Start GPU energy monitor
    echo "Starting GPU energy monitor..."
    python3 gpu_energy_monitor.py "$result_dir" &
    local logger_pid=$!

    echo "Running DynamoLLM benchmark with SLO tuple: ${slo_tuple}"
    run_benchmark_config "${chunk_size}" "$result_dir" "" "$tbt_slo" "$tbt_slo" "$ttft_slo" "" "$config_num_pp"
    
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

# Generic experiment template
# Arguments:
# 1. base_dir
# 2. experiment_id (e.g., "A_vanilla", "B2_dynamollm_dvfs", "D_s1_s2")
# 3. experiment_name
# 4. extra vllm args string
# 5. chunk_size override (optional, default: "Dynamic")
run_experiment_template() {
    local base_dir="$1"
    local exp_id="$2"
    local exp_name="$3"
    local extra_vllm_args="$4"
    local chunk_size="${5:-Dynamic}"
    
    echo -e "\n===== Experiment: ${exp_name} ====="
    
    # Parse TP:PP from config
    IFS=':' read -r num_tp num_pp <<< "${parallelism_configs[0]}"
    
    # Create config directory
    local config_dir="${base_dir}/${exp_id}_tp${num_tp}_pp${num_pp}"
    mkdir -p "$config_dir"
    
    # Create traced_dataset subdirectory for benchmark results
    local result_dir="${config_dir}/traced_dataset"
    mkdir -p "$result_dir"
    
    echo "Configuration: ${exp_name}, TP=${num_tp}, PP=${num_pp}, Chunk=${FIXED_CHUNK_SIZE}"
    release_all_gpus_clock
    
    # Build server command
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
        ${extra_vllm_args} \
        > \"$config_dir/server.log\" 2>&1 &"
    
    eval "$server_cmd"
    local server_pid=$!
    
    wait_for_server_ready
    
    # Run benchmark (results go to traced_dataset/)
    run_single_benchmark "$result_dir" "$num_pp" "$config_dir" "$chunk_size"
    
    echo "${exp_name} done. Killing server..."
    kill -TERM $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    # Save configuration info
    echo "Experiment: ${exp_name}" > "$config_dir/experiment_info.txt"
    if [ -n "$EXPERIMENT_INFO" ]; then
        echo -e "${EXPERIMENT_INFO}" >> "$config_dir/experiment_info.txt"
    fi
    echo "Parallelism Configuration: TP=${num_tp}, PP=${num_pp}" >> "$config_dir/experiment_info.txt"
    echo "Chunk Size: ${FIXED_CHUNK_SIZE}" >> "$config_dir/experiment_info.txt"
    echo "Test completed: $(date)" >> "$config_dir/experiment_info.txt"
    
    echo "Completed ${exp_name}"
}

run_experiment_a() {
    run_experiment_template "$1" "A_vanilla_vllm" "Experiment A - Vanilla vLLM" "--max-num-batched-tokens 256 ${VLLM_EXTRA_ARGS}" 256
}

run_experiment_b2() {
    run_experiment_template "$1" "B2_dynamollm_dvfs" "Experiment B2 - DynamoLLM-style Scheduling" "--use-window-based-dvfs-only --max-num-batched-tokens 256 ${VLLM_EXTRA_ARGS}" 256
}

run_experiment_d() {
    run_experiment_template "$1" "D_s1_s2" "Experiment D - S1 + S2 (ours)" "--use-s1 --use-s2 ${VLLM_EXTRA_ARGS}"
}
