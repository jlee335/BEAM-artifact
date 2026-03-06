#!/bin/bash

# Environment setup following project conventions
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
export HF_HOME=/workspace/.cache/huggingface/
# export TORCH_CUDA_ARCH_LIST="8.6+PTX"
# Add parent directory to Python path so benchmark_dataset can be imported
export PYTHONPATH="${PYTHONPATH}:/workspace/disagg/energy-inf-v1-disagg/benchmarks"
set -m

echo "🚀 Starting automated profiling based on DynamoLLM paper..."

# --- Configuration ---
# Set the model and other static parameters here
model_name="Qwen/Qwen2.5-32B"
# model_name="meta-llama/Llama-3.3-70B-Instruct"
python_script="benchmark_script.py"
python_exec="python3"
log_dir="dynamollm_profiles/dynamollm_profiling_logs_$(date +%Y%m%d_%H%M%S)"

# Get number of GPUs and set parallelism configurations
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Multiple parallelism configurations to test (format: tp:pp)
# parallelism_configs="4:2 2:4"
parallelism_configs="2:2 1:4"

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
    _TP="${_TP:-$(echo "$parallelism_configs" | cut -d: -f1 | awk '{print $1}')}"
    _PP="${_PP:-$(echo "$parallelism_configs" | cut -d: -f2 | awk '{print $1}')}"
    parallelism_configs="${_TP}:${_PP}"
fi

TIME_LIMIT=60

# Safety cleanup function - ensures GPU clocks are released and server is stopped on script exit
cleanup_on_exit() {
    echo "Script exiting - ensuring cleanup..."
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server with PID $SERVER_PID..."
        kill -TERM $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    # Note: Energy monitor cleanup happens in innermost loop per test case
    echo "Releasing GPU clock locks..."
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

# GPU clock management functions
lock_all_gpus_to_clock() {
    local clk=$1
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        echo "Locking GPU $gpu clock to $clk MHz..."
        nvidia-smi -i "$gpu" -lgc "$clk","$clk"
    done
}

release_all_gpus_clock() {
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        echo "Releasing clock lock for GPU $gpu..."
        nvidia-smi -i "$gpu" -rgc
    done
}

# Checks current GPU clocks to verify they are locked to the expected clock
check_all_gpu_clocks() {
    local expected_clk=$1
    local tolerance=50  # Allow 50 MHz tolerance for clock variations
    
    echo "Checking GPU clocks - expected: ${expected_clk} MHz (tolerance: ±${tolerance} MHz)"
    
    for ((gpu=0; gpu<$NUM_GPUS; gpu++)); do
        # Get current graphics clock for the GPU
        local current_clk=$(nvidia-smi -i "$gpu" --query-gpu=clocks.current.graphics --format=csv,noheader,nounits)
        
        # Check if the current clock is within tolerance of expected clock
        local lower_bound=$((expected_clk - tolerance))
        local upper_bound=$((expected_clk + tolerance))
        
        if [ "$current_clk" -ge "$lower_bound" ] && [ "$current_clk" -le "$upper_bound" ]; then
            echo "✅ GPU $gpu: ${current_clk} MHz (within tolerance of ${expected_clk} MHz)"
        else
            echo "❌ GPU $gpu: ${current_clk} MHz (expected ${expected_clk} MHz ±${tolerance} MHz)"
            return 1
        fi
    done
    
    echo "All GPUs are locked to the expected clock speed."
    return 0
}

# # --- Parameter Variations from Paper ---
# # [cite_start]GPU Frequencies in MHz [cite: 385]
# frequencies="800 1200 1600 1830"

# # [cite_start]System Loads in Tokens Per Second (TPS) [cite: 169]
# # These will be converted to request-rate by dividing by input_len
# tps_values="400 800 1600" # A6000

# # [cite_start]Representative Input Lengths for Short, Medium, Long [cite: 174]
# input_lens="128 512 4096"

# # [cite_start]Representative Output Lengths for Short, Medium, Long [cite: 174]
# output_lens="50 150"


# Token Category Analysis: - m-small 1-day
#   Input Tokens:
#     S (below 33rd): 43,124 requests (33.0%), mean: 45
#     M (33rd-66th): 42,834 requests (32.8%), mean: 527
#     L (above 66th): 44,787 requests (34.3%), mean: 1036
#   Output Tokens:
#     S (below 33rd): 43,031 requests (32.9%), mean: 4
#     M (33rd-66th): 13,411 requests (10.3%), mean: 36
#     L (above 66th): 74,303 requests (56.8%), mean: 147
# # --- Parameter Variations from Paper ---
# # [cite_start]GPU Frequencies in MHz [cite: 385]

frequencies="800 1000 1200 1400 1600 1800"
# # A100
# frequencies="600 800 1000 1200 1400"
# # [cite_start]System Loads in Tokens Per Second (TPS) [cite: 169]
# # These will be converted to request-rate by dividing by input_len
# tps_values="400 800 1200 1600" # A6000
tps_values="400 800 1200 1600 2000 2400" # A600≥0
# tps_values="600 1200 2400 3600 4800" # A6000÷


# For single - to - single comparison, use LL category only

# [cite_start]Representative Input Lengths for Short, Medium, Long [cite: 174]
input_lens="1050"

# [cite_start]Representative Output Lengths for Short, Medium, Long [cite: 174]
output_lens="150"




# --- Check if profile already exists ---
_gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
_model_name_clean="${model_name//\//_}"
_DYNAMO_DIR="$(dirname "$0")/dynamollm_profiles"
_csv_file="${_DYNAMO_DIR}/dynamo_dvfs_profile_${_gpu_name}_${_model_name_clean}.csv"

if [ -f "$_csv_file" ]; then
    echo "Profile already exists: $_csv_file"
    echo "Skipping profiling. Running visualization only."
    echo ""
    echo "====== Post-processing: Generating visualizations ======"
    (cd "$_DYNAMO_DIR" && python3 visualize.py "dynamo_dvfs_profile_${_gpu_name}_${_model_name_clean}.csv")
    exit 0
fi

# --- Execution Loop ---
# Create a directory to store the logs
mkdir -p $log_dir
echo "📝 Logs will be saved in the '$log_dir' directory."

# Calculate total number of runs for progress tracking
total_freqs=$(echo $frequencies | wc -w)
total_parallelism_configs=$(echo $parallelism_configs | wc -w)
total_tps=$(echo $tps_values | wc -w)
total_in_lens=$(echo $input_lens | wc -w)
total_out_lens=$(echo $output_lens | wc -w)
total_runs=$((total_freqs * total_parallelism_configs * total_tps * total_in_lens * total_out_lens))
current_run=0


# Nested loops to iterate through all parameter combinations
# Outermost loop: parallelism configurations
for config in $parallelism_configs; do
    # Parse tp:pp pair
    tp_config=$(echo $config | cut -d: -f1)
    pp_config=$(echo $config | cut -d: -f2)
    
    echo "========== Starting parallelism config: TP=${tp_config}, PP=${pp_config} =========="
    
    # Validate that pp_config * tp_config <= NUM_GPUS
    total_gpus_needed=$((pp_config * tp_config))
    if [ $total_gpus_needed -gt $NUM_GPUS ]; then
        echo "⚠️ Skipping TP=${tp_config}, PP=${pp_config}: requires ${total_gpus_needed} GPUs but only ${NUM_GPUS} available"
        continue
    fi
    
    # Start vLLM server for this parallelism configuration
    echo "Starting vLLM server for TP=${tp_config}, PP=${pp_config}..."
    server_config_dir="${log_dir}/tp${tp_config}_pp${pp_config}_server"
    mkdir -p "$server_config_dir"
    
    # Build server command with specific parallelism settings
    server_cmd="HF_HOME=/workspace/.cache/huggingface/ \\
                VLLM_TORCH_PROFILER_DIR=${server_config_dir} \\
                VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000 \\
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\
                VLLM_ATTENTION_BACKEND=FLASHINFER \\
                vllm serve ${model_name} \\
                    --dtype auto \\
                    --kv-cache-dtype auto \\
                    -tp ${tp_config} \\
                    -pp ${pp_config} \\
                    --distributed-executor-backend ray \\
                    --max-model-len 10000 \\
                    --max-num-seqs 256 \\
                    --gpu-memory-utilization 0.9 \\
                    --custom-profiler --no-enable-prefix-caching \\
                    --max-num-batched-tokens 256 \\
                    > \"$server_config_dir/server.log\" 2>&1 &"
    
    # Execute the server command
    eval "$server_cmd"
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID (TP=${tp_config}, PP=${pp_config})"
    
    # Wait for server to be ready
    wait_for_server_ready
    
    # Test all frequencies for this parallelism configuration
    for freq in $frequencies; do
        echo "---------- Setting GPU clock frequency to ${freq} MHz ----------"
        
        # Lock all GPUs to the specified frequency
        lock_all_gpus_to_clock $freq
        sleep 2
        
        # Verify that all GPUs are locked to the expected clock
        echo "Verifying GPU clock locks..."
        if ! check_all_gpu_clocks $freq; then
            echo "ERROR: GPU clocks are not properly locked to ${freq} MHz. Skipping this frequency."
            release_all_gpus_clock
            continue
        fi
        
        # Run all TPS/input/output combinations for this frequency
        for tps in $tps_values; do
            for in_len in $input_lens; do
                for out_len in $output_lens; do

                    target_time=$TIME_LIMIT
                    
                    # Calculate request rate by dividing TPS by input length (using bash arithmetic)
                    # Convert to format with 2 decimal places: tps*100/input_len/100
                    request_rate_int=$((tps * 100 / in_len))
                    request_rate=$(printf "%.2f" "$(echo "scale=2; $request_rate_int / 100" | bc -l 2>/dev/null || echo "$tps $in_len" | awk '{printf "%.2f", $1/$2}')")
                    
                    # Calculate number of requests using floating-point arithmetic with bc
                    num_requests=$(echo "scale=0; $target_time * $request_rate / 1" | bc -l)
                    
                    # Ensure at least 5 requests
                    if [ "$num_requests" -lt 5 ]; then
                        num_requests=5
                    fi
                    
                    current_run=$((current_run + 1))
                    echo "------------------------------------------------------------"
                    echo "📊 Running Test [$current_run / $total_runs]: TP=${tp_config}, PP=${pp_config}, Freq=${freq}MHz, TPS=${tps}, ReqRate=${request_rate}req/s, In=${in_len}t, Out=${out_len}t"

                    
                    # Create specific directory for this configuration including parallelism
                    config_dir="${log_dir}/tp${tp_config}_pp${pp_config}_f${freq}_tps${tps}_in${in_len}_out${out_len}"
                    mkdir -p "$config_dir"
                    
                    # Define a descriptive log file name within the config directory
                    log_file="${config_dir}/benchmark_rr${request_rate}.log"

                    result_dir="${config_dir}/traced_dataset"
                    mkdir -p "$result_dir"

                    
                    # Start GPU energy monitor for this specific test case
                    echo "Starting GPU energy monitor for this test case..."
                    python3 gpu_energy_monitor.py "$result_dir" &
                    ENERGY_MONITOR_PID=$!
                    
                    # Execute the python script with the current set of parameters
                    # Note: Using unmodified vllm (no --use_s1,s2 etc. flags)
                    # Using random dataset with specified input/output lengths
                    # Redirect both stdout and stderr to the log file
                    $python_exec $python_script \
                        --backend vllm \
                        --model "$model_name" \
                        --dataset-name random \
                        --request-rate "$request_rate" \
                        --random-input-len "$in_len" \
                        --random-output-len "$out_len" \
                        --profile \
                        --num-prompts ${num_requests} \
                        --burstiness 1 \
                        --random-range-ratio 0.3 \
                        --seed 42 \
                        --microbatch-size ${pp_config} \
                        --save-result \
                        --ignore-eos \
                        --result-dir "$result_dir"
                    
                    # Capture benchmark exit code before stopping energy monitor
                    benchmark_exit_code=$?
                    
                    # Stop GPU energy monitor for this test case
                    echo "Stopping GPU energy monitor for this test case..."
                    if [ -n "$ENERGY_MONITOR_PID" ]; then
                        kill -TERM $ENERGY_MONITOR_PID 2>/dev/null
                        wait $ENERGY_MONITOR_PID 2>/dev/null
                        echo "Energy monitor with PID $ENERGY_MONITOR_PID stopped."
                    fi
                    
                    # Check benchmark exit code
                    if [ $benchmark_exit_code -eq 0 ]; then
                        echo "✅ Success. Output saved to $log_file"
                    else
                        echo "🔥 Failure. Check log file for errors: $log_file"
                    fi
                done
            done
        done
        
        echo "Releasing clock locks for frequency ${freq} MHz..."
        release_all_gpus_clock
        sleep 1
    done
    
    # Stop vLLM server for this parallelism configuration
    echo "Stopping vLLM server for TP=${tp_config}, PP=${pp_config}..."
    if [ -n "$SERVER_PID" ]; then
        kill -TERM $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        echo "Server with PID $SERVER_PID stopped."
    fi
    echo "========== Completed parallelism config: TP=${tp_config}, PP=${pp_config} =========="
done

echo "------------------------------------------------------------"
echo "🎉 All $total_runs profiling runs completed."

# Post-processing: collect metrics and generate visualizations
DYNAMO_DIR="$(dirname "$0")/dynamollm_profiles"
log_dir_name="$(basename "$log_dir")"

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
model_name_clean="${model_name//\//_}"

echo ""
echo "====== Post-processing: Collecting metrics ======"
(cd "$DYNAMO_DIR" && python3 collect.py --folder "${log_dir_name}" --gpu-name "${gpu_name}" --model "${model_name}")

csv_file="${DYNAMO_DIR}/dynamo_dvfs_profile_${gpu_name}_${model_name_clean}.csv"
if [ -f "$csv_file" ]; then
    echo ""
    echo "====== Post-processing: Generating visualizations ======"
    (cd "$DYNAMO_DIR" && python3 visualize.py "dynamo_dvfs_profile_${gpu_name}_${model_name_clean}.csv")
    echo "Visualizations saved in: $DYNAMO_DIR"
else
    echo "WARNING: CSV file not found at $csv_file, skipping visualization."
fi