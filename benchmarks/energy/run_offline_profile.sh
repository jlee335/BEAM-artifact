#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
export NSYS=/opt/nvidia/nsight-systems/2024.5.1/bin/
export PATH=$PATH:$NSYS
export HF_HOME=/workspace/.cache/huggingface/
export HF_TOKEN=${HF_TOKEN}
# A100-80GB architecture
export TORCH_CUDA_ARCH_LIST="8.0"
# Script to run offline profiling for mixed prefill/decode batches
# Profiles latency and energy consumption across different configurations
# Edit the variables below to configure the profiling run

# ========== CONFIGURATION ==========
MODEL="Qwen/Qwen2.5-32B"
# MODEL="meta-llama/Llama-3.3-70B-Instruct"
# MODEL="ByteResearch/Llama-3-8B-Instruct"
# MODEL="Qwen/Qwen2.5-14B"


TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.90

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --tp|--tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift ;;
        --pp|--pipeline-parallel-size) PIPELINE_PARALLEL_SIZE="$2"; shift ;;
        --gpu-memory|--gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ===================================

echo "======================================================================"
echo "Running Offline Profile Test"
echo "======================================================================"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "======================================================================"
echo ""

# Create output directory
mkdir -p offline_profile_results

# Run the profiling test
VLLM_TORCH_PROFILER_DIR=. \
VLLM_ATTENTION_BACKEND=FLASHINFER \
 python test_offline_profile_new.py \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --distributed-executor-backend ray
echo ""
echo "======================================================================"
echo "Profiling Complete!"
echo "Results saved to: offline_profile_results/"
echo "======================================================================"
