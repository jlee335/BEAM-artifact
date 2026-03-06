"""
Offline profiling test for mixed prefill/decode batches.

This script profiles latency and energy consumption across various configurations
with built-in capacity checking to prevent OOM errors.

Key Features:
- Validates batch configurations against system capacity before profiling
- Checks: max_num_batched_tokens, max_num_seqs, max_model_len
- Skips invalid configurations and logs them in the output CSV
- Supports resume from partial results
- Profiles across prefill tokens, context lengths, and GPU frequencies
"""

import argparse
import dataclasses
import os
import time
import torch
import pandas as pd

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
                    nvmlDeviceGetTotalEnergyConsumption,
                    nvmlDeviceResetGpuLockedClocks,
                    nvmlDeviceSetGpuLockedClocks, nvmlInit)

nvmlInit()

NUM_TRIALS = 20  # Fixed number of iterations per configuration


def runner_func(batch_size, decode_size, total_ctx_len, model_executor, num_trials):
    # For batch_size < 128, use uniform_decode mode
    if batch_size < 128:
        uniform_decode = True
        decode_size = batch_size
        prefill_size_real = 0
    else:
        uniform_decode = False
        # if prefill_size is smaller than decode_size, that prefill_size is 0
        prefill_size_real = max(batch_size - decode_size, 0)
    
    model_executor.collective_rpc(
        "_dummy_run_mixed",
        args=(prefill_size_real, decode_size, total_ctx_len, uniform_decode, num_trials))
    


def run_n_times(func, n_trials, *args, **kwargs) -> tuple[int, float]:
    """Run func exactly n_trials times, return number of trials and total time taken."""
    start_time = time.time()
    with torch.inference_mode():
        func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    time_taken = end_time - start_time
    return n_trials, time_taken



def main(args: argparse.Namespace):
    print(args)

    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size

    engine_args = EngineArgs.from_cli_args(args)
    print("engine_args : ", engine_args)

    # Set GPU memory utilization.
    engine_args.gpu_memory_utilization = 0.90

    llm = LLM(**dataclasses.asdict(engine_args))
    
    ############################################################
    # Capture model for CUDA graph optimization before profiling
    print("Capturing CUDA graphs for model optimization...")
    try:
        llm.llm_engine.engine_core.collective_rpc("compile_or_warm_up_model")
        print("CUDA graph capture completed successfully.")
    except Exception as e:
        print(f"CUDA graph capture failed or not supported: {e}")
    ############################################################

    model_executor = llm.llm_engine.engine_core
    
    # Get GPU information and indices for tensor parallelism
    gpu_indices = [0]  # Default to GPU 0 for single GPU
    if tp_size > 1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                local_rank = dist.get_rank()
                world_size = dist.get_world_size()
                gpu_indices = list(range(tp_size))
                print(f"Distributed environment: Local rank={local_rank}, World size={world_size}")
            else:
                gpu_indices = list(range(tp_size))
                print(f"Using local GPUs for TP: {gpu_indices}")
        except Exception as e:
            gpu_indices = list(range(tp_size))
            print(f"Fallback to local GPUs for TP: {gpu_indices} (Error: {e})")
    
    # Get GPU name
    gpu_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(gpu_indices[0]))
    gpu_name = gpu_name.decode('utf-8') if isinstance(gpu_name, bytes) else gpu_name
    print(f"GPU name: {gpu_name}")
    
    # Format model name for file naming
    model_name = args.model.replace("/", "_")
    
    # Get available GPU graphics clocks (using max memory clock)
    import pynvml
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        # Get the maximum memory clock to query graphics clocks
        mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        max_mem_clock = max(mem_clocks)
        print(f"Using memory clock: {max_mem_clock} MHz")
        
        # Get graphics clocks for this memory clock
        graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_mem_clock)
        clocks = sorted(graphics_clocks, reverse=True)
        
        # Filter out very low clocks
        clocks = [clk for clk in clocks if clk > 400]
        clocks = clocks[::4]
        
        # if GPU is A6000, min_clock is 800, else, 500
        min_clock = 800 if "A6000" in gpu_name else 500
        clocks = [clk for clk in clocks if clk > min_clock]

        
        print(f"Available graphics clocks: {len(graphics_clocks)}")
    except Exception as e:
        print(f"Could not get clocks from pynvml: {e}, using default")
        clocks = [1410, 1530, 1650]  # Default fallback clocks
    
    print(f"Using clocks: {clocks}")
    
    ############################################################
    # Check System Capacity
    ############################################################
    def check_batch_capacity(llm, batch_size, num_decode_tokens, total_ctx_len):
        """
        Check if the batch configuration fits within the system's capacity.
        Returns (is_valid, error_message).
        """
        scheduler_config = llm.llm_engine.vllm_config.scheduler_config
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_num_seqs = scheduler_config.max_num_seqs
        max_model_len = llm.llm_engine.vllm_config.model_config.max_model_len
        
        # Calculate total tokens in batch
        total_tokens = batch_size + num_decode_tokens
        total_seqs = num_decode_tokens + (1 if batch_size > 0 else 0)
        
        # Check 1: Total tokens vs max_num_batched_tokens
        if total_tokens > max_num_batched_tokens:
            return False, (f"Total tokens ({total_tokens}) exceeds max_num_batched_tokens "
                          f"({max_num_batched_tokens})")
        
        # Check 2: Number of sequences vs max_num_seqs
        if total_seqs > max_num_seqs:
            return False, (f"Total sequences ({total_seqs}) exceeds max_num_seqs "
                          f"({max_num_seqs})")
        
        # Check 3: Batch size vs max_model_len
        if batch_size > max_model_len:
            return False, (f"Batch size ({batch_size}) exceeds max_model_len "
                          f"({max_model_len})")
        
        # Check 4: Context length per decode sequence
        if num_decode_tokens > 0:
            ctx_per_decode = total_ctx_len // num_decode_tokens
            if ctx_per_decode > max_model_len:
                return False, (f"Context per decode ({ctx_per_decode}) exceeds max_model_len "
                              f"({max_model_len})")
        
        return True, None
    
    # Print system capacity
    scheduler_config = llm.llm_engine.vllm_config.scheduler_config
    print("\n" + "="*80)
    print("SYSTEM CAPACITY")
    print("="*80)
    print(f"Max model length: {llm.llm_engine.vllm_config.model_config.max_model_len}")
    print(f"Max num batched tokens: {scheduler_config.max_num_batched_tokens}")
    print(f"Max num sequences: {scheduler_config.max_num_seqs}")
    print(f"GPU memory utilization: {llm.llm_engine.vllm_config.cache_config.gpu_memory_utilization}")
    print("="*80 + "\n")
    
    ############################################################
    # Profile Configuration
    ############################################################
    # batch_size_list = [
    #         1, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 320, 384, 448, 512,
    #         640, 768, 896, 1024, 1280, 1536, 1792, 2048
    #     ]

    # batch_size_list = [
    #     128, 256, 512, 1024
    # ]
    batch_size_list = [
        1, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 320, 384, 448, 512
    ]

    total_ctx_lens = [2048, 40000] # Big and Little 
    num_decode_tokens = 60  # Fixed
    
    print("="*80)
    print("PROFILING CONFIGURATION")
    print("="*80)
    print(f"Batch sizes: {batch_size_list}")
    print(f"Total context lengths: {total_ctx_lens}")
    print(f"Decode tokens (fixed): {num_decode_tokens}")
    print(f"GPU clocks: {clocks}")
    print("="*80 + "\n")
    
    # Validate all configurations before starting
    print("="*80)
    print("VALIDATING BATCH CONFIGURATIONS")
    print("="*80)
    invalid_configs = []
    for total_ctx_len in total_ctx_lens:
        for batch_size in batch_size_list:
            is_valid, error_msg = check_batch_capacity(
                llm, batch_size, num_decode_tokens, total_ctx_len)
            if not is_valid:
                invalid_configs.append((batch_size, num_decode_tokens, total_ctx_len, error_msg))
                print(f"❌ INVALID: batch_size={batch_size}, decode={num_decode_tokens}, "
                      f"ctx_len={total_ctx_len}")
                print(f"   Reason: {error_msg}")
            else:
                print(f"✓ VALID: batch_size={batch_size}, decode={num_decode_tokens}, "
                      f"ctx_len={total_ctx_len}, total_tokens={batch_size + num_decode_tokens}")
    
    if invalid_configs:
        print(f"\n⚠️  WARNING: {len(invalid_configs)} invalid configurations found!")
        print("These configurations will be skipped during profiling.")
    else:
        print(f"\n✓ All {len(batch_size_list) * len(total_ctx_lens)} configurations are valid!")
    print("="*80 + "\n")
    
    def set_gpu_clock(clock):
        """Set GPU clock for all GPUs in the tensor parallel group."""
        for gpu_idx in gpu_indices:
            nvmlDeviceSetGpuLockedClocks(
                nvmlDeviceGetHandleByIndex(gpu_idx), clock, clock)
    
    def reset_gpu_clock():
        """Reset GPU clock for all GPUs in the tensor parallel group."""
        for gpu_idx in gpu_indices:
            nvmlDeviceResetGpuLockedClocks(nvmlDeviceGetHandleByIndex(gpu_idx))
    
    # Determine output file name
    output_dir = "offline_profile_results"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/mixed_profile_{gpu_name}_{model_name}_tp{tp_size}_pp{pp_size}.csv"
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(file_name):
        print(f"Loading existing data from {file_name}")
        existing_df = pd.read_csv(file_name)
        for _, row in existing_df.iterrows():
            # Support both old column name (prefill_tokens) and new (batch_size)
            batch_size_col = 'batch_size' if 'batch_size' in row else 'prefill_tokens'
            key = (row['clock'], row[batch_size_col], row['decode_tokens'], 
                   row['total_ctx_len'])
            # Only add if num_trials equals NUM_TRIALS
            num_trials = row['num_trials']
            if num_trials == NUM_TRIALS:
                existing_data[key] = {
                    'time_taken': row['time_taken'],
                    'energy_consumption': row['energy_consumption'],
                    'num_trials': int(row['num_trials'])
                }
        print(f"Found {len(existing_data)} existing datapoints")
    
    # Initialize data list to store results
    data = []
    
    # Start profiling
    # torch.cuda.cudart().cudaProfilerStart()
    
    total_configs = len(batch_size_list) * len(total_ctx_lens) * len(clocks)
    current_config = 0
    
    for total_ctx_len in total_ctx_lens:
        for batch_size in batch_size_list:
            # For batch-size over 128, only run in max total_ctx_len
            if batch_size > 128 and total_ctx_len != max(total_ctx_lens):
                continue
            for clock in clocks:
                current_config += 1
                
                # Check if this datapoint already exists
                datapoint_key = (clock, batch_size, num_decode_tokens, total_ctx_len)
                if datapoint_key in existing_data:
                    print(f"[{current_config}/{total_configs}] Skipping existing: "
                          f"clock={clock}, batch_size={batch_size}, "
                          f"decode={num_decode_tokens}, ctx_len={total_ctx_len}")
                    data.append({
                        "clock": clock,
                        "batch_size": batch_size,
                        "decode_tokens": num_decode_tokens,
                        "total_ctx_len": total_ctx_len,
                        "time_taken": existing_data[datapoint_key]['time_taken'],
                        "energy_consumption": existing_data[datapoint_key]['energy_consumption'],
                        "num_trials": existing_data[datapoint_key]['num_trials'],
                        "valid": 1
                    })
                    continue
                
                # Check if this configuration is valid
                is_valid, error_msg = check_batch_capacity(
                    llm, batch_size, num_decode_tokens, total_ctx_len)
                if not is_valid:
                    print(f"[{current_config}/{total_configs}] ❌ SKIPPING INVALID: "
                          f"clock={clock}, batch_size={batch_size}, "
                          f"decode={num_decode_tokens}, ctx_len={total_ctx_len}")
                    print(f"  Reason: {error_msg}")
                    data.append({
                        "clock": clock,
                        "batch_size": batch_size,
                        "decode_tokens": num_decode_tokens,
                        "total_ctx_len": total_ctx_len,
                        "time_taken": 0,
                        "energy_consumption": 0,
                        "num_trials": 0,
                        "valid": 0,
                        "error": error_msg
                    })
                    continue
                
                print(f"\n[{current_config}/{total_configs}] Profiling: "
                      f"clock={clock} MHz, batch_size={batch_size}, "
                      f"decode={num_decode_tokens}, ctx_len={total_ctx_len}")
                
                # Set GPU clock
                set_gpu_clock(clock)
                
                # Warmup run
                runner_func(batch_size, num_decode_tokens, total_ctx_len, model_executor, 1)
                time.sleep(0.1)
                
                # Measure energy consumption across all GPUs
                start_power_per_gpu = []
                for gpu_idx in gpu_indices:
                    start_power_per_gpu.append(
                        nvmlDeviceGetTotalEnergyConsumption(
                            nvmlDeviceGetHandleByIndex(gpu_idx)))
                
                # Run exactly NUM_TRIALS times
                num_trials, time_taken = run_n_times(
                    runner_func, NUM_TRIALS, batch_size, num_decode_tokens,
                    total_ctx_len, model_executor, NUM_TRIALS)
                
                torch.cuda.synchronize()
                
                # Measure end energy
                end_power_per_gpu = []
                for gpu_idx in gpu_indices:
                    end_power_per_gpu.append(
                        nvmlDeviceGetTotalEnergyConsumption(
                            nvmlDeviceGetHandleByIndex(gpu_idx)))
                
                # Calculate averages
                avg_time_taken = time_taken / num_trials
                total_energy_consumption = sum(
                    end_power_per_gpu[i] - start_power_per_gpu[i]
                    for i in range(len(gpu_indices)))
                avg_energy_consumption = total_energy_consumption / num_trials
                
                print(f"  Completed {num_trials} trials in {time_taken:.2f}s")
                print(f"  Avg latency: {avg_time_taken*1000:.2f}ms")
                print(f"  Avg energy: {avg_energy_consumption:.2f}mJ")
                
                data.append({
                    "clock": clock,
                    "batch_size": batch_size,
                    "decode_tokens": num_decode_tokens,
                    "total_ctx_len": total_ctx_len,
                    "time_taken": avg_time_taken,
                    "energy_consumption": avg_energy_consumption,
                    "num_trials": num_trials,
                    "valid": 1
                })
                
                # Save intermediate results
                if current_config % 10 == 0 or current_config == total_configs:
                    df = pd.DataFrame(data)
                    df.to_csv(file_name, index=False)
                    print(f"  Saved intermediate results to {file_name}")
    
    # Stop profiling
    # torch.cuda.cudart().cudaProfilerStop()
    
    # Reset GPU clocks
    reset_gpu_clock()
    
    # Save final results
    df = pd.DataFrame(data)
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print(df)
    print(f"\nResults saved to: {file_name}")
    print(f"Total configurations: {len(df)}")
    print(f"Valid configurations: {df['valid'].sum()}")
    print(f"Total GPUs monitored: {len(gpu_indices)}")
    print(f"GPU indices: {gpu_indices}")
    print("="*80)
    
    df.to_csv(file_name, index=False)


if __name__ == "__main__":    
    parser = FlexibleArgumentParser(
        description="Profile latency and energy consumption for mixed prefill/decode batches.")
    
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)


