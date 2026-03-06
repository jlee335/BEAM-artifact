"""
Offline profiling test for two separate benchmarks:
A) Prefill benchmark - tests prefill latency/energy across token counts
B) Decode-only benchmark - tests decode latency/energy across batch sizes and context lengths

Key Features:
- Validates batch configurations against system capacity before profiling
- Checks: max_num_batched_tokens, max_num_seqs, max_model_len
- Skips invalid configurations and logs them in the output CSV
- Supports resume from partial results
- Profiles across GPU frequencies
"""

import argparse
import dataclasses
import os
import time
import torch
import pandas as pd
import numpy as np

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
                    nvmlDeviceGetTotalEnergyConsumption,
                    nvmlDeviceResetGpuLockedClocks,
                    nvmlDeviceSetGpuLockedClocks, nvmlInit)

nvmlInit()

NUM_TRIALS = 20  # Default number of iterations per configuration
NUM_TRIALS_DECODE = 50  # Number of trials for decode-only benchmark


def runner_func_prefill(num_tokens, model_executor, num_trials, create_mixed_batch=True):
    """
    Runner function for prefill benchmark using _dummy_run.
    
    Args:
        num_tokens: Number of tokens for prefill
        model_executor: The model executor instance
        num_trials: Number of trials to run (handled by external loop)
        create_mixed_batch: Whether to create mixed batch
    """
    # Note: num_trials is handled by the external run_n_times loop
    # Each call here is a single iteration
    model_executor.collective_rpc(
        "_dummy_run",
        args=(num_tokens,),
        kwargs={
            "cudagraph_runtime_mode": None,  # Let system decide
            "uniform_decode": False,
            "create_mixed_batch": create_mixed_batch,
        })


def runner_func_decode(decode_batch_size, total_ctx_len, model_executor, num_trials):
    """
    Runner function for decode-only benchmark using _dummy_run_mixed.
    
    Args:
        decode_batch_size: Number of decode sequences
        total_ctx_len: Total context length across all sequences
        model_executor: The model executor instance
        num_trials: Number of trials to run (handled internally by _dummy_run_mixed)
    """
    # _dummy_run_mixed handles the num_trials looping internally
    model_executor.collective_rpc(
        "_dummy_run_mixed",
        args=(0, decode_batch_size, total_ctx_len, True, num_trials))


def run_n_times(func, n_trials, *args, **kwargs) -> tuple[int, float]:
    """Run func exactly n_trials times, return number of trials and total time taken."""
    start_time = time.time()
    with torch.inference_mode():
        for _ in range(n_trials):
            func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    time_taken = end_time - start_time
    return n_trials, time_taken


def calculate_max_kv_cache_capacity(llm):
    """
    Calculate the maximum KV cache capacity based on system configuration.
    Returns the maximum context length (in tokens) that can be supported.
    
    This is calculated as: num_gpu_blocks * block_size
    where num_gpu_blocks is the number of KV cache blocks allocated
    and block_size is the number of tokens per block.
    """
    cache_config = llm.llm_engine.vllm_config.cache_config
    model_config = llm.llm_engine.vllm_config.model_config
    
    # Get the actual number of GPU blocks allocated
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    
    if num_gpu_blocks is None or block_size is None:
        # Fallback to conservative estimate if profiling hasn't been done
        print("Warning: KV cache not yet profiled, using conservative estimate")
        max_kv_capacity = min(
            model_config.max_model_len,
            llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
        )
    else:
        # Calculate maximum tokens from blocks
        # Note: We subtract 1 block as buffer/null block
        max_kv_capacity = max((num_gpu_blocks - 1) * block_size, 0)
        
        # Cap at max_model_len to avoid exceeding model's context limit
        max_kv_capacity = min(max_kv_capacity, model_config.max_model_len)
    
    print(f"KV Cache Configuration:")
    print(f"  - Number of GPU blocks: {num_gpu_blocks}")
    print(f"  - Block size: {block_size} tokens")
    print(f"  - Calculated max KV capacity: {max_kv_capacity} tokens")
    print(f"  - Model max length: {model_config.max_model_len} tokens")
    
    return max_kv_capacity


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
        
        # Sample clocks: A100 uses every 4th clock, others use every 8th
        stride = 4 if "A100" in gpu_name else 8
        clocks = clocks[::stride]
        
        # if GPU is A6000, min_clock is 800, else, 500
        min_clock = 800 if "A6000" in gpu_name else 500
        clocks = [clk for clk in clocks if clk > min_clock]
        
        print(f"Available graphics clocks: {len(graphics_clocks)}")
    except Exception as e:
        print(f"Could not get clocks from pynvml: {e}, using default")
        clocks = [1410, 1530, 1650]  # Default fallback clocks
    
    print(f"Using clocks: {clocks}")
    
    ############################################################
    # System Capacity Information
    ############################################################
    scheduler_config = llm.llm_engine.vllm_config.scheduler_config
    model_config = llm.llm_engine.vllm_config.model_config
    
    print("\n" + "="*80)
    print("SYSTEM CAPACITY")
    print("="*80)
    print(f"Max model length: {model_config.max_model_len}")
    print(f"Max num batched tokens: {scheduler_config.max_num_batched_tokens}")
    print(f"Max num sequences: {scheduler_config.max_num_seqs}")
    print(f"GPU memory utilization: {llm.llm_engine.vllm_config.cache_config.gpu_memory_utilization}")
    print("="*80 + "\n")
    
    # Calculate max KV cache capacity
    max_kv_capacity = calculate_max_kv_cache_capacity(llm)
    
    ############################################################
    # A) PREFILL BENCHMARK
    ############################################################
    print("\n" + "="*80)
    print("BENCHMARK A: PREFILL")
    print("="*80)
    
    # Generate prefill token counts: 64, 128, 256, 384, 512, ..., 2048
    # Start with 64, then increment by 128
    prefill_token_counts = [1, 16, 32, 64]
    num_tokens = 128
    while num_tokens <= 2048:
        prefill_token_counts.append(num_tokens)
        num_tokens += 128
    
    print(f"Prefill token counts: {prefill_token_counts}")
    print(f"GPU clocks: {clocks}")
    print(f"Trials per configuration: {NUM_TRIALS}")
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
    
    # Determine output file names
    output_dir = "offline_profile_results_v2"
    os.makedirs(output_dir, exist_ok=True)
    prefill_file = f"{output_dir}/prefill_profile_{gpu_name}_{model_name}_tp{tp_size}_pp{pp_size}.csv"
    decode_file = f"{output_dir}/decode_profile_{gpu_name}_{model_name}_tp{tp_size}_pp{pp_size}.csv"
    
    # Load existing prefill data if file exists
    existing_prefill_data = {}
    if os.path.exists(prefill_file):
        print(f"Loading existing prefill data from {prefill_file}")
        existing_df = pd.read_csv(prefill_file)
        for _, row in existing_df.iterrows():
            key = (row['clock'], row['num_tokens'])
            # Reuse data if existing trials >= required trials
            # if row['num_trials'] >= NUM_TRIALS:
            existing_prefill_data[key] = {
                'time_taken': row['time_taken'],
                'energy_consumption': row['energy_consumption'],
                'num_trials': int(row['num_trials'])
            }
        print(f"Found {len(existing_prefill_data)} existing prefill datapoints with sufficient trials (>= {NUM_TRIALS})")
    
    # Initialize prefill data list
    prefill_data = []
    
    # Validate and run prefill benchmark
    print("="*80)
    print("VALIDATING PREFILL CONFIGURATIONS")
    print("="*80)
    
    total_prefill_configs = len(prefill_token_counts) * len(clocks)
    current_config = 0
    
    for num_tokens in prefill_token_counts:
        # Check if configuration is valid
        if num_tokens > model_config.max_model_len:
            print(f"❌ INVALID: num_tokens={num_tokens} exceeds max_model_len={model_config.max_model_len}")
            for clock in clocks:
                prefill_data.append({
                    "clock": clock,
                    "num_tokens": num_tokens,
                    "time_taken": 0,
                    "energy_consumption": 0,
                    "num_trials": 0,
                    "valid": 0,
                    "error": f"num_tokens exceeds max_model_len"
                })
            continue
        
        if num_tokens > scheduler_config.max_num_batched_tokens:
            print(f"❌ INVALID: num_tokens={num_tokens} exceeds max_num_batched_tokens={scheduler_config.max_num_batched_tokens}")
            for clock in clocks:
                prefill_data.append({
                    "clock": clock,
                    "num_tokens": num_tokens,
                    "time_taken": 0,
                    "energy_consumption": 0,
                    "num_trials": 0,
                    "valid": 0,
                    "error": f"num_tokens exceeds max_num_batched_tokens"
                })
            continue
        
        print(f"✓ VALID: num_tokens={num_tokens}")
        
        for clock in clocks:
            current_config += 1
            
            # Check if this datapoint already exists
            datapoint_key = (clock, num_tokens)
            if datapoint_key in existing_prefill_data:
                existing_trials = existing_prefill_data[datapoint_key]['num_trials']
                print(f"[{current_config}/{total_prefill_configs}] Skipping existing: "
                      f"clock={clock}, num_tokens={num_tokens} "
                      f"(existing trials: {existing_trials} >= required: {NUM_TRIALS})")
                prefill_data.append({
                    "clock": clock,
                    "num_tokens": num_tokens,
                    "time_taken": existing_prefill_data[datapoint_key]['time_taken'],
                    "energy_consumption": existing_prefill_data[datapoint_key]['energy_consumption'],
                    "num_trials": existing_prefill_data[datapoint_key]['num_trials'],
                    "valid": 1
                })
                continue
            
            print(f"\n[{current_config}/{total_prefill_configs}] Profiling prefill: "
                  f"clock={clock} MHz, num_tokens={num_tokens}, trials={NUM_TRIALS}")
            
            # Set GPU clock
            set_gpu_clock(clock)
            
            # Warmup run
            runner_func_prefill(num_tokens, model_executor, 1, True)
            torch.cuda.synchronize()
            time.sleep(0.1)
            
            # Measure energy consumption across all GPUs
            start_power_per_gpu = []
            for gpu_idx in gpu_indices:
                start_power_per_gpu.append(
                    nvmlDeviceGetTotalEnergyConsumption(
                        nvmlDeviceGetHandleByIndex(gpu_idx)))
            
            # Run benchmark (run NUM_TRIALS times)
            num_trials, time_taken = run_n_times(
                runner_func_prefill, NUM_TRIALS, num_tokens, model_executor, NUM_TRIALS, True)
            
            torch.cuda.synchronize()
            
            # Measure end energy
            end_power_per_gpu = []
            for gpu_idx in gpu_indices:
                end_power_per_gpu.append(
                    nvmlDeviceGetTotalEnergyConsumption(
                        nvmlDeviceGetHandleByIndex(gpu_idx)))
            
            # Calculate averages
            avg_time_taken = time_taken / NUM_TRIALS
            total_energy_consumption = sum(
                end_power_per_gpu[i] - start_power_per_gpu[i]
                for i in range(len(gpu_indices)))
            avg_energy_consumption = total_energy_consumption / NUM_TRIALS
            
            print(f"  Completed {NUM_TRIALS} trials in {time_taken:.2f}s")
            print(f"  Avg latency: {avg_time_taken*1000:.2f}ms")
            print(f"  Avg energy: {avg_energy_consumption:.2f}mJ")
            
            prefill_data.append({
                "clock": clock,
                "num_tokens": num_tokens,
                "time_taken": avg_time_taken,
                "energy_consumption": avg_energy_consumption,
                "num_trials": NUM_TRIALS,
                "valid": 1
            })
            
            # Save intermediate results
            if current_config % 10 == 0 or current_config == total_prefill_configs:
                df = pd.DataFrame(prefill_data)
                df.to_csv(prefill_file, index=False)
                print(f"  Saved intermediate results to {prefill_file}")
    
    # Save final prefill results
    prefill_df = pd.DataFrame(prefill_data)
    prefill_df.to_csv(prefill_file, index=False)
    
    print("\n" + "="*80)
    print("PREFILL BENCHMARK COMPLETE")
    print("="*80)
    print(prefill_df)
    print(f"\nResults saved to: {prefill_file}")
    print(f"Total configurations: {len(prefill_df)}")
    print(f"Valid configurations: {prefill_df['valid'].sum()}")
    print("="*80)
    
    exit()
    
    
    ############################################################
    # B) DECODE-ONLY BENCHMARK
    ############################################################
    print("\n" + "="*80)
    print("BENCHMARK B: DECODE-ONLY")
    print("="*80)
    
    # Decode batch sizes
    decode_batch_sizes = [1, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    
    # Generate decode context lengths: 256, 2048, 4096, 6144, ...
    # max total decode length 50000
    
    decode_ctx_lengths = []
    # Divide 256 ~ max_kv_capacity into roughly 10 points (including 256)
    num_points = 10
    if max_kv_capacity <= 256:
        decode_ctx_lengths = [256]
    else:
        # Linearly spaced, including 256 and max_kv_capacity
        decode_ctx_lengths = [int(round(x)) for x in 
                              list(
                                  set(
                                      [256] + 
                                      [int(round(i))
                                       for i in 
                                       np.linspace(256, max_kv_capacity, num_points)]
                                  )
                              )
                             ]
        decode_ctx_lengths = sorted(decode_ctx_lengths)
    
    print(f"Decode batch sizes: {decode_batch_sizes}")
    print(f"Decode context lengths: {decode_ctx_lengths}")
    print(f"GPU clocks: {clocks}")
    print(f"Trials per configuration: {NUM_TRIALS_DECODE}")
    print("="*80 + "\n")
    
    # Load existing decode data if file exists
    existing_decode_data = {}
    if os.path.exists(decode_file):
        print(f"Loading existing decode data from {decode_file}")
        existing_df = pd.read_csv(decode_file)
        for _, row in existing_df.iterrows():
            key = (row['clock'], row['batch_size'], row['total_ctx_len'])
            # Reuse data if existing trials >= required trials
            if row['num_trials'] >= NUM_TRIALS_DECODE:
                existing_decode_data[key] = {
                    'time_taken': row['time_taken'],
                    'energy_consumption': row['energy_consumption'],
                    'num_trials': int(row['num_trials'])
                }
        print(f"Found {len(existing_decode_data)} existing decode datapoints with sufficient trials (>= {NUM_TRIALS_DECODE})")
    
    # Initialize decode data list
    decode_data = []
    
    # Validate and run decode benchmark
    print("="*80)
    print("VALIDATING DECODE CONFIGURATIONS")
    print("="*80)
    
    total_decode_configs = len(decode_batch_sizes) * len(decode_ctx_lengths) * len(clocks)
    current_config = 0
    
    for batch_size in decode_batch_sizes:
        for total_ctx_len in decode_ctx_lengths:
            # Check if configuration is valid
            is_valid = True
            error_msg = None
            
            # Check 1: Number of sequences vs max_num_seqs
            if batch_size > scheduler_config.max_num_seqs:
                is_valid = False
                error_msg = f"Batch size ({batch_size}) exceeds max_num_seqs ({scheduler_config.max_num_seqs})"
            
            # Check 2: Total tokens vs max_num_batched_tokens
            # For decode, we have batch_size sequences each generating 1 token
            elif batch_size > scheduler_config.max_num_batched_tokens:
                is_valid = False
                error_msg = f"Batch size ({batch_size}) exceeds max_num_batched_tokens ({scheduler_config.max_num_batched_tokens})"
            
            # Check 3: Context length per sequence
            elif batch_size > 0:
                ctx_per_seq = total_ctx_len // batch_size
                if ctx_per_seq > model_config.max_model_len:
                    is_valid = False
                    error_msg = f"Context per sequence ({ctx_per_seq}) exceeds max_model_len ({model_config.max_model_len})"
            
            if not is_valid:
                print(f"❌ INVALID: batch_size={batch_size}, total_ctx_len={total_ctx_len}")
                print(f"   Reason: {error_msg}")
                for clock in clocks:
                    decode_data.append({
                        "clock": clock,
                        "batch_size": batch_size,
                        "total_ctx_len": total_ctx_len,
                        "time_taken": 0,
                        "energy_consumption": 0,
                        "num_trials": 0,
                        "valid": 0,
                        "error": error_msg
                    })
                continue
            
            print(f"✓ VALID: batch_size={batch_size}, total_ctx_len={total_ctx_len}, "
                  f"ctx_per_seq={total_ctx_len//batch_size if batch_size > 0 else 0}")
            
            for clock in clocks:
                current_config += 1
                
                # Check if this datapoint already exists
                datapoint_key = (clock, batch_size, total_ctx_len)
                if datapoint_key in existing_decode_data:
                    existing_trials = existing_decode_data[datapoint_key]['num_trials']
                    print(f"[{current_config}/{total_decode_configs}] Skipping existing: "
                          f"clock={clock}, batch_size={batch_size}, ctx_len={total_ctx_len} "
                          f"(existing trials: {existing_trials} >= required: {NUM_TRIALS_DECODE})")
                    decode_data.append({
                        "clock": clock,
                        "batch_size": batch_size,
                        "total_ctx_len": total_ctx_len,
                        "time_taken": existing_decode_data[datapoint_key]['time_taken'],
                        "energy_consumption": existing_decode_data[datapoint_key]['energy_consumption'],
                        "num_trials": existing_decode_data[datapoint_key]['num_trials'],
                        "valid": 1
                    })
                    continue
                
                print(f"\n[{current_config}/{total_decode_configs}] Profiling decode: "
                      f"clock={clock} MHz, batch_size={batch_size}, "
                      f"ctx_len={total_ctx_len}, trials={NUM_TRIALS_DECODE}")
                
                # Set GPU clock
                set_gpu_clock(clock)
                
                # Warmup run
                runner_func_decode(batch_size, total_ctx_len, model_executor, 1)
                torch.cuda.synchronize()
                time.sleep(0.1)
                
                # Measure energy consumption across all GPUs
                start_power_per_gpu = []
                for gpu_idx in gpu_indices:
                    start_power_per_gpu.append(
                        nvmlDeviceGetTotalEnergyConsumption(
                            nvmlDeviceGetHandleByIndex(gpu_idx)))
                
                # Run benchmark (_dummy_run_mixed handles NUM_TRIALS_DECODE internally)
                start_time = time.time()
                runner_func_decode(batch_size, total_ctx_len, model_executor, NUM_TRIALS_DECODE)
                torch.cuda.synchronize()
                end_time = time.time()
                time_taken = end_time - start_time
                
                torch.cuda.synchronize()
                
                # Measure end energy
                end_power_per_gpu = []
                for gpu_idx in gpu_indices:
                    end_power_per_gpu.append(
                        nvmlDeviceGetTotalEnergyConsumption(
                            nvmlDeviceGetHandleByIndex(gpu_idx)))
                
                # Calculate averages
                avg_time_taken = time_taken / NUM_TRIALS_DECODE
                total_energy_consumption = sum(
                    end_power_per_gpu[i] - start_power_per_gpu[i]
                    for i in range(len(gpu_indices)))
                avg_energy_consumption = total_energy_consumption / NUM_TRIALS_DECODE
                
                print(f"  Completed {NUM_TRIALS_DECODE} trials in {time_taken:.2f}s")
                print(f"  Avg latency: {avg_time_taken*1000:.2f}ms")
                print(f"  Avg energy: {avg_energy_consumption:.2f}mJ")
                
                decode_data.append({
                    "clock": clock,
                    "batch_size": batch_size,
                    "total_ctx_len": total_ctx_len,
                    "time_taken": avg_time_taken,
                    "energy_consumption": avg_energy_consumption,
                    "num_trials": NUM_TRIALS_DECODE,
                    "valid": 1
                })
                
                # Save intermediate results
                if current_config % 10 == 0 or current_config == total_decode_configs:
                    df = pd.DataFrame(decode_data)
                    df.to_csv(decode_file, index=False)
                    print(f"  Saved intermediate results to {decode_file}")
    
    # Reset GPU clocks
    reset_gpu_clock()
    
    # Save final decode results
    decode_df = pd.DataFrame(decode_data)
    decode_df.to_csv(decode_file, index=False)
    
    print("\n" + "="*80)
    print("DECODE BENCHMARK COMPLETE")
    print("="*80)
    print(decode_df)
    print(f"\nResults saved to: {decode_file}")
    print(f"Total configurations: {len(decode_df)}")
    print(f"Valid configurations: {decode_df['valid'].sum()}")
    print("="*80)
    
    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETE")
    print("="*80)
    print(f"Prefill results: {prefill_file}")
    print(f"Decode results: {decode_file}")
    print(f"Total GPUs monitored: {len(gpu_indices)}")
    print(f"GPU indices: {gpu_indices}")
    print("="*80)


if __name__ == "__main__":    
    parser = FlexibleArgumentParser(
        description="Profile latency and energy consumption for prefill and decode-only benchmarks.")
    
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
