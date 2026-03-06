# SPDX-License-Identifier: Apache-2.0
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import math
import os
import time
from typing import Any

import pandas as pd
import torch
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
                    nvmlDeviceGetTotalEnergyConsumption,
                    nvmlDeviceResetGpuLockedClocks,
                    nvmlDeviceSetGpuLockedClocks, nvmlInit)

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

nvmlInit()

RUN_TIME = 4
DEFAULT_CLOCK = 1710  # TODO: arbitrary value


def run_until_timeout(func, timeout, *args, **kwargs):
    """
    Continuously runs `func` with given arguments until `timeout` seconds have elapsed.
    
    Parameters:
        func (callable): The function to run repeatedly.
        timeout (float): The total time in seconds to keep running the function.
        *args: Positional arguments for `func`.
        **kwargs: Keyword arguments for `func`.
    """
    num_trials = 0
    start_time = time.time()
    prev_curr_time = start_time
    while True:
        func(*args, **kwargs)
        curr_time = time.time()
        num_trials += 1
        if curr_time - start_time > timeout:
            num_trials -= 1  # Don't count the trial that exceeded timeout
            break
        prev_curr_time = curr_time

    time_taken = prev_curr_time - start_time
    return num_trials, time_taken


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_all_possible_clocks():
    pass


def main(args: argparse.Namespace):
    print(args)

    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size

    engine_args = EngineArgs.from_cli_args(args)

    print("engine_args : ", engine_args)
    # gpu_memory_utilization = 0.99
    engine_args.gpu_memory_utilization = 0.90

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))

    ############################################################
    # Capture model for CUDA graph optimization before profiling
    print("Capturing CUDA graphs for model optimization...")
    try:
        # Try v1 style (no arguments)
        llm.llm_engine.model_executor.collective_rpc("_warm_up_model")
        print("CUDA graph capture completed successfully.")
    except Exception as e:
        print(f"CUDA graph capture failed or not supported: {e}")
    ############################################################

    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len +
        args.output_len), ("Please ensure that max_model_len is greater than"
                           " the sum of input_len and output_len.")

    def run_test(llm):
        # Get the model executor to run _dummy_run on all workers
        model_executor = llm.llm_engine.model_executor

        # Print types of model_executor and related components
        print("model_executor type : ", type(model_executor))
        print("parallel_config.tensor_parallel_size : ",
              model_executor.parallel_config.tensor_parallel_size)
        print("parallel_config.pipeline_parallel_size : ",
              model_executor.parallel_config.pipeline_parallel_size)

        # Get clocks from pynvml. Use every other clock
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            # Get supported clocks for the GPU
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
            # For each memory clock, get supported graphics clocks
            clocks = []
            for mem_clock in mem_clocks:
                graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                    handle, mem_clock)
                # Use every other clock (step=2)
                clocks.extend(graphics_clocks[::2])
            # Remove duplicates and sort
            clocks = sorted(list(set(clocks)))
            # use every other clock

            clocks = clocks[::2]
            # Clocks over 500
            clocks = [clk for clk in clocks if clk > 500]

        except Exception as e:
            print(f"Could not get clocks from pynvml: {e}")
            exit()

        print("clocks : ", clocks)

        batch_sizes = [
            1, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 320, 384, 448, 512,
            640, 768, 896, 1024, 1280, 1536, 1792, 2048, 3072, 4096
        ]

        # Get gpu name and determine GPU indices for tensor parallelism
        # For TP, we need to manage all GPUs in the tensor parallel group
        local_rank = 0
        world_size = 1
        gpu_indices = [0]  # Default to GPU 0 for single GPU

        # If we're in a distributed environment, get the proper ranks
        if tp_size > 1:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    local_rank = dist.get_rank()
                    world_size = dist.get_world_size()
                    # For tensor parallelism, we use consecutive GPU indices
                    # In a single-node multi-GPU TP setup, all GPUs are used
                    gpu_indices = list(range(tp_size))
                    print(
                        f"Distributed environment detected. Local rank: {local_rank}, World size: {world_size}"
                    )
                else:
                    # If distributed is not initialized, assume we're using local GPUs
                    gpu_indices = list(range(tp_size))
                    print(f"Using local GPUs for TP: {gpu_indices}")
            except Exception as e:
                # Fallback to local GPUs if distributed is not available
                gpu_indices = list(range(tp_size))
                print(
                    f"Fallback to local GPUs for TP: {gpu_indices} (Error: {e})"
                )

        # Get gpu name from the first GPU (they should all be the same)
        gpu_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(
            gpu_indices[0]))
        print("GPU name: ", gpu_name)
        # remove b' stuff
        gpu_name = gpu_name.decode('utf-8')
        model_name = args.model
        # format model name
        model_name = model_name.replace("/", "_")

        ############################################################
        # any overrides..
        # batch_sizes = [16]
        # total_ctx_lens = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
        total_ctx_lens = [4096]
        # clocks = [1530]

        # batch_sizes = [
        #     128, 256, 512, 1024
        # ]

        decode_size = 10

        if args.is_idle:
            batch_sizes = [1]

        ############################################################

        def set_gpu_clock_wrapper(clock):
            # nvtx push
            torch.cuda.nvtx.range_push("set_gpu_clock")
            # Set clock for all GPUs in the tensor parallel group
            for gpu_idx in gpu_indices:
                nvmlDeviceSetGpuLockedClocks(
                    nvmlDeviceGetHandleByIndex(gpu_idx), clock, clock)
            torch.cuda.nvtx.range_pop()

        def set_default_clock():
            # Reset clock for all GPUs in the tensor parallel group
            for gpu_idx in gpu_indices:
                nvmlDeviceResetGpuLockedClocks(
                    nvmlDeviceGetHandleByIndex(gpu_idx))

        def runner_func(prefill_size, decode_size, total_ctx_len,
                        block_tables):
            if not args.is_idle:
                with torch.inference_mode():
                    model_executor.collective_rpc(
                        "_dummy_run_mixed",
                        args=(prefill_size, decode_size, total_ctx_len,
                              block_tables))
            else:
                time.sleep(1)

        # Determine file name
        if not args.is_idle:
            file_name = f"offline_profile_results/dvfs_profile_{gpu_name}_{model_name}_tp{tp_size}_pp{pp_size}_context_one.csv"
        else:
            file_name = f"offline_profile_results/idle_profile_{gpu_name}_idle.csv"

        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(file_name):
            print(f"Loading existing data from {file_name}")
            existing_df = pd.read_csv(file_name)
            # Create a dict with tuples (clock, batch_size, total_ctx_len) as keys for quick lookup
            for _, row in existing_df.iterrows():
                key = (row['clock'], row['batch_size'], row['total_ctx_len'])
                existing_data[key] = {
                    'time_taken': row['time_taken'],
                    'energy_consumption': row['energy_consumption']
                }
            print(f"Found {len(existing_data)} existing datapoints")
        # Initialize data list to store new results
        data = []

        # nvtx total
        torch.cuda.cudart().cudaProfilerStart()

        for total_ctx_len in total_ctx_lens:
            for batch_size in batch_sizes:

                # if batch_size smaller than decode_size, that prefill_size is 0
                if batch_size < decode_size:
                    prefill_size = 0
                    decode_size = batch_size
                else:
                    prefill_size = batch_size - decode_size

                # run in torch.inference_mode - Use collective_rpc to run _dummy_run on all workers
                # This ensures all workers in the TP group participate in collective operations
                block_size = llm.llm_engine.cache_config.block_size
                # ceil
                ctx_len_per_sequence = int(
                    math.ceil(total_ctx_len / batch_size))
                max_num_blocks_per_seq = (ctx_len_per_sequence + block_size -
                                          1) // block_size
                block_tables = torch.randint(
                    0,
                    128, (batch_size, max_num_blocks_per_seq),
                    dtype=torch.int32)

                # make it List[List[int]]
                block_tables = block_tables.tolist()

                for clock in clocks:
                    # Check if this datapoint already exists
                    datapoint_key = (clock, batch_size, total_ctx_len)
                    if datapoint_key in existing_data:
                        print(
                            f"Skipping existing datapoint: clock={clock}, batch_size={batch_size}, total_ctx_len={total_ctx_len}"
                        )
                        # add to data
                        data.append({
                            "clock":
                            clock,
                            "batch_size":
                            batch_size,
                            "time_taken":
                            existing_data[datapoint_key]['time_taken'],
                            "energy_consumption":
                            existing_data[datapoint_key]['energy_consumption'],
                            "valid":
                            1,
                            "total_ctx_len":
                            total_ctx_len
                        })
                        continue

                    set_gpu_clock_wrapper(clock)

                    # # Test drive
                    if not args.is_idle:
                        with torch.inference_mode():
                            runner_func(prefill_size, decode_size,
                                        total_ctx_len, block_tables)

                    time.sleep(0.1)

                    # Timer here
                    # tqdm
                    res = []
                    time_taken = []
                    power_consumption = []

                    # Measure energy consumption across all GPUs in the tensor parallel group
                    start_power_per_gpu = []
                    for gpu_idx in gpu_indices:
                        start_power_per_gpu.append(
                            nvmlDeviceGetTotalEnergyConsumption(
                                nvmlDeviceGetHandleByIndex(gpu_idx)))
                    start_time = time.perf_counter()

                    if batch_size < 1024 and not args.is_idle:
                        run_time = 1
                    else:
                        run_time = RUN_TIME
                    num_trials, time_taken = run_until_timeout(
                        runner_func, run_time, prefill_size, decode_size,
                        total_ctx_len, block_tables)

                    if num_trials == 0:
                        print(
                            f"Time exceeded for {batch_size} and clock {clock}"
                        )
                        continue

                    print(
                        f"Batch size: {batch_size}, Clock: {clock}, Num trials: {num_trials}, Time taken: {time_taken}"
                    )

                    torch.cuda.synchronize()
                    end_power_per_gpu = []
                    for gpu_idx in gpu_indices:
                        end_power_per_gpu.append(
                            nvmlDeviceGetTotalEnergyConsumption(
                                nvmlDeviceGetHandleByIndex(gpu_idx)))

                    avg_time_taken = time_taken / num_trials
                    # Calculate total energy consumption across all GPUs
                    total_energy_consumption = sum(
                        end_power_per_gpu[i] - start_power_per_gpu[i]
                        for i in range(len(gpu_indices)))
                    avg_energy_consumption = total_energy_consumption / num_trials
                    # if any of res == None, then don't add to data
                    valid = 1
                    if None in res:
                        print(
                            f"Time exceeded for {batch_size} and clock {clock}"
                        )
                        valid = 0
                    data.append({
                        "clock": clock,
                        "batch_size": batch_size,
                        "time_taken": avg_time_taken,
                        "energy_consumption": avg_energy_consumption,
                        "valid": valid,
                        "total_ctx_len": total_ctx_len
                    })

        # nvtx total pop
        torch.cuda.cudart().cudaProfilerStop()

        # Make pandas dataframe
        df = pd.DataFrame(data)

        print(df)

        if not args.is_idle:
            file_name = f"offline_profile_results/dvfs_profile_{gpu_name}_{model_name}_tp{tp_size}_pp{pp_size}_one_ctx.csv"
        else:
            file_name = f"offline_profile_results/dvfs_profile_{gpu_name}_idle.csv"

        print(f"Saving results to: {file_name}")
        print(f"Total GPUs monitored: {len(gpu_indices)}")
        print(f"GPU indices: {gpu_indices}")
        df.to_csv(file_name, index=False)

    print("Running Test")
    run_test(llm)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default=None,
        help=("path to save the pytorch profiler output. Can be visualized "
              "with ui.perfetto.dev or Tensorboard."),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize responses (i.e. do not include "
              "detokenization time in the latency measurement)"),
    )

    parser.add_argument('--is-idle', action='store_true')

    # eager mode

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
