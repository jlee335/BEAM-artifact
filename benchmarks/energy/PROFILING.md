# Profiling System: How Metrics Are Collected

This document explains how the custom profiling system works in this vLLM fork — what triggers metric collection, what files are produced, and where they land.

---

## Overview

Metric collection is a two-layer system triggered by the benchmark client:

```
benchmark_script.py (--profile)
    → POST /start_profile  →  EnergyLogger.start()        [worker]
                           →  Scheduler.start_profile()   [scheduler]
    [ requests run ]
    → POST /stop_profile   →  EnergyLogger.stop()         writes CSV files
                           →  Scheduler.stop_profile()    writes CSV files
```

---

## Prerequisites

Both the **server** and the **client** must be configured correctly.

### Server side

The server must be launched with:
1. `VLLM_TORCH_PROFILER_DIR=<some_dir>` — enables the `/start_profile` and `/stop_profile` HTTP endpoints (see `vllm/entrypoints/openai/api_server.py:1285`). Without this env var, the endpoints are not registered and all profiling calls will 404.
2. `--custom-profiler` flag — enables `EnergyLogger` inside the GPU worker (`vllm/v1/worker/gpu_worker.py`).

Example (from `utils.sh` `run_experiment_template()`):
```bash
VLLM_TORCH_PROFILER_DIR=${config_dir} \
vllm serve ${model_name} \
    --custom-profiler \
    ...
```

### Client side

`benchmark_script.py` must be invoked with `--profile`:
```bash
python3 benchmark_script.py \
    --profile \
    --result-dir ${result_dir} \
    ...
```

When `--profile` is present, the script calls `POST /start_profile` (passing `result_dir` as `output_dir`) before sending any requests, and `POST /stop_profile` after all requests complete.

---

## Layer 1 — Worker-side: EnergyLogger

**Source:** `vllm/v1/worker/gpu_worker.py` — class `EnergyLogger`

Activated when the server receives `POST /start_profile`. The `output_dir` from the request body overrides `VLLM_TORCH_PROFILER_DIR` as the write destination.

### Output files (one set per GPU rank)

| File | Columns | Description |
|------|---------|-------------|
| `batch_log_GPU_{rank}.csv` | step, phase, elapsed_time, batch_size, gpu_clock, num_reqs, start_time, current_time, curr_energy | Per-step batch execution info |
| `dvfs_log_GPU_{rank}.csv` | current_time, gpu_clock, max_batch_size, expected_max_running_time | DVFS clock change events |
| `swap_in_log_GPU_{rank}.csv` | curr_time, num_blocks, time_taken | KV cache swap-in events |
| `swap_out_log_GPU_{rank}.csv` | curr_time, num_blocks, time_taken | KV cache swap-out events |
| `copy_log_GPU_{rank}.csv` | curr_time, num_blocks, time_taken | KV cache copy events |
| `wait_for_layer_load_log_GPU_{rank}.csv` | invoked_time, layer_name, time_taken_seconds | Layer load latency |
| `start_load_kv_log_GPU_{rank}.csv` | invoked_time, time_taken_seconds | KV load start timing |

All logs are collected in memory during the run and flushed to disk atomically when `POST /stop_profile` is received.

---

## Layer 2 — Scheduler-side: S1S2DecisionLogger

**Source:** `vllm/v1/core/sched/scheduler.py` — class `S1S2DecisionLogger`

Activated simultaneously with Layer 1 (same `/start_profile` call triggers `Scheduler.start_profile(output_dir)`). Writes to the same `output_dir`.

### Output files

| File | Columns | Description |
|------|---------|-------------|
| `s1_log.csv` | timestamp, optimal_chunk_size | S1 prefill scheduling decisions |
| `s2_log.csv` | timestamp, optimal_num_microbatches | S2 decode scheduling decisions |
| `kv_usage_log.csv` | timestamp, kv_usage | KV cache occupancy (logged every step) |
| `s1_exec_time_log.csv` | timestamp, execution_time | S1 algorithm wall-clock overhead |
| `s2_exec_time_log.csv` | timestamp, execution_time | S2 algorithm wall-clock overhead |
| `schedule_exec_time_log.csv` | timestamp, execution_time | Full `schedule()` call overhead |

`s1_log.csv` and `s2_log.csv` are only populated when `--use-s1` / `--use-s2` is active on the server.

---

## External Monitor: gpu_energy_monitor.py

Runs as a **separate background process** (started by `run_single_benchmark()` in `utils.sh`). It polls `nvidia-smi` / NVML independently and writes:

- `gpu_energy_and_frequency_{model_name_cleaned}.csv` — timestamp, per-GPU joules and clock frequency

This file is produced regardless of the `--profile` flag and does not go through the profiler API.

---

## Expected Directory Layout

```
${config_dir}/                          ← VLLM_TORCH_PROFILER_DIR (enables /start_profile)
├── server.log
├── experiment_info.txt
├── slo_info.txt
└── traced_dataset/                     ← --result-dir passed to benchmark_script.py
    ├── batch_log_GPU_0.csv             ← Layer 1 (EnergyLogger)
    ├── batch_log_GPU_1.csv             ← (one per GPU if multi-GPU)
    ├── dvfs_log_GPU_0.csv
    ├── swap_in_log_GPU_0.csv
    ├── swap_out_log_GPU_0.csv
    ├── copy_log_GPU_0.csv
    ├── wait_for_layer_load_log_GPU_0.csv
    ├── start_load_kv_log_GPU_0.csv
    ├── s1_log.csv                      ← Layer 2 (S1S2DecisionLogger)
    ├── s2_log.csv
    ├── kv_usage_log.csv
    ├── s1_exec_time_log.csv
    ├── s2_exec_time_log.csv
    ├── schedule_exec_time_log.csv
    ├── gpu_energy_and_frequency_*.csv  ← External monitor
    └── *.json                          ← Benchmark results (latency, throughput)
```

---

## Common Pitfalls

### `/start_profile` returns 404
The server was not started with `VLLM_TORCH_PROFILER_DIR` set. The profiler endpoints are conditionally registered only when this env var is present.

### Layer 1/2 CSV files are missing but `gpu_energy_and_frequency_*.csv` exists
The benchmark client was invoked **without** `--profile`. The external GPU monitor runs unconditionally, but the profiler API (start/stop) is only called when `--profile` is passed. Add `--profile` to the benchmark command (see `run_benchmark_config()` in `utils.sh`).

### `EnergyLogger`: `Output directory is not set` error
`start_profile` was called without `output_dir` in the request body, and the logger's `output_dir` was never set. Ensure the client passes `output_dir` (which `benchmark_script.py` does automatically via `args.result_dir`).
