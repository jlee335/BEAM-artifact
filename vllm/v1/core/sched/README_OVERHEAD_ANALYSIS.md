# Overhead Analysis for S1 and S2 Scheduling Algorithms

## Overview

The `overhead_analysis.py` module provides comprehensive benchmarking tools to measure the execution time of the S1 (prefill) and S2 (decode) scheduling algorithms used in the vLLM scheduler.

## Components

### 1. `OverheadStats`
A dataclass that stores timing statistics:
- `num_calls`: Total number of function invocations
- `total_time`: Cumulative execution time
- `min_time`: Minimum execution time
- `max_time`: Maximum execution time
- `avg_time`: Average execution time (computed property)

### 2. `OverheadAnalyzer`
Main class for tracking and analyzing overhead:
- `record(function_name, execution_time)`: Record a single execution
- `get_stats(function_name)`: Get statistics for a specific function
- `get_all_stats()`: Get statistics for all tracked functions
- `print_summary()`: Print formatted summary of all statistics
- `reset()`: Clear all collected statistics

### 3. Benchmark Functions

#### `benchmark_s1_algorithm()`
Benchmarks the S1 (prefill) scheduling algorithm with various configurations:
- **Parameters:**
  - `energy_simulator`: EnergySimulator instance
  - `pp_size`: Pipeline parallel size (default: 4)
  - `num_iterations`: Number of iterations per configuration (default: 100)
  - `num_prefill_tokens_range`: List of prefill token counts to test
  - `num_running_reqs_range`: List of running request counts to test

#### `benchmark_s2_algorithm()`
Benchmarks the S2 (decode) scheduling algorithm with various configurations:
- **Parameters:**
  - `energy_simulator`: EnergySimulator instance
  - `pp_size`: Pipeline parallel size (default: 4)
  - `num_iterations`: Number of iterations per configuration (default: 100)
  - `num_running_reqs_range`: List of running request counts to test
  - `context_lengths_range`: List of context lengths to test

## Usage

### Running the Standalone Benchmark

```bash
# Make sure you're in the project root directory
cd /workspace/disagg/energy-inf-v1-disagg

# Run the overhead analysis
python -m vllm.v1.core.sched.overhead_analysis
```

### Prerequisites

1. **Energy Profile CSV**: The benchmark requires an offline energy profile CSV file located in `offline_profile_results/`. The file should follow the naming convention:
   ```
   dvfs_profile_{GPU_NAME}_{MODEL_NAME}_tp{TP_SIZE}_pp{PP_SIZE}_one.csv
   ```

2. **Environment**: Ensure you have access to a GPU and the necessary NVML libraries installed.

### Integration with Scheduler

The scheduler in `scheduler.py` already logs execution times for S1 and S2 algorithms:

```python
# S1 execution time logging (lines 1495-1500)
s1_start = time.time()
optimal_chunk_size, optimal_clock = self.beam_schedule_tbt_based_s1(...)
s1_end = time.time()
self.s1_s2_logger.log_s1_exec_time(s1_end - s1_start)

# S2 execution time logging (lines 1546-1551)
s2_start = time.time()
optimal_num_microbatches, optimal_clock = self.beam_schedule_tbt_based_s2(...)
s2_end = time.time()
self.s1_s2_logger.log_s2_exec_time(s2_end - s2_start)
```

These logs are saved to CSV files when `stop_profile()` is called:
- `s1_exec_time_log.csv`: S1 algorithm execution times
- `s2_exec_time_log.csv`: S2 algorithm execution times

## Typical Input Values

Based on the scheduler implementation, here are typical input ranges:

### S1 (Prefill) Algorithm
- **tbt_slo**: 0.45 seconds (default from scheduler)
- **ttft_slo**: 4.0 seconds (default from scheduler)
- **num_waiting_prefill_tokens**: 128 - 8192 tokens
- **num_running_reqs**: 0 - 64 concurrent requests
- **use_s1_dvfs_only**: True/False (DVFS-only mode)
- **disagg_mode**: True/False (disaggregated mode)

### S2 (Decode) Algorithm
- **tbt_slo**: 0.45 seconds (default from scheduler)
- **num_running_reqs**: 4 - 128 concurrent requests
- **total_ctx_len**: Sum of all running request context lengths (typically 1K - 100K tokens)

## Output Example

```
============================================================
OVERHEAD ANALYSIS SUMMARY
============================================================
S1_prefill512_running4:
  Calls: 50
  Total: 125.342ms
  Average: 2.507ms
  Min: 2.234ms
  Max: 3.891ms
------------------------------------------------------------
S2_running16_ctx2048:
  Calls: 50
  Total: 45.123ms
  Average: 0.902ms
  Min: 0.845ms
  Max: 1.234ms
------------------------------------------------------------
```

## Performance Insights

### Expected Overhead Ranges (Approximate)

**S1 Algorithm:**
- Small prefills (< 512 tokens, few running): 1-3ms
- Medium prefills (512-2048 tokens): 2-5ms
- Large prefills (> 2048 tokens, many running): 5-15ms

**S2 Algorithm:**
- Small decode batches (< 16 requests): 0.5-2ms
- Medium decode batches (16-32 requests): 1-3ms
- Large decode batches (> 32 requests): 2-5ms

### Factors Affecting Overhead

1. **Search Space Size:**
   - Number of chunk sizes to evaluate (S1)
   - Number of clock frequencies available
   - Pipeline parallel size (affects microbatch options in S2)

2. **Workload Complexity:**
   - Number of prefill tokens
   - Number of running requests
   - Context lengths

3. **System Configuration:**
   - CPU speed
   - Memory access patterns
   - Energy simulator lookup performance

## Notes

- The overhead analysis uses `time.perf_counter()` for high-resolution timing
- All times are reported in milliseconds (ms) for readability
- The benchmark runs multiple iterations to get stable averages
- Mock objects are used to avoid dependencies on actual requests


