# Use S1 or S2 overhead analysis
import os
import time
import math
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from vllm.logger import init_logger
from vllm.v1.core.sched.energy_model import EnergySimulator
from vllm.v1.request import Request

from pynvml import (nvmlInit, nvmlDeviceGetName, nvmlDeviceGetClock, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceResetGpuLockedClocks,
                    nvmlDeviceSetGpuLockedClocks)

logger = init_logger(__name__)

nvmlInit()

def reset_gpu_clock(rank):
    device_handle = nvmlDeviceGetHandleByIndex(rank)
    nvmlDeviceResetGpuLockedClocks(device_handle)
    # threading.Thread(target=nvmlDeviceResetGpuLockedClocks,
    #                 args=(device_handle, )).start()
    
def lock_gpu_clock(rank, new_clock):
    device_handle = nvmlDeviceGetHandleByIndex(rank)
    nvmlDeviceSetGpuLockedClocks(device_handle, new_clock, new_clock)
    # threading.Thread(target=nvmlDeviceSetGpuLockedClocks,
    #                 args=(device_handle, new_clock, new_clock)).start()

def apply_dvfs(new_clock: int, num_ranks: int):
    """
    Apply DVFS to the GPU clocks.
    """
    global CURRENT_CLOCK

    # Parse CUDA_VISIBLE_DEVICES to identify which GPUs to control
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        # Handle cases like "0,1"
        target_ranks = [int(x.strip()) for x in cuda_visible_devices.split(',') if x.strip()]
        # We only take the first num_ranks devices
    else:
        target_ranks = list(range(num_ranks))
    
    # logger.info(f"APPLY DVFS: {target_ranks} (physical ranks) to {new_clock} MHz")
    
    if CURRENT_CLOCK == new_clock:
        return
    
    if new_clock == None: # Reset
        for rank in target_ranks:
            reset_gpu_clock(rank)
    else:
        for rank in target_ranks:
            lock_gpu_clock(rank, new_clock)
    
    CURRENT_CLOCK = new_clock



def beam_schedule_tbt_based_s1(self, tbt_slo: float, ttft_slo: float, num_waiting_prefill_tokens: int, use_s1_dvfs_only: bool, new_requests: list[Request], disagg_mode = False):
    # Scheduling algorithm that caps TBT.
    
    min_energy_chunk_size = None
    min_energy_clock = None
    min_energy = float('inf')
    min_ttft = float('inf')
    
    has_ttft_adhering_choices = False
    has_tbt_adhering_choices = False
    
    chunk_sizes = self.energy_simulator.available_chunk_sizes
    clocks = self.energy_simulator.available_clocks
    
    # if use_s1_dvfs_only, we fix chunk_size choice to 256
    # When measuring TTFT, we also need to consider queuing time.
    now = time.time()
    # For each new_requests, compute max queuing time.
    max_queuing_time = float('-inf')
    for req in new_requests:
        queuing_time = now - req.arrival_time
        if queuing_time > max_queuing_time:
            max_queuing_time = queuing_time
    max_queuing_time = max(max_queuing_time, 0.0)
    # logger.info(f"Max queuing time: {max_queuing_time}")

    time_since_last_prefill = time.monotonic() - self.last_prefill_timestamp
    # If so, limit chunk size choices to 1 above, same, or 1 below.
    # Drastic chunk-size changes create bubbles.
    if time_since_last_prefill < self.last_prefill_ttft:
        # Get current chunk size
        current_chunk_size = self.max_num_scheduled_tokens
        # Find the index of current chunk size in available sizes
        available_chunks = self.energy_simulator.available_chunk_sizes
        if current_chunk_size in available_chunks:
            current_idx = available_chunks.index(current_chunk_size)
            # Limit to adjacent chunk sizes (1 below, same, 1 above)
            start_idx = max(0, current_idx - 1)
            end_idx = min(len(available_chunks), current_idx + 2)
            chunk_sizes = available_chunks[start_idx:end_idx]
            # logger.info(f"Prefill in progress: limiting chunk_sizes from {available_chunks} to {chunk_sizes}")

    if use_s1_dvfs_only:
        chunk_sizes = [256]
        

    for chunk_size in chunk_sizes:
        for clock in clocks:
            per_chunk_energy = self.energy_simulator.get_energy(chunk_size, clock)
            per_chunk_latency = self.energy_simulator.get_time_taken(chunk_size, clock)
            avg_decodes_per_chunk = len(self.running) // self.pp_size
            num_chunks_floor = math.floor(num_waiting_prefill_tokens / (chunk_size - avg_decodes_per_chunk)) * self.pp_size
            num_ttft_chunks_floor = math.floor(num_waiting_prefill_tokens / (chunk_size - avg_decodes_per_chunk)) + (self.pp_size - 1)
            
            
            if chunk_size >= num_waiting_prefill_tokens:
                # Single chunk can handle entire request
                per_chunk_energy = self.energy_simulator.get_energy(num_waiting_prefill_tokens, clock)
                per_chunk_latency = self.energy_simulator.get_time_taken(num_waiting_prefill_tokens, clock)
                
                tbt = (per_chunk_latency + self.energy_simulator.scheduling_overhead) * self.pp_size
                ttft = (per_chunk_latency + self.energy_simulator.scheduling_overhead) * self.pp_size
                ttft = self._previous_prefill_interrupt_moderation(ttft, per_chunk_latency) + max_queuing_time
                energy = per_chunk_energy * self.pp_size
                
                ttft_adhering = ttft <= ttft_slo
                if disagg_mode: 
                    tbt_adhering = True
                else:
                    tbt_adhering = tbt <= tbt_slo
                
            else:
                # Multiple chunks needed
                num_leftover_tokens = num_waiting_prefill_tokens % chunk_size
                leftover_chunk_energy = self.energy_simulator.get_energy(num_leftover_tokens, clock)
                leftover_chunk_latency = self.energy_simulator.get_time_taken(num_leftover_tokens, clock)
                
                tbt = (per_chunk_latency + self.energy_simulator.scheduling_overhead) * self.pp_size
                ttft = num_ttft_chunks_floor * (per_chunk_latency + self.energy_simulator.scheduling_overhead) + leftover_chunk_latency
                ttft = self._previous_prefill_interrupt_moderation(ttft, per_chunk_latency) + max_queuing_time
                energy = per_chunk_energy * num_chunks_floor + leftover_chunk_energy * self.pp_size
                                    
                ttft_adhering = ttft <= ttft_slo
                if disagg_mode: 
                    tbt_adhering = True
                else:
                    tbt_adhering = tbt <= tbt_slo
            
            if ttft_adhering:
                has_ttft_adhering_choices = True
            if tbt_adhering:
                has_tbt_adhering_choices = True
                
            if ttft_adhering and tbt_adhering and energy < min_energy:
                min_energy = energy
                min_energy_chunk_size = chunk_size
                min_energy_clock = clock
                min_ttft = ttft
                min_tbt = tbt

    self.last_prefill_timestamp = time.monotonic()
    if min_energy_chunk_size is None or min_energy_clock is None:
        # If no tbt_adhering, use chunk-size 128, if no ttft_adhering, choose 512
        if not has_tbt_adhering_choices:
            logger.info("!!! No tbt_adhering choices, using chunk-size 128")
            chunk_size = 128
        if not has_ttft_adhering_choices:
            logger.info("!!! No ttft_adhering choices, using chunk-size 256")
            chunk_size = 256 # Default size for vLLM. Has precedence.
        
        # Pick default chunk x clock
        num_ttft_chunks = math.ceil(num_waiting_prefill_tokens / chunk_size) + (self.pp_size - 1)
        min_ttft = num_ttft_chunks * (self.energy_simulator.get_time_taken(chunk_size, self.energy_simulator.default_clock_high) + self.energy_simulator.scheduling_overhead)
        self.last_prefill_ttft = min_ttft
        return chunk_size, self.energy_simulator.default_clock_high

    else:
        self.last_prefill_ttft = min_ttft
        return min_energy_chunk_size, min_energy_clock

def beam_schedule_tbt_based_s2(self, tbt_slo: float, num_running_reqs: int):
    # Scheduling algorithm that caps TBT. In decode-only traces.
    min_energy_num_microbatches = None
    min_energy_clock = None
    min_energy = float('inf')

    clocks = self.energy_simulator.available_clocks
    
    # Get total context length of all running requests
    total_ctx_len = 0
    for req in self.running:
        total_ctx_len += req.num_tokens_with_spec
    # Iterate all num_microbatches x clock to find the optimal num_microbatches x clock that can cap TBT.
    for num_microbatches in range(1, self.pp_size + 1):
        for clock in clocks:
            
            ctx_len_per_microbatch = total_ctx_len / num_microbatches
            
            # Get per-microbatch energy and latency
            num_reqs_per_microbatch = math.ceil(num_running_reqs / num_microbatches)
            # TBT is bound to individual microbatch
            tbt = (self.energy_simulator.get_time_taken(num_reqs_per_microbatch, clock, ctx_len_per_microbatch) + self.energy_simulator.scheduling_overhead_decode_only) * self.pp_size
            # Energy is bound to all microbatches
            energy = self.energy_simulator.get_energy(num_reqs_per_microbatch, clock) * num_microbatches
            
            if tbt <= tbt_slo and energy < min_energy:
                min_energy = energy
                min_energy_num_microbatches = num_microbatches
                min_energy_clock = clock

    if min_energy_num_microbatches is None or min_energy_clock is None:
        return self.pp_size, self.energy_simulator.default_clock_high

    return min_energy_num_microbatches, min_energy_clock

if __name__ == "__main__":
    
    tbt_slo = 0.4
    ttft_slo = 1
    num_waiting_prefill_tokens = 32
    
    NUM_TRIALS = 200
    
    NUM_PP = 4
    NUM_TP = 1
    MODEL_NAME = "Qwen/Qwen2.5-32B"
    GPU_NAME = "NVIDIA RTX A6000"
    # GPU_NAME = "NVIDIA A100-SXM4-80GB"
    # MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
    MODEL_NAME = MODEL_NAME.replace("/", "_")

    # Create profiling CSV path
    model_name_clean = MODEL_NAME.replace("/", "_")
    gpu_name = GPU_NAME

    # (hj) Load offline profile data from csv file. In benchmarks/energy/offline_profile_results/
    file_name_working = f"dvfs_profile_{gpu_name}_{model_name_clean}_tp{NUM_TP}_pp{NUM_PP}_one.csv"
    dir = "offline_profile_results"  # TODO: Change this to something more sensible
    file_full_path_working = f"/workspace/disagg/energy-inf-v1-disagg/benchmarks/energy/{dir}/{file_name_working}" 
    
    # Initialize EnergySimulator
    print(f"Initializing EnergySimulator with profiling data: {file_full_path_working}")
    energy_simulator = EnergySimulator(
        profiling_csv_path=file_full_path_working,
        num_pp=NUM_PP,
        num_tp=NUM_TP,
        gpu_name=GPU_NAME,
        model_name=MODEL_NAME
    )
    
    # Print chunk-size, clocks
    print(f"Chunk-size: {energy_simulator.available_chunk_sizes}")
    print(f"Clocks: {energy_simulator.available_clocks}")
    
    # Search space size = len(chunk-size) * len(clocks)
    search_space_size = len(energy_simulator.available_chunk_sizes) * len(energy_simulator.available_clocks)
    print(f"Search space size: {search_space_size}")
    
    # Create a mock scheduler object with required attributes
    class MockScheduler:
        def __init__(self, energy_simulator, pp_size):
            self.energy_simulator = energy_simulator
            self.pp_size = pp_size
            self.running = []  # List of running requests
            self.max_num_scheduled_tokens = 256
            self.last_prefill_timestamp = time.monotonic()
            self.last_prefill_ttft = 0.0
            
        def _previous_prefill_interrupt_moderation(self, ttft, per_chunk_latency):
            # Simple moderation logic - just return ttft as is
            return ttft
    
    # Create mock scheduler
    mock_scheduler = MockScheduler(energy_simulator, NUM_PP)
    
    # Create mock request for testing
    class MockRequest:
        def __init__(self, arrival_time, num_tokens=None):
            self.arrival_time = arrival_time
            self.num_tokens_with_spec = num_tokens if num_tokens else 1024
    
    # Test S1 scheduling overhead
    print("\n" + "="*70)
    print("Testing S1 Scheduling (Prefill with TBT/TTFT constraints)")
    print("="*70)
    
    s1_timings = []
    for i in range(NUM_TRIALS):
        # Create new mock requests
        new_requests = [MockRequest(time.time() - 0.001)]
        
        # Bind the function to the mock scheduler
        start = time.perf_counter()
        chunk_size, clock = beam_schedule_tbt_based_s1(
            mock_scheduler,
            tbt_slo=tbt_slo,
            ttft_slo=ttft_slo,
            num_waiting_prefill_tokens=num_waiting_prefill_tokens,
            use_s1_dvfs_only=False,
            new_requests=new_requests,
            disagg_mode=False
        )
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Convert to ms
        s1_timings.append(elapsed)
        
        if i == 0:
            print(f"First run: chunk_size={chunk_size}, clock={clock}, time={elapsed:.4f}ms")
    
    s1_avg = sum(s1_timings) / len(s1_timings)
    s1_min = min(s1_timings)
    s1_max = max(s1_timings)
    s1_p50 = sorted(s1_timings)[len(s1_timings)//2]
    s1_p95 = sorted(s1_timings)[int(len(s1_timings)*0.95)]
    s1_p99 = sorted(s1_timings)[int(len(s1_timings)*0.99)]
    
    print(f"\nS1 Scheduling Overhead Statistics ({NUM_TRIALS} trials):")
    print(f"  Average: {s1_avg:.4f} ms")
    print(f"  Min:     {s1_min:.4f} ms")
    print(f"  Max:     {s1_max:.4f} ms")
    print(f"  P50:     {s1_p50:.4f} ms")
    print(f"  P95:     {s1_p95:.4f} ms")
    print(f"  P99:     {s1_p99:.4f} ms")
    
    # Test S2 scheduling overhead
    print("\n" + "="*70)
    print("Testing S2 Scheduling (Decode-only with TBT constraints)")
    print("="*70)
    
    # Add some running requests for S2 testing
    num_running_reqs = 8
    mock_scheduler.running = [MockRequest(time.time(), num_tokens=512) for _ in range(num_running_reqs)]
    
    s2_timings = []
    for i in range(NUM_TRIALS):
        start = time.perf_counter()
        num_microbatches, clock = beam_schedule_tbt_based_s2(
            mock_scheduler,
            tbt_slo=tbt_slo,
            num_running_reqs=num_running_reqs
        )
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Convert to ms
        s2_timings.append(elapsed)
        
        if i == 0:
            print(f"First run: num_microbatches={num_microbatches}, clock={clock}, time={elapsed:.4f}ms")
    
    s2_avg = sum(s2_timings) / len(s2_timings)
    s2_min = min(s2_timings)
    s2_max = max(s2_timings)
    s2_p50 = sorted(s2_timings)[len(s2_timings)//2]
    s2_p95 = sorted(s2_timings)[int(len(s2_timings)*0.95)]
    s2_p99 = sorted(s2_timings)[int(len(s2_timings)*0.99)]
    
    print(f"\nS2 Scheduling Overhead Statistics ({NUM_TRIALS} trials):")
    print(f"  Average: {s2_avg:.4f} ms")
    print(f"  Min:     {s2_min:.4f} ms")
    print(f"  Max:     {s2_max:.4f} ms")
    print(f"  P50:     {s2_p50:.4f} ms")
    print(f"  P95:     {s2_p95:.4f} ms")
    print(f"  P99:     {s2_p99:.4f} ms")
    
    # Summary comparison
    print("\n" + "="*70)
    print("Summary Comparison")
    print("="*70)
    print(f"S1 vs S2 Average Overhead: {s1_avg:.4f} ms vs {s2_avg:.4f} ms")
    print(f"S1 vs S2 P95 Overhead:     {s1_p95:.4f} ms vs {s2_p95:.4f} ms")
    print(f"S1 vs S2 P99 Overhead:     {s1_p99:.4f} ms vs {s2_p99:.4f} ms")
    
    # Test DVFS overhead for different numbers of GPU ranks
    print("\n" + "="*70)
    print("Testing DVFS (Dynamic Voltage and Frequency Scaling) Overhead")
    print("="*70)
    
    # Get available clocks from energy simulator
    available_clocks = sorted(energy_simulator.available_clocks)
    print(f"Available clocks: {available_clocks}")
    
    # Define clock transition scenarios
    clock_scenarios = []
    
    # Scenario 1: Reset -> Clock (lowest)
    clock_scenarios.append(('reset_to_low', None, available_clocks[0], f"Reset -> {available_clocks[0]} MHz"))
    
    if len(available_clocks) >= 2:
        # Scenario 2: Lowest -> Highest
        clock_scenarios.append(('low_to_high', available_clocks[0], available_clocks[-1], 
                               f"{available_clocks[0]} -> {available_clocks[-1]} MHz (Low->High)"))
        
        # Scenario 3: Highest -> Lowest
        clock_scenarios.append(('high_to_low', available_clocks[-1], available_clocks[0],
                               f"{available_clocks[-1]} -> {available_clocks[0]} MHz (High->Low)"))
    
    if len(available_clocks) >= 3:
        # Scenario 4: Adjacent clocks (low to low) - first two lowest clocks
        clock_scenarios.append(('adj_low_to_low', available_clocks[0], available_clocks[1],
                               f"{available_clocks[0]} -> {available_clocks[1]} MHz (Adjacent Low)"))
        
        # Scenario 5: Adjacent clocks (high to high) - last two highest clocks
        clock_scenarios.append(('adj_high_to_high', available_clocks[-2], available_clocks[-1],
                               f"{available_clocks[-2]} -> {available_clocks[-1]} MHz (Adjacent High)"))
    
    print(f"\nTesting {len(clock_scenarios)} clock transition scenarios:")
    for i, (name, from_clock, to_clock, desc) in enumerate(clock_scenarios, 1):
        print(f"  {i}. {desc}")
    
    # Always test with 4 GPU ranks
    TEST_NUM_RANKS = 4
    print(f"\nTesting DVFS with {TEST_NUM_RANKS} GPU ranks")
    
    # Declare global before any assignment
    global CURRENT_CLOCK
    
    dvfs_results = {}
    dvfs_raw_timings = {}  # Store raw timings for histogram generation
    
    # Calculate statistics helper
    def calc_stats(timings):
        return {
            'avg': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings),
            'p50': sorted(timings)[len(timings)//2],
            'p95': sorted(timings)[int(len(timings)*0.95)],
            'p99': sorted(timings)[int(len(timings)*0.99)],
        }
    
    print(f"\n--- Testing DVFS with {TEST_NUM_RANKS} GPU rank(s) ---")
    
    for scenario_name, from_clock, to_clock, description in clock_scenarios:
        print(f"  Testing: {description}")
        timings = []
        
        for i in range(NUM_TRIALS):
            # Set initial state
            CURRENT_CLOCK = from_clock
            
            # Set to from_clock
            apply_dvfs(from_clock, TEST_NUM_RANKS)
            time.sleep(0.1)
            
            # Measure transition
            start = time.perf_counter()
            apply_dvfs(to_clock, TEST_NUM_RANKS)
            end = time.perf_counter()
            elapsed = (end - start) * 1000  # Convert to ms
            timings.append(elapsed)
        
        # Store statistics and raw timings for this scenario
        dvfs_results[scenario_name] = calc_stats(timings)
        dvfs_raw_timings[scenario_name] = timings
    
    # Print statistics
    print(f"\n  DVFS Statistics for {TEST_NUM_RANKS} GPU rank(s) ({NUM_TRIALS} trials each):")
    for scenario_name, from_clock, to_clock, description in clock_scenarios:
        stats = dvfs_results[scenario_name]
        print(f"    {description}:")
        print(f"    Avg: {stats['avg']:.4f} ms, "
              f"    P50: {stats['p50']:.4f} ms, "
              f"    P95: {stats['p95']:.4f} ms, "
              f"    P99: {stats['p99']:.4f} ms")
    
    # Summary table
    print("\n" + "="*70)
    print(f"DVFS Overhead Summary ({TEST_NUM_RANKS} GPU Ranks)")
    print("="*70)
    print(f"{'Operation':<40} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 95)
    for scenario_name, from_clock, to_clock, description in clock_scenarios:
        stats = dvfs_results[scenario_name]
        # Shorten description for table
        short_desc = description.split(' (')[0] if ' (' in description else description
        print(f"{short_desc:<40} {stats['avg']:<12.4f} {stats['p50']:<12.4f} {stats['p95']:<12.4f} {stats['p99']:<12.4f}")
    
    # Generate and save histograms
    print("\n" + "="*70)
    print("Generating DVFS Overhead Histograms")
    print("="*70)
    
    # Create output directory for histograms
    output_dir = Path("dvfs_overhead_histograms")
    output_dir.mkdir(exist_ok=True)
    
    # Generate histograms for each scenario
    for scenario_name, from_clock, to_clock, description in clock_scenarios:
        timings = dvfs_raw_timings[scenario_name]
        stats = dvfs_results[scenario_name]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(timings, bins=100, alpha=0.7, color='blue', edgecolor='black')
        
        # Add vertical lines for statistics
        ax.axvline(stats['avg'], color='red', linestyle='--', linewidth=2, label=f"Avg: {stats['avg']:.3f} ms")
        ax.axvline(stats['p50'], color='green', linestyle='--', linewidth=2, label=f"P50: {stats['p50']:.3f} ms")
        ax.axvline(stats['p95'], color='orange', linestyle='--', linewidth=2, label=f"P95: {stats['p95']:.3f} ms")
        ax.axvline(stats['p99'], color='purple', linestyle='--', linewidth=2, label=f"P99: {stats['p99']:.3f} ms")
        
        # Labels and title
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'DVFS Overhead: {description}\n{TEST_NUM_RANKS} GPU Ranks, {NUM_TRIALS} trials', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with additional stats
        textstr = f'Min: {stats["min"]:.3f} ms\nMax: {stats["max"]:.3f} ms\nStd: {np.std(timings):.3f} ms'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Save figure
        filename = f"dvfs_histogram_{scenario_name}.png"
        filepath = output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {filepath}")
    
    # Create a combined overview plot with all scenarios
    print(f"\n  Creating combined overview...")
    
    num_scenarios = len(clock_scenarios)
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(12, 4 * num_scenarios))
    
    if num_scenarios == 1:
        axes = [axes]
    
    for idx, (scenario_name, from_clock, to_clock, description) in enumerate(clock_scenarios):
        timings = dvfs_raw_timings[scenario_name]
        stats = dvfs_results[scenario_name]
        
        ax = axes[idx]
        
        # Plot histogram
        n, bins, patches = ax.hist(timings, bins=100, alpha=0.7, color='blue', edgecolor='black')
        
        # Add vertical lines for statistics
        ax.axvline(stats['avg'], color='red', linestyle='--', linewidth=2, label=f"Avg: {stats['avg']:.3f} ms")
        ax.axvline(stats['p95'], color='orange', linestyle='--', linewidth=2, label=f"P95: {stats['p95']:.3f} ms")
        ax.axvline(stats['p99'], color='purple', linestyle='--', linewidth=2, label=f"P99: {stats['p99']:.3f} ms")
        
        # Labels and title
        ax.set_xlabel('Latency (ms)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{description}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'DVFS Overhead Overview - {TEST_NUM_RANKS} GPU Ranks', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    overview_filepath = output_dir / f"dvfs_overview_{TEST_NUM_RANKS}ranks.png"
    plt.savefig(overview_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved overview: {overview_filepath}")
    print(f"\nAll histograms saved to: {output_dir.absolute()}")
    
    # Determine if overhead is acceptable
    print("\n" + "="*70)
    print("Overhead Analysis")
    print("="*70)
    scheduling_overhead_budget = 12.0  # ms (0.012s from energy_model)
    dvfs_overhead_budget = energy_simulator.dvfs_delay * 1000  # Convert to ms
    
    print(f"Scheduling overhead budget: {scheduling_overhead_budget:.4f} ms")
    print(f"DVFS overhead budget: {dvfs_overhead_budget:.4f} ms")
    
    print(f"\nS1 P95 overhead meets budget: {s1_p95 < scheduling_overhead_budget}")
    print(f"S2 P95 overhead meets budget: {s2_p95 < scheduling_overhead_budget}")
    
    # Check DVFS overhead across all clock transition scenarios
    # Find worst case P95 across all clock transition scenarios
    dvfs_worst_p95 = max(
        stats['p95'] for stats in dvfs_results.values()
    )
    # Find which scenario is the worst
    worst_scenario = max(
        dvfs_results.items(),
        key=lambda x: x[1]['p95']
    )
    worst_scenario_name = next(
        desc for name, _, _, desc in clock_scenarios if name == worst_scenario[0]
    )
    
    print(f"\nDVFS P95 overhead (worst case, {TEST_NUM_RANKS} ranks): {dvfs_worst_p95:.4f} ms")
    print(f"  Worst scenario: {worst_scenario_name}")
    print(f"DVFS P95 overhead meets budget: {dvfs_worst_p95 < dvfs_overhead_budget}")
    
    all_meet_budget = (s1_p95 < scheduling_overhead_budget and 
                      s2_p95 < scheduling_overhead_budget and
                      dvfs_worst_p95 < dvfs_overhead_budget)
    
    if all_meet_budget:
        print("\n✓ All operations meet their overhead budgets!")
    else:
        print("\n✗ Warning: Some operations exceed their overhead budgets!")
    
    # Reset DVFS state
    CURRENT_CLOCK = None
    try:
        apply_dvfs(None, TEST_NUM_RANKS)
        print("\n[Cleanup] GPU clocks reset to default.")
    except Exception as e:
        print(f"\n[Cleanup] Warning: Could not reset GPU clocks: {e}")