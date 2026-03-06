#!/usr/bin/env python3
"""
Example script demonstrating how to use the overhead analyzer
to benchmark S1 and S2 scheduling algorithms.
"""

import os
import sys
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName

from vllm.v1.core.sched.energy_model import EnergySimulator
from vllm.v1.core.sched.overhead_analysis import (
    benchmark_s1_algorithm,
    benchmark_s2_algorithm,
    OverheadAnalyzer,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def main():
    """Main function to run overhead analysis."""
    
    # Configuration
    model_name_clean = "Qwen_Qwen2.5-32B"
    tp_size = 1
    pp_size = 4
    
    # Initialize NVML to get GPU name
    nvmlInit()
    gpu_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(0))
    
    # Construct profile path
    profile_dir = "offline_profile_results"
    profile_file = f"dvfs_profile_{gpu_name}_{model_name_clean}_tp{tp_size}_pp{pp_size}_one.csv"
    profile_path = os.path.join(os.getcwd(), profile_dir, profile_file)
    
    # Check if profile exists
    if not os.path.exists(profile_path):
        logger.error(f"Profile file not found: {profile_path}")
        logger.info("Please ensure you have the required offline profile CSV.")
        logger.info("Expected format: dvfs_profile_<GPU>_<MODEL>_tp<TP>_pp<PP>_one.csv")
        return 1
    
    # Initialize energy simulator
    logger.info(f"Loading energy profile from: {profile_path}")
    try:
        energy_simulator = EnergySimulator(
            profiling_csv_path=profile_path,
            num_pp=pp_size,
            num_tp=tp_size,
            gpu_name=gpu_name,
            model_name=model_name_clean
        )
        logger.info("✓ Energy simulator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize energy simulator: {e}")
        return 1
    
    # ===================================================================
    # Example 1: Quick benchmark with default settings
    # ===================================================================
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Quick Benchmark (10 iterations)")
    logger.info("="*70)
    
    s1_analyzer = benchmark_s1_algorithm(
        energy_simulator=energy_simulator,
        pp_size=pp_size,
        num_iterations=10,
        num_prefill_tokens_range=[512, 1024, 2048],
        num_running_reqs_range=[0, 8]
    )
    
    logger.info("\nS1 Results:")
    for stats in s1_analyzer.get_all_stats():
        logger.info(f"  {stats.function_name}: {stats.avg_time*1000:.3f}ms avg")
    
    # ===================================================================
    # Example 2: Detailed benchmark with more iterations
    # ===================================================================
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Detailed Benchmark (50 iterations)")
    logger.info("="*70)
    
    # S1 with more comprehensive settings
    s1_analyzer_detailed = benchmark_s1_algorithm(
        energy_simulator=energy_simulator,
        pp_size=pp_size,
        num_iterations=50,
        num_prefill_tokens_range=[128, 256, 512, 1024, 2048, 4096],
        num_running_reqs_range=[0, 4, 8, 16, 32]
    )
    
    logger.info("\nS1 Detailed Results:")
    s1_analyzer_detailed.print_summary()
    
    # S2 benchmark
    logger.info("\n" + "="*70)
    logger.info("S2 Algorithm Benchmark")
    logger.info("="*70)
    
    s2_analyzer = benchmark_s2_algorithm(
        energy_simulator=energy_simulator,
        pp_size=pp_size,
        num_iterations=50,
        num_running_reqs_range=[4, 8, 16, 32],
        context_lengths_range=[512, 1024, 2048]
    )
    
    s2_analyzer.print_summary()
    
    # ===================================================================
    # Example 3: Aggregate statistics across all configurations
    # ===================================================================
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Aggregate Statistics")
    logger.info("="*70)
    
    s1_all_stats = s1_analyzer_detailed.get_all_stats()
    s2_all_stats = s2_analyzer.get_all_stats()
    
    # Calculate overall statistics
    s1_times = [s.avg_time for s in s1_all_stats]
    s2_times = [s.avg_time for s in s2_all_stats]
    
    logger.info("\nS1 (Prefill) Algorithm:")
    logger.info(f"  Total configurations tested: {len(s1_all_stats)}")
    logger.info(f"  Overall average: {sum(s1_times)/len(s1_times)*1000:.3f}ms")
    logger.info(f"  Best case (min): {min(s1_times)*1000:.3f}ms")
    logger.info(f"  Worst case (max): {max(s1_times)*1000:.3f}ms")
    logger.info(f"  Standard deviation: {(sum((t - sum(s1_times)/len(s1_times))**2 for t in s1_times) / len(s1_times))**0.5 * 1000:.3f}ms")
    
    logger.info("\nS2 (Decode) Algorithm:")
    logger.info(f"  Total configurations tested: {len(s2_all_stats)}")
    logger.info(f"  Overall average: {sum(s2_times)/len(s2_times)*1000:.3f}ms")
    logger.info(f"  Best case (min): {min(s2_times)*1000:.3f}ms")
    logger.info(f"  Worst case (max): {max(s2_times)*1000:.3f}ms")
    logger.info(f"  Standard deviation: {(sum((t - sum(s2_times)/len(s2_times))**2 for t in s2_times) / len(s2_times))**0.5 * 1000:.3f}ms")
    
    # ===================================================================
    # Example 4: Identify most expensive configurations
    # ===================================================================
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Most Expensive Configurations")
    logger.info("="*70)
    
    # Sort S1 by average time
    s1_sorted = sorted(s1_all_stats, key=lambda s: s.avg_time, reverse=True)
    logger.info("\nTop 5 slowest S1 configurations:")
    for i, stats in enumerate(s1_sorted[:5], 1):
        logger.info(f"  {i}. {stats.function_name}: {stats.avg_time*1000:.3f}ms")
    
    # Sort S2 by average time
    s2_sorted = sorted(s2_all_stats, key=lambda s: s.avg_time, reverse=True)
    logger.info("\nTop 5 slowest S2 configurations:")
    for i, stats in enumerate(s2_sorted[:5], 1):
        logger.info(f"  {i}. {stats.function_name}: {stats.avg_time*1000:.3f}ms")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Overhead analysis completed successfully!")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


