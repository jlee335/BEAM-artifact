#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Script to create combined plots showing TPS, TTFT, and TPOT vs chunk size (batch size) 
for each GPU at target clock frequencies.
"""

import argparse
from pathlib import Path
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHUNK_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
MIN_PREFILL_CHUNK_SIZE = 64
REQUEST_LENGTH = 4096

TPOT_SLO = 0.3
TTFT_SLO = 4.0

FIG_SIZE = (4, 8)

def extract_config_from_filename(filename):
    """Extract configuration details from filename"""
    basename = os.path.basename(filename)

    # Parse filename pattern: dvfs_profile_[GPU]_[model]_tp[x]_pp[y]_one.csv
    # Handle cases where GPU names might have spaces
    pattern = r'dvfs_profile_(.+?)_(.+?)_tp(\d+)_pp(\d+)_one\.csv'
    match = re.search(pattern, basename)

    if match:
        gpu = match.group(1)
        model = match.group(2)
        tp = int(match.group(3))
        pp = int(match.group(4))
        return gpu, model, tp, pp

    return None, None, None, None


def sanitize_filename(filename):
    """Sanitize filename by replacing problematic characters"""
    return filename.replace(' ', '_').replace('/', '_').replace(
        '\\',
        '_').replace(':', '_').replace('<', '_').replace('>', '_').replace(
            '|', '_').replace('?', '_').replace('*', '_').replace('"', '_')


def interpolate_time_taken(batch_size, batch_sizes_sorted, time_taken_sorted):
    """
    Interpolate time taken for a given batch size using available data points.
    
    Args:
        batch_size: The batch size to interpolate for
        batch_sizes_sorted: Sorted array of available batch sizes
        time_taken_sorted: Corresponding time taken values
    
    Returns:
        Interpolated time taken for the given batch size
    """
    if batch_size in batch_sizes_sorted:
        # Exact match found
        idx = np.where(batch_sizes_sorted == batch_size)[0][0]
        return time_taken_sorted[idx]
    else:
        # Interpolate/extrapolate using numpy interpolation
        return float(np.interp(batch_size, batch_sizes_sorted, time_taken_sorted,
                              left=time_taken_sorted[0], right=time_taken_sorted[-1]))


def find_optimal_clocks_by_energy(csv_files, TPOT_SLO = TPOT_SLO):
    """
    Find the optimal clock frequency for each GPU x batch-size combination based on minimum energy consumption.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        dict: (GPU type, batch_size) -> optimal clock frequency
    """
    optimal_clocks = {}
    gpu_batch_data = {}  # Store data for analysis
    
    print("Analyzing energy consumption per GPU x batch-size combination...")
    
    for csv_file in csv_files:
        gpu, model, tp, pp = extract_config_from_filename(csv_file)
        if gpu is None:
            continue
            
        # Read the CSV file to analyze energy consumption
        df = pd.read_csv(csv_file)
        
        # Only process valid entries
        df_valid = df[df['valid'] == 1].copy()
        if len(df_valid) == 0:
            continue
            
        # Convert energy from mJ to J if needed (following display_results.py pattern)
        df_valid['energy_consumption'] = df_valid['energy_consumption'].astype(float)
        if df_valid['energy_consumption'].max() > 100:  # Assume mJ if values > 100
            df_valid['energy_consumption'] = df_valid['energy_consumption'] / 1000
            
        # Calculate energy per token for fair comparison
        df_valid['energy_per_token'] = df_valid['energy_consumption'] / df_valid['batch_size']
        
        # Group by batch_size and clock frequency to find optimal clocks per batch size
        for batch_size in df_valid['batch_size'].unique():
            batch_data = df_valid[df_valid['batch_size'] == batch_size]
            
            if len(batch_data) == 0:
                continue            
            # Iterate all clocks for this batch size
            for clock in batch_data['clock'].unique():
                clock_data = batch_data[batch_data['clock'] == clock]
                # Filter out rows where time_taken * pp > TPOT_SLO
                valid_rows = clock_data[clock_data['time_taken'] * pp <= TPOT_SLO]
                if len(valid_rows) == 0:
                    continue
                # Compute min energy per token for this clock
                min_energy = valid_rows['energy_per_token'].min()
                key = (gpu, batch_size, clock)
                optimal_clocks[key] = min_energy

                # Store for summary display (grouped by gpu and batch_size)
                if gpu not in gpu_batch_data:
                    gpu_batch_data[gpu] = {}
                if batch_size not in gpu_batch_data[gpu]:
                    gpu_batch_data[gpu][batch_size] = []
                gpu_batch_data[gpu][batch_size].append((clock, min_energy))
                
                
    # Print summary of optimal clocks per GPU x batch-size
    print("\nOptimal clock frequencies by GPU x batch-size:")
    for gpu in sorted(gpu_batch_data.keys()):
        print(f"\n  {gpu}:")
        batch_data = gpu_batch_data[gpu]
        for batch_size in sorted(batch_data.keys()):
            optimal_clock, min_energy = batch_data[batch_size]
            print(f"    Batch size {batch_size:4d}: {optimal_clock:4d} MHz (energy: {min_energy:.6f} J/token)")
        
    return optimal_clocks


def process_single_csv_file(csv_file, use_energy_optimal, optimal_clocks_per_batch, fallback_clocks, target_clocks, request_length, scheduling_delay_s, use_all_frequencies=False):
    """
    Process a single CSV file and return filtered data with calculated metrics.
    
    Args:
        csv_file: Path to CSV file
        use_energy_optimal: Whether to use energy-optimal clocks
        optimal_clocks_per_batch: Dict of (GPU, batch_size) -> optimal clock
        fallback_clocks: Dict of GPU -> fallback clock for energy mode
        target_clocks: Dict of GPU -> target clock for normal mode  
        request_length: Total request length in tokens for TTFT calculation
        scheduling_delay_s: Scheduling delay per chunk in seconds
        use_all_frequencies: If True, process all frequencies (ignore target clock filtering)
        
    Returns:
        pandas.DataFrame or None: Processed data for this CSV file
    """
    # Extract configuration from filename
    gpu, model, tp, pp = extract_config_from_filename(csv_file)
    
    if gpu is None:
        return None
        
    # Check if we have target clocks for this GPU (skip if using all frequencies)
    if not use_all_frequencies:
        if not use_energy_optimal:
            if gpu not in target_clocks:
                return None
        else:
            # For energy optimal mode, check if we have any optimal clocks for this GPU
            has_optimal_clocks = any(key[0] == gpu for key in optimal_clocks_per_batch.keys())
            if not has_optimal_clocks and gpu not in fallback_clocks:
                return None
            
    print(f"Processing: {gpu} - {model} (TP={tp}, PP={pp})")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Only keep valid entries first
    df_valid = df[df['valid'] == 1].copy()
    
    # Filter to only use specified chunk sizes
    # df_valid = df_valid[df_valid['batch_size'].isin(CHUNK_SIZES)]
    df_valid = df_valid[df_valid['batch_size'] >= MIN_PREFILL_CHUNK_SIZE]
    # 
    # print all batch sizes
    print(f"Batch sizes: {df_valid['batch_size'].unique()}")
    
    if len(df_valid) == 0:
        print(f"  No valid data for specified chunk sizes")
        return None
        
    # Now filter based on clock frequencies (or use all if requested)
    df_filtered_list = []
    
    if use_all_frequencies:
        # Use all available frequencies without filtering
        df_filtered = df_valid.copy()
        df_filtered['target_clock'] = df_filtered['clock']  # Set target_clock to actual clock
        df_filtered_list.append(df_filtered)
    elif use_energy_optimal:
        # Process each batch size with its optimal clock
        for batch_size in df_valid['batch_size'].unique():
            batch_data = df_valid[df_valid['batch_size'] == batch_size]
            
            # Get optimal clock for this (gpu, batch_size) combination
            key = (gpu, batch_size)
            if key in optimal_clocks_per_batch:
                optimal_clock = optimal_clocks_per_batch[key]
                # Filter to optimal clock for this batch size
                batch_filtered = batch_data[batch_data['clock'] == optimal_clock].copy()
                if len(batch_filtered) > 0:
                    batch_filtered['target_clock'] = optimal_clock
                    df_filtered_list.append(batch_filtered)
            else:
                # No optimal clock found for this combination, try fallback
                if gpu in fallback_clocks:
                    fallback_clock = fallback_clocks[gpu]
                    batch_filtered = batch_data[batch_data['clock'] == fallback_clock].copy()
                    if len(batch_filtered) > 0:
                        batch_filtered['target_clock'] = fallback_clock
                        df_filtered_list.append(batch_filtered)
                        print(f"  Using fallback clock {fallback_clock} MHz for batch size {batch_size}")
    else:
        # Original behavior: use same target clock for all batch sizes
        target_clock = target_clocks[gpu]
        df_filtered = df_valid[df_valid['clock'] == target_clock].copy()
        if len(df_filtered) > 0:
            df_filtered['target_clock'] = target_clock
            df_filtered_list.append(df_filtered)
    
    # Combine all filtered data
    if len(df_filtered_list) == 0:
        print(f"  No data at optimal clock frequencies")
        return None
        
    df_filtered = pd.concat(df_filtered_list, ignore_index=True)
    
    # Calculate TPS
    df_filtered['tps'] = df_filtered['batch_size'] / df_filtered['time_taken']
    
    # Calculate TTFT with interpolation
    batch_sizes_sorted = np.array(sorted(df_filtered['batch_size'].unique()))
    time_taken_sorted = np.array([df_filtered[df_filtered['batch_size'] == bs]['time_taken'].mean() 
                                 for bs in batch_sizes_sorted])
    
    ttft_values = []
    for _, row in df_filtered.iterrows():
        chunk_size = row['batch_size']
        
        if chunk_size >= request_length:
            # If chunk is larger than or equal to request length, use interpolation
            # and multiply by chunk_size * pp
            if request_length in batch_sizes_sorted:
                base_time = time_taken_sorted[np.where(batch_sizes_sorted == request_length)[0][0]]
            else:
                base_time = interpolate_time_taken(request_length, batch_sizes_sorted, time_taken_sorted)
            
            # Scale by chunk_size * pp
            ttft = base_time * pp
        else:
            # Original logic for smaller chunks
            num_full_chunks = int(request_length // chunk_size) + (pp - 1)
            # num_full_chunks = int(request_length // chunk_size) * pp
            remaining_tokens = request_length % chunk_size
            
            # Time for full chunks
            full_chunk_time = row['time_taken']
            
            # Time for remaining tokens (interpolated if needed)
            if remaining_tokens > 0:
                if remaining_tokens in batch_sizes_sorted:
                    remaining_time = time_taken_sorted[np.where(batch_sizes_sorted == remaining_tokens)[0][0]]
                else:
                    remaining_time = interpolate_time_taken(remaining_tokens, batch_sizes_sorted, time_taken_sorted)
            else:
                remaining_time = 0
            
            # Calculate TTFT: scheduling delays + processing time
            scheduling_delays = (num_full_chunks + (1 if remaining_tokens > 0 else 0) - 1) * scheduling_delay_s
            processing_time = num_full_chunks * full_chunk_time + remaining_time
            ttft = scheduling_delays + processing_time
        
        ttft_values.append(ttft)
    
    df_filtered['ttft'] = ttft_values
    
    # Calculate TPOT (Time to Process Single Chunk)
    df_filtered['tpot'] = (df_filtered['time_taken'] + scheduling_delay_s) * pp
    
    # Calculate total energy for prefill processing
    total_energy_values = []
    for _, row in df_filtered.iterrows():
        chunk_size = row['batch_size']
        energy_per_chunk = row['energy_consumption'] / 1000 if row['energy_consumption'] > 100 else row['energy_consumption']  # Convert mJ to J if needed
        
        if chunk_size >= request_length:
            # Single chunk can handle entire request
            num_chunks = 1
        else:
            # Multiple chunks needed
            num_chunks = int(np.ceil(request_length / chunk_size))
        
        # Total energy = num_chunks * pp * energy_per_chunk
        total_energy = num_chunks * pp * energy_per_chunk
        total_energy_values.append(total_energy)
    
    df_filtered['total_prefill_energy'] = total_energy_values
    
    # Add SLO violation flags during preprocessing
    df_filtered['ttft_violation'] = df_filtered['ttft'] > TTFT_SLO
    df_filtered['tpot_violation'] = df_filtered['tpot'] > TPOT_SLO  
    df_filtered['slo_violation'] = df_filtered['ttft_violation'] | df_filtered['tpot_violation']
    
    # Add configuration columns
    df_filtered['gpu'] = gpu
    df_filtered['model'] = model
    df_filtered['tp'] = tp
    df_filtered['pp'] = pp
    # target_clock is already set per batch size above
    df_filtered['scheduling_delay_ms'] = scheduling_delay_s * 1000
    
    # Print processing results
    available_chunk_sizes = sorted(df_filtered['batch_size'].unique())
    
    if use_energy_optimal:
        # Show clocks used for each batch size
        batch_clocks = df_filtered.groupby('batch_size')['target_clock'].first().to_dict()
        clock_info = ", ".join([f"{bs}:{clk}MHz" for bs, clk in sorted(batch_clocks.items())])
        print(f"  Found {len(df_filtered)} valid data points using optimal clocks")
        print(f"  Batch sizes and clocks: {clock_info}")
    else:
        target_clock = df_filtered['target_clock'].iloc[0]
        print(f"  Found {len(df_filtered)} valid data points at {target_clock} MHz")
        print(f"  Available chunk sizes: {available_chunk_sizes}")
    
    return df_filtered


def load_and_process_data(request_length=REQUEST_LENGTH, use_energy_optimal=False, use_all_frequencies=False):
    """
    Load and process all CSV data, calculating TPS, TTFT, and TPOT metrics.
    
    Args:
        request_length: Total request length in tokens for TTFT calculation
        use_energy_optimal: If True, use energy-optimal clock frequencies instead of hardcoded targets
        use_all_frequencies: If True, process all available frequencies (ignore target clock filtering)
        
    Returns:
        pandas.DataFrame: Combined data with all metrics calculated
    """
    # Parameters for TTFT calculation
    scheduling_delay_ms = 3  # Scheduling delay per chunk in ms
    scheduling_delay_s = scheduling_delay_ms / 1000.0

    # Find all CSV files in the directory first
    csv_pattern = '/workspace/vllm-energy-v1/vllm/benchmarks/energy/offline_profile_results/dvfs_profile_*.csv'
    csv_files = glob.glob(csv_pattern)
    
    # Filter out idle files
    csv_files = [f for f in csv_files if 'idle' not in f]
    
    # Define clock frequencies to use for each GPU type
    if use_all_frequencies:
        print("\nUsing all available clock frequencies for each GPU")
        optimal_clocks_per_batch = None
        fallback_clocks = None
        target_clocks = None
    elif use_energy_optimal:
        # Get optimal clocks per (GPU, batch_size) combination
        optimal_clocks_per_batch = find_optimal_clocks_by_energy(csv_files)
        print("\nUsing energy-optimal clock frequencies per GPU x batch-size combination")
        
        # Create a fallback dictionary for cases where we don't have batch-specific data
        fallback_clocks = {
            'NVIDIA RTX A6000': 1770,
            'NVIDIA A100-SXM4-40GB': 1350,
            'NVIDIA A100-SXM4-80GB': 1350
        }
        target_clocks = None
    else:
        optimal_clocks_per_batch = None
        fallback_clocks = None
        # Define specific clock frequencies to use for each GPU type
        target_clocks = {
            'NVIDIA RTX A6000': 1770,
            'NVIDIA A100-SXM4-40GB': 1350,
            'NVIDIA A100-SXM4-80GB': 1350
        }
        print("Using hardcoded target clock frequencies by GPU:")
        
        for gpu, target_clock in target_clocks.items():
            print(f"  {gpu}: {target_clock} MHz")
        print()
    
    print(f"Using only these chunk sizes: {CHUNK_SIZES}")
    print(f"Found {len(csv_files)} profile CSV files")
    print()

    # Process each CSV file and collect data
    all_data = []
    
    for csv_file in csv_files:
        # Process this CSV file using the extracted function
        processed_data = process_single_csv_file(
            csv_file=csv_file,
            use_energy_optimal=use_energy_optimal,
            optimal_clocks_per_batch=optimal_clocks_per_batch,
            fallback_clocks=fallback_clocks,
            target_clocks=target_clocks,
            request_length=request_length,
            scheduling_delay_s=scheduling_delay_s,
            use_all_frequencies=use_all_frequencies
        )
        
        if processed_data is not None:
            all_data.append(processed_data)
    
    if not all_data:
        print("No data found!")
        return None
        
    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data points: {len(df_combined)}")
    
    return df_combined


def create_energy_heatmaps(request_length=REQUEST_LENGTH):
    """
    Create pixel-like heatmaps showing energy consumption vs chunk size and frequency.
    Colors represent energy for valid SLO data, with violation markers for invalid data.
    
    Args:
        request_length: Total request length in tokens (default: 4096)
    """
    # Create output directory
    output_dir = Path(
        '/workspace/vllm-energy-v1/vllm/benchmarks/energy/offline_profile_results/tps_imgs'
    )
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data using the refactored function with all frequencies
    df_combined = load_and_process_data(request_length, use_energy_optimal=False, use_all_frequencies=True)
    if df_combined is None:
        return
    
    # Group by GPU first, then by configuration within each GPU
    gpu_groups = df_combined.groupby('gpu')
    print(f"Creating separate energy heatmaps for {len(gpu_groups)} GPU types...")

    # Find global min/max energy for consistent color scaling across all GPUs
    global_min_energy = df_combined[~df_combined['slo_violation']]['total_prefill_energy'].min()
    global_max_energy = df_combined[~df_combined['slo_violation']]['total_prefill_energy'].max()
    print(f"Global energy range (valid SLO): {global_min_energy:.3f} - {global_max_energy:.3f} J")
    
    # Set up matplotlib parameters
    plt.rcParams['font.size'] = 10

    # Process each GPU separately
    for gpu_name, gpu_data in gpu_groups:
        print(f"\nProcessing {gpu_name}...")
        
        # Group configurations within this GPU
        config_groups = gpu_data.groupby(['model', 'tp', 'pp'])
        print(f"  Found {len(config_groups)} configurations for {gpu_name}")
        
        # Calculate subplot layout for this GPU - stack vertically
        n_configs = len(config_groups)
        n_rows = n_configs + 1  # Stack all plots vertically
        n_cols = 1  # Single column
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
        if n_configs == 1:
            axes = [axes]
        else:
            axes = list(axes)

        # Create heatmap for each configuration within this GPU
        for i, ((model, tp, pp), group) in enumerate(config_groups):
            ax = axes[i]
            
            # Get unique frequencies and chunk sizes for this configuration
            frequencies = sorted(group['clock'].unique())
            chunk_sizes = sorted(group['batch_size'].unique())
            
            print(f"    {model.split('/')[-1]} (TP={tp}, PP={pp})")
            # print(f"      Frequencies: {len(frequencies)} values from {min(frequencies)} to {max(frequencies)} MHz")
            # print(f"      Chunk sizes: {chunk_sizes}")
            
            # Create grid for heatmap
            freq_indices = {freq: idx for idx, freq in enumerate(frequencies)}
            chunk_indices = {chunk: idx for idx, chunk in enumerate(chunk_sizes)}
            
            # Initialize arrays for heatmap data
            energy_grid = np.full((len(frequencies), len(chunk_sizes)), np.nan)
            ttft_violation_grid = np.full((len(frequencies), len(chunk_sizes)), False, dtype=bool)
            tpot_violation_grid = np.full((len(frequencies), len(chunk_sizes)), False, dtype=bool)
            
            # Fill the grids
            for _, row in group.iterrows():
                freq_idx = freq_indices[row['clock']]
                chunk_idx = chunk_indices[row['batch_size']]
                
                if not row['slo_violation']:
                    # Only color if within SLO
                    energy_grid[freq_idx, chunk_idx] = row['total_prefill_energy']
                
                # Track violations for markers
                ttft_violation_grid[freq_idx, chunk_idx] = row['ttft_violation']
                tpot_violation_grid[freq_idx, chunk_idx] = row['tpot_violation']
            
            # Create heatmap - use reversed viridis so brighter = less energy
            im = ax.imshow(energy_grid, cmap='viridis_r', aspect='auto', 
                          vmin=global_min_energy, vmax=global_max_energy,
                          origin='lower')
            
            # Find minimum energy point among valid SLO data
            valid_slo_data = group[~group['slo_violation']]
            min_energy_point = None
            min_energy_value = None
            min_energy_coords = None
            
            if len(valid_slo_data) > 0:
                min_idx = valid_slo_data['total_prefill_energy'].idxmin()
                min_energy_row = valid_slo_data.loc[min_idx]
                min_energy_value = min_energy_row['total_prefill_energy']
                min_freq = min_energy_row['clock']
                min_chunk = min_energy_row['batch_size']
                
                # Get grid coordinates for the minimum energy point
                if min_freq in freq_indices and min_chunk in chunk_indices:
                    min_freq_idx = freq_indices[min_freq]
                    min_chunk_idx = chunk_indices[min_chunk]
                    min_energy_coords = (min_chunk_idx, min_freq_idx)
                    
                    print(f"      Min energy: {min_energy_value:.2f}J at {min_freq}MHz, chunk {min_chunk}")

            # Add violation markers
            for freq_idx, freq in enumerate(frequencies):
                for chunk_idx, chunk in enumerate(chunk_sizes):
                    x, y = chunk_idx, freq_idx
                    
                    ttft_viol = ttft_violation_grid[freq_idx, chunk_idx]
                    tpot_viol = tpot_violation_grid[freq_idx, chunk_idx]
                    
                    # Add grey background to the cell
                    if ttft_viol or tpot_viol:
                        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='dimgray', alpha=0.5, zorder=0))
                    
                    if ttft_viol and tpot_viol:
                        # Both violations - diamond
                        ax.scatter(x, y, marker='D', s=100, color='black', 
                                  edgecolors='black', linewidths=2, alpha=0.8)
                    elif ttft_viol:
                        # TTFT violation only - triangle
                        ax.scatter(x, y, marker='^', s=120, color='black', 
                                  edgecolors='black', linewidths=2, alpha=0.8)
                    elif tpot_viol:
                        # TPOT violation only - square
                        ax.scatter(x, y, marker='s', s=100, color='black', 
                                  edgecolors='black', linewidths=2, alpha=0.8)
            
            # Add minimum energy marker and annotation
            if min_energy_coords is not None:
                x_min, y_min = min_energy_coords
                # Add a bright star marker for minimum energy point
                ax.scatter(x_min, y_min, marker='*', s=200, color='gold', 
                          edgecolors='orange', linewidths=3, alpha=1.0, zorder=20)
                
                # Add text annotation above the star
                ax.annotate(f'{min_energy_value:.1f}J', 
                           xy=(x_min, y_min), xytext=(0, 15), 
                           textcoords='offset points', ha='center', va='bottom',
                           fontsize=10, fontweight='bold', color='orange',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            # Configure axes
            ax.set_xticks(range(len(chunk_sizes)))
            ax.set_xticklabels([f"{c}" if c < 1024 else f"{c//1024}K" for c in chunk_sizes])
            ax.set_xlabel('Chunk Size', fontsize=12)
            
            ax.set_yticks(range(len(frequencies)))
            # Show fewer frequency labels if there are too many
            if len(frequencies) > 10:
                step = max(1, len(frequencies) // 8)
                tick_positions = range(0, len(frequencies), step)
                tick_labels = [f"{frequencies[i]}" for i in tick_positions]
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
            else:
                ax.set_yticklabels([f"{f}" for f in frequencies])
            ax.set_ylabel('Frequency (MHz)', fontsize=12)
            
            # Title
            config_name = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
            ax.set_title(f"{config_name}", fontsize=12, pad=10)
            
            # Add grid
            ax.set_xticks(np.arange(-0.5, len(chunk_sizes), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(frequencies), 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=1, alpha=0.5)
            
        # Hide unused subplots
        for j in range(len(config_groups), len(axes)):
            axes[j].set_visible(False)
        
        # Add single colorbar for all subplots in this GPU, lower it further
        cbar = fig.colorbar(im, ax=axes[:len(config_groups) + 1], shrink=0.6, aspect=20, 
                           location='bottom', pad=0.05)
        cbar.set_label('Total Prefill Energy (J) - Brighter = Less Energy', fontsize=12)
        
        # Add legend for violation markers and minimum energy point
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='gold', linewidth=0, markersize=15,
                   markeredgecolor='orange', markeredgewidth=2, alpha=1.0,
                   label='Minimum Energy Point'),
            Line2D([0], [0], marker='^', color='black', linewidth=0, markersize=10,
                   markeredgecolor='black', markeredgewidth=2, alpha=0.8,
                   label=f'TTFT Violation (>{TTFT_SLO}s)'),
            Line2D([0], [0], marker='s', color='black', linewidth=0, markersize=10,
                   markeredgecolor='black', markeredgewidth=2, alpha=0.8,
                   label=f'TPOT Violation (>{TPOT_SLO}s)'),
            Line2D([0], [0], marker='D', color='black', linewidth=0, markersize=10,
                   markeredgecolor='black', markeredgewidth=2, alpha=0.8,
                   label='Both Violations')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.3),
                  ncol=3, fontsize=12)
        
        # Add main title for the GPU
        fig.suptitle(f'{gpu_name} - Energy Heatmaps (Request Length: {request_length} tokens)', 
                     fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20, top=0.93, hspace=0.3)  # Make room for colorbar, legend, and main title

        # Save the plot for this GPU
        gpu_safe_name = sanitize_filename(gpu_name)
        filename = f"energy_heatmap_{gpu_safe_name}_freq_vs_chunk_{request_length}tokens.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")

    print(f"\nAll energy heatmaps saved to: {output_dir}")


def create_combined_plots(request_length=REQUEST_LENGTH, use_energy_optimal=False):
    """
    Create combined TPS, TTFT, and TPOT vs chunk size plots in a single image
    
    Args:
        request_length: Total request length in tokens (default: 4096)
        use_energy_optimal: If True, use energy-optimal clock frequencies instead of hardcoded targets
    """
    # Create output directory
    output_dir = Path(
        '/workspace/vllm-energy-v1/vllm/benchmarks/energy/offline_profile_results/tps_imgs'
    )
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data
    df_combined = load_and_process_data(REQUEST_LENGTH, use_energy_optimal)
    if df_combined is None:
        return

    # Group by GPU for individual plots
    gpu_groups = df_combined.groupby('gpu')
    print(f"Creating combined TPS/TTFT/TPOT plots for {len(gpu_groups)} GPUs...")

    # Set up matplotlib parameters
    plt.rcParams['figure.figsize'] = FIG_SIZE
    plt.rcParams['font.size'] = 10

    for gpu, gpu_data in gpu_groups:
        # Create figure with 4 subplots stacked vertically
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True)
        
        # Create colors based on model/configuration for identification
        config_groups = gpu_data.groupby(['model', 'tp', 'pp'])
        colors = plt.cm.tab10(np.linspace(0, 1, len(config_groups)))
        
        target_clock = gpu_data['target_clock'].iloc[0]
        scheduling_delay_ms = gpu_data['scheduling_delay_ms'].iloc[0]
        
        # Plot each metric on its subplot (original colors for all points)
        for i, ((model, tp, pp), group) in enumerate(config_groups):
            config_label = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
            group_sorted = group.sort_values('batch_size')
            
            markersize = 6 if len(config_groups) > 3 else 8
            
            # Plot all points in original colors
            # Plot 1: TPS vs Chunk Size
            ax1.plot(group_sorted['batch_size'], group_sorted['tps'], 
                    'o-', color=colors[i], linewidth=2, markersize=markersize,
                    alpha=0.8, label=config_label, markeredgecolor='black', markeredgewidth=0.5)
            
            # Plot 2: TTFT vs Chunk Size
            ax2.plot(group_sorted['batch_size'], group_sorted['ttft'], 
                    'o-', color=colors[i], linewidth=2, markersize=markersize,
                    alpha=0.8, label=config_label, markeredgecolor='black', markeredgewidth=0.5)
            
            # Plot 3: TPOT vs Chunk Size
            ax3.plot(group_sorted['batch_size'], group_sorted['tpot'], 
                    'o-', color=colors[i], linewidth=2, markersize=markersize,
                    alpha=0.8, label=config_label, markeredgecolor='black', markeredgewidth=0.5)
            
            # Plot 4: Total Prefill Energy vs Chunk Size
            ax4.plot(group_sorted['batch_size'], group_sorted['total_prefill_energy'], 
                    'o-', color=colors[i], linewidth=2, markersize=markersize,
                    alpha=0.8, label=config_label, markeredgecolor='black', markeredgewidth=0.5)
        
        # Add grey symbols on top of SLO violations with different shapes
        markersize = 6 if len(config_groups) > 3 else 8  # Define markersize for violation symbols
        for i, ((model, tp, pp), group) in enumerate(config_groups):
            group_sorted = group.sort_values('batch_size')
            
            # Get TTFT-only violations (triangles)
            ttft_only_violations = group_sorted[group_sorted['ttft_violation'] & ~group_sorted['tpot_violation']]
            
            # Get TPOT-only violations (squares)
            tpot_only_violations = group_sorted[group_sorted['tpot_violation'] & ~group_sorted['ttft_violation']]
            
            # Get both violations (diamonds)
            both_violations = group_sorted[group_sorted['ttft_violation'] & group_sorted['tpot_violation']]
            
            violation_size = (markersize + 2) * 8  # Size for violation markers
            
            # Add grey triangles for TTFT-only violations
            if len(ttft_only_violations) > 0:
                ax1.scatter(ttft_only_violations['batch_size'], ttft_only_violations['tps'], 
                           marker='^', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax2.scatter(ttft_only_violations['batch_size'], ttft_only_violations['ttft'], 
                           marker='^', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax3.scatter(ttft_only_violations['batch_size'], ttft_only_violations['tpot'], 
                           marker='^', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax4.scatter(ttft_only_violations['batch_size'], ttft_only_violations['total_prefill_energy'], 
                           marker='^', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
            
            # Add grey squares for TPOT-only violations
            if len(tpot_only_violations) > 0:
                ax1.scatter(tpot_only_violations['batch_size'], tpot_only_violations['tps'], 
                           marker='s', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax2.scatter(tpot_only_violations['batch_size'], tpot_only_violations['ttft'], 
                           marker='s', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax3.scatter(tpot_only_violations['batch_size'], tpot_only_violations['tpot'], 
                           marker='s', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax4.scatter(tpot_only_violations['batch_size'], tpot_only_violations['total_prefill_energy'], 
                           marker='s', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
            
            # Add grey diamonds for both violations
            if len(both_violations) > 0:
                ax1.scatter(both_violations['batch_size'], both_violations['tps'], 
                           marker='D', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax2.scatter(both_violations['batch_size'], both_violations['ttft'], 
                           marker='D', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax3.scatter(both_violations['batch_size'], both_violations['tpot'], 
                           marker='D', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
                
                ax4.scatter(both_violations['batch_size'], both_violations['total_prefill_energy'], 
                           marker='D', color='grey', s=violation_size, alpha=0.9, zorder=10, 
                           edgecolors='darkgrey', linewidths=1.5)
            
        # Add annotations for each subplot
        for i, ((model, tp, pp), group) in enumerate(config_groups):
            group_sorted = group.sort_values('batch_size')
            
            for _, row in group_sorted.iterrows():
                # Use preprocessed violation flag
                annotation_color = 'darkgrey' if row['slo_violation'] else 'black'
                
                # TPS annotations
                ax1.annotate(f"{row['tps']:.0f}",
                           xy=(row['batch_size'], row['tps']),
                           xytext=(3, 8), textcoords='offset points',
                           fontsize=7, alpha=0.9, ha='center', weight='bold',
                           color=annotation_color)
                
                # TTFT annotations
                ax2.annotate(f"{row['ttft']:.3f}",
                           xy=(row['batch_size'], row['ttft']),
                           xytext=(3, 8), textcoords='offset points',
                           fontsize=7, alpha=0.9, ha='center', weight='bold',
                           color=annotation_color)
                
                # TPOT annotations
                ax3.annotate(f"{row['tpot']:.4f}",
                           xy=(row['batch_size'], row['tpot']),
                           xytext=(3, 8), textcoords='offset points',
                           fontsize=7, alpha=0.9, ha='center', weight='bold',
                           color=annotation_color)
                
                # Total Energy annotations  
                ax4.annotate(f"{row['total_prefill_energy']:.3f}",
                           xy=(row['batch_size'], row['total_prefill_energy']),
                           xytext=(3, 8), textcoords='offset points',
                           fontsize=7, alpha=0.9, ha='center', weight='bold',
                           color=annotation_color)

        # Configure subplots
        ax1.set_ylabel('Tokens per Second (TPS)', fontsize=12)
        if use_energy_optimal:
            clock_info = f"Energy-Optimal Clocks"
        else:
            clock_info = f"{target_clock} MHz"
        ax1.set_title(f'TPS vs Batch Size ({clock_info})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        ax2.set_ylabel('Time To First Token (TTFT) (seconds)', fontsize=12)
        ax2.set_title(f'TTFT vs Batch Size ({clock_info})\n(Request Length={request_length}, Sched Delay={scheduling_delay_ms}ms)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        # Add TTFT SLO line
        ax2.axhline(y=TTFT_SLO, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'TTFT SLO: {TTFT_SLO}s')
        
        ax3.set_ylabel('Time to Process Single Chunk (seconds)', fontsize=12)
        ax3.set_title(f'Single Chunk Processing Time vs Batch Size ({clock_info})', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        # Add TPOT SLO line
        ax3.axhline(y=TPOT_SLO, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'TPOT SLO: {TPOT_SLO}s')
        
        ax4.set_xlabel('Batch Size (Chunk Size)', fontsize=12)
        ax4.set_ylabel('Total Prefill Energy (Joules)', fontsize=12)
        ax4.set_title(f'Total Energy for Prefill Processing ({clock_info})\n(Request Length={request_length}, PP={scheduling_delay_ms}ms delay)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        # Add single shared legend at the bottom
        handles, labels = ax1.get_legend_handles_labels()
        
        # Add legend entries for different SLO violation types
        from matplotlib.lines import Line2D
        
        # Triangle for TTFT violations
        ttft_violation_handle = Line2D([0], [0], marker='^', color='grey', linewidth=0, 
                                      markersize=10, markeredgecolor='darkgrey', markeredgewidth=1.5,
                                      label='TTFT Violation (>4.0s)')
        handles.append(ttft_violation_handle)
        labels.append('TTFT Violation (>4.0s)')
        
        # Square for TPOT violations
        tpot_violation_handle = Line2D([0], [0], marker='s', color='grey', linewidth=0, 
                                      markersize=10, markeredgecolor='darkgrey', markeredgewidth=1.5,
                                      label='TPOT Violation (>0.2s)')
        handles.append(tpot_violation_handle)
        labels.append('TPOT Violation (>0.2s)')
        
        # Diamond for both violations
        both_violation_handle = Line2D([0], [0], marker='D', color='grey', linewidth=0, 
                                      markersize=8, markeredgecolor='darkgrey', markeredgewidth=1.5,
                                      label='Both Violations')
        handles.append(both_violation_handle)
        labels.append('Both Violations')
        
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=min(len(labels), 4), fontsize=9)
        
        # Add box with minimal TTFT values for each configuration
        min_ttft_values = []
        for (model, tp, pp), group in config_groups:
            group_sorted = group.sort_values('batch_size')
            min_ttft = group_sorted['ttft'].min()
            min_chunk_size = group_sorted.loc[group_sorted['ttft'].idxmin(), 'batch_size']
            config_name = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
            min_ttft_values.append(f"{config_name}: {min_ttft:.3f}s @ {min_chunk_size}")
        
        # Create text box with minimal TTFT values
        min_ttft_text = "Minimal TTFT Values:\n" + "\n".join(min_ttft_values)
        ax2.text(0.02, 0.98, min_ttft_text, transform=ax2.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Add box with maximal TPS values for each configuration
        max_tps_values = []
        for (model, tp, pp), group in config_groups:
            group_sorted = group.sort_values('batch_size')
            max_tps = group_sorted['tps'].max()
            max_chunk_size = group_sorted.loc[group_sorted['tps'].idxmax(), 'batch_size']
            config_name = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
            max_tps_values.append(f"{config_name}: {max_tps:.0f} TPS @ {max_chunk_size}")
        
        # Create text box with maximal TPS values
        max_tps_text = "Maximal TPS Values:\n" + "\n".join(max_tps_values)
        ax1.text(0.02, 0.98, max_tps_text, transform=ax1.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Add box with largest chunk size under TPOT SLO for each configuration
        tpot_slo_values = []
        for (model, tp, pp), group in config_groups:
            group_sorted = group.sort_values('batch_size')
            # Find configurations that meet TPOT SLO
            under_slo = group_sorted[group_sorted['tpot'] <= TPOT_SLO]
            if len(under_slo) > 0:
                largest_chunk = under_slo['batch_size'].max()
                config_name = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
                tpot_slo_values.append(f"{config_name}: {largest_chunk} (≤{TPOT_SLO}s)")
            else:
                config_name = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
                tpot_slo_values.append(f"{config_name}: None (>{TPOT_SLO}s)")
        
        # Create text box with TPOT SLO values
        tpot_slo_text = f"Largest Chunk Under TPOT SLO ({TPOT_SLO}s):\n" + "\n".join(tpot_slo_values)
        ax3.text(0.02, 0.98, tpot_slo_text, transform=ax3.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # Add box with minimal energy values for each configuration
        min_energy_values = []
        for (model, tp, pp), group in config_groups:
            group_sorted = group.sort_values('batch_size')
            min_energy = group_sorted['total_prefill_energy'].min()
            min_chunk_size = group_sorted.loc[group_sorted['total_prefill_energy'].idxmin(), 'batch_size']
            config_name = f"{model.split('/')[-1]} (TP={tp}, PP={pp})"
            min_energy_values.append(f"{config_name}: {min_energy:.3f}J @ {min_chunk_size}")
        
        # Create text box with minimal energy values
        min_energy_text = "Minimal Prefill Energy:\n" + "\n".join(min_energy_values)
        ax4.text(0.02, 0.98, min_energy_text, transform=ax4.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # Set x-axis to log scale for all subplots
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xscale('log')
        ax4.set_xscale('log')
        
        # Set x-axis ticks to only show the chunk sizes that are present in the data
        batch_sizes = sorted(gpu_data['batch_size'].unique())
        # Use only the chunk sizes that are actually present in the data
        valid_ticks = [t for t in CHUNK_SIZES if t in batch_sizes]
        
        # Create descriptive labels for batch sizes
        tick_labels = []
        for tick in valid_ticks:
            if tick >= 1024:
                if tick % 1024 == 0:
                    tick_labels.append(f"{tick//1024}K\n({tick})")
                else:
                    tick_labels.append(f"{tick}")
            else:
                tick_labels.append(f"{tick}")
        
        ax3.set_xticks(valid_ticks)
        ax3.set_xticklabels([])  # No labels for ax3 since ax4 will have them
        
        ax4.set_xticks(valid_ticks)
        ax4.set_xticklabels(tick_labels, fontsize=9)
        
        plt.tight_layout()

        # Create filename and save
        if use_energy_optimal:
            filename = f"{sanitize_filename(gpu)}_combined_tps_ttft_tpot_vs_chunk_energy_optimal.png"
        else:
            target_clock = gpu_data['target_clock'].iloc[0]
            filename = f"{sanitize_filename(gpu)}_combined_tps_ttft_tpot_vs_chunk_{target_clock}MHz.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")
        
        # Print statistics
        if use_energy_optimal:
            clock_range = f"{gpu_data['target_clock'].min()}-{gpu_data['target_clock'].max()}"
            print(f"  {gpu} (Energy-optimal clocks: {clock_range} MHz):")
        else:
            target_clock = gpu_data['target_clock'].iloc[0]
            print(f"  {gpu} (Target Clock: {target_clock} MHz):")
        print(f"    TPS range: {gpu_data['tps'].min():.1f}-{gpu_data['tps'].max():.1f}")
        print(f"    TTFT range: {gpu_data['ttft'].min():.3f}-{gpu_data['ttft'].max():.3f}s")
        print(f"    Single chunk processing time range: {gpu_data['tpot'].min():.4f}-{gpu_data['tpot'].max():.4f}s")
        print(f"    Total prefill energy range: {gpu_data['total_prefill_energy'].min():.3f}-{gpu_data['total_prefill_energy'].max():.3f}J")
        print(f"    Batch size range: {gpu_data['batch_size'].min()}-{gpu_data['batch_size'].max()}")
        print(f"    Configurations: {len(config_groups)}")
        print(f"    Total measurements: {len(gpu_data)}")
        print()

    print(f"All combined plots saved to: {output_dir}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create visualization plots from energy profiling data')
    parser.add_argument('--energy', action='store_true', 
                        help='Use energy-optimal clock frequencies instead of hardcoded target clocks')
    parser.add_argument('--request-length', type=int, default=REQUEST_LENGTH,
                        help=f'Request length in tokens for TTFT calculations (default: {REQUEST_LENGTH})')
    parser.add_argument('--heatmap', action='store_true',
                        help='Create energy heatmaps showing frequency vs chunk size with SLO violations')
    
    args = parser.parse_args()
    
    if args.heatmap:
        print("Creating energy heatmaps (frequency vs chunk size)...")
        create_energy_heatmaps(request_length=args.request_length)
    else:
        print("Creating combined TPS/TTFT/TPOT vs Chunk Size plots...")
        if args.energy:
            print("Using --energy mode: Finding optimal clock frequencies per GPU x batch-size combination")
        else:
            print("Using hardcoded target clock frequencies")
            
        create_combined_plots(request_length=args.request_length, use_energy_optimal=args.energy)
    
    print("\nAll plots completed!")