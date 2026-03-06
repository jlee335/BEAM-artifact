#!/usr/bin/env python3
"""
Windowed timeline visualization for TTFT, TBT, and Energy metrics.

This script creates timeline plots showing how metrics evolve over time
in configurable time windows (default: 10s).
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import glob
import argparse


def load_json_trace(json_file):
    """Load trace data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def load_energy_data(csv_file):
    """Load energy data from CSV file."""
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_start_finish_time(batch_log_file):
    """
    Get start and finish time from batch log file.
    
    Args:
        batch_log_file: Path to batch_log_GPU_0.csv
        
    Returns:
        start_time: pandas Timestamp for experiment start
        finish_time: pandas Timestamp for experiment finish
    """
    df = pd.read_csv(batch_log_file)
    start_time = pd.to_datetime(df['start_time'].iloc[0])
    finish_time = pd.to_datetime(df['current_time'].iloc[-1])
    return start_time, finish_time


def compute_windowed_ttft(entry_times, ttfts, window_size=10.0):
    """
    Compute mean TTFT for each time window.
    
    Args:
        entry_times: List of request entry times (in seconds from start)
        ttfts: List of TTFT values (in seconds)
        window_size: Size of time window in seconds
        
    Returns:
        windows: List of window start times
        values: List of mean TTFT values for each window
    """
    if not entry_times:
        return [], []
    
    max_time = max(entry_times)
    num_windows = int(np.ceil(max_time / window_size))
    
    windows = []
    values = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        
        # Find TTFTs that fall in this window
        window_ttfts = [
            ttft for entry, ttft in zip(entry_times, ttfts)
            if window_start <= entry < window_end
        ]
        
        if window_ttfts:
            windows.append(window_start)
            values.append(np.mean(window_ttfts) * 1000)  # Convert to ms
    
    return windows, values


def compute_windowed_tbt(entry_times, ttfts, itls, window_size=10.0):
    """
    Compute mean TBT for each time window.
    
    For each ITL (TBT), calculate its entry time as:
    itl_entry_time = entry_time_for_request + ttft + cumulative_sum(previous_itls)
    
    Args:
        entry_times: List of request entry times (in seconds from start)
        ttfts: List of TTFT values (in seconds)
        itls: List of lists - each sublist contains ITLs for a request
        window_size: Size of time window in seconds
        
    Returns:
        windows: List of window start times
        values: List of mean TBT values for each window
    """
    if not entry_times:
        return [], []
    
    # Collect all TBTs with their entry times
    tbt_data = []
    
    for entry_time, ttft, itl_list in zip(entry_times, ttfts, itls):
        cumulative_time = 0.0
        for itl in itl_list:
            # ITL entry time = request entry + TTFT + cumulative sum of previous ITLs
            itl_entry_time = entry_time + ttft + cumulative_time
            tbt_data.append((itl_entry_time, itl))
            cumulative_time += itl
    
    if not tbt_data:
        return [], []
    
    max_time = max(t for t, _ in tbt_data)
    num_windows = int(np.ceil(max_time / window_size))
    
    windows = []
    values = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        
        # Find TBTs that fall in this window
        window_tbts = [
            tbt for time, tbt in tbt_data
            if window_start <= time < window_end
        ]
        
        if window_tbts:
            windows.append(window_start)
            values.append(np.mean(window_tbts) * 1000)  # Convert to ms
    
    return windows, values


def compute_windowed_energy(energy_df, start_time, window_size=10.0):
    """
    Compute average power consumption (watts) for each time window.
    
    Note: Energy values in the CSV are in millijoules (mJ) and are converted to joules.
    
    Args:
        energy_df: DataFrame with energy data (values in millijoules)
        start_time: Start timestamp for the experiment
        window_size: Size of time window in seconds
        
    Returns:
        windows: List of window start times
        values: List of average power consumption (Watts) for each window
    """
    if energy_df.empty:
        return [], []
    
    # Calculate time offset from start
    energy_df = energy_df.copy()
    energy_df['time_offset'] = (energy_df['timestamp'] - start_time).dt.total_seconds()
    
    # Get GPU columns
    gpu_joule_cols = [col for col in energy_df.columns if col.endswith('_joules')]
    
    if not gpu_joule_cols:
        return [], []
    
    max_time = energy_df['time_offset'].max()
    num_windows = int(np.ceil(max_time / window_size))
    
    windows = []
    values = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        
        # Get data in this window
        window_data = energy_df[
            (energy_df['time_offset'] >= window_start) &
            (energy_df['time_offset'] < window_end)
        ]
        
        if len(window_data) > 1:
            # Calculate energy delta (end - start) for each GPU (in millijoules)
            total_energy_mj = 0.0
            for col in gpu_joule_cols:
                energy_start = window_data[col].iloc[0]
                energy_end = window_data[col].iloc[-1]
                total_energy_mj += (energy_end - energy_start)
            
            # Calculate actual time duration for this window
            time_start = window_data['time_offset'].iloc[0]
            time_end = window_data['time_offset'].iloc[-1]
            actual_duration = time_end - time_start
            
            if actual_duration > 0:
                # Convert millijoules to joules, then calculate power
                # Power = Energy / Time (Watts = Joules / seconds)
                total_energy_j = total_energy_mj / 1000.0  # Convert mJ to J
                power_watts = total_energy_j / actual_duration
                windows.append(window_start)
                values.append(power_watts)
    
    return windows, values


def compute_windowed_frequency(energy_df, start_time, window_size=10.0):
    """
    Compute average GPU frequency (MHz) for each time window.
    
    Args:
        energy_df: DataFrame with energy data (includes frequency columns)
        start_time: Start timestamp for the experiment
        window_size: Size of time window in seconds. If None, return raw frequencies without windowing.
        
    Returns:
        windows: List of window start times (or raw time offsets if window_size is None)
        values: List of average GPU frequency (MHz) for each window (or raw average frequencies if window_size is None)
    """
    if energy_df.empty:
        return [], []
    
    # Calculate time offset from start
    energy_df = energy_df.copy()
    energy_df['time_offset'] = (energy_df['timestamp'] - start_time).dt.total_seconds()
    
    # Get GPU frequency columns
    gpu_freq_cols = [col for col in energy_df.columns if col.endswith('_freq_mhz')]
    
    if not gpu_freq_cols:
        return [], []
    
    # If window_size is None, return raw frequencies without windowing
    if window_size is None:
        times = energy_df['time_offset'].tolist()
        # Calculate average frequency across all GPUs for each timestamp
        avg_freqs = energy_df[gpu_freq_cols].mean(axis=1).tolist()
        return times, avg_freqs
    
    max_time = energy_df['time_offset'].max()
    num_windows = int(np.ceil(max_time / window_size))
    
    windows = []
    values = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        
        # Get data in this window
        window_data = energy_df[
            (energy_df['time_offset'] >= window_start) &
            (energy_df['time_offset'] < window_end)
        ]
        
        if len(window_data) > 0:
            # Calculate average frequency across all GPUs in this window
            avg_freq = 0.0
            for col in gpu_freq_cols:
                avg_freq += window_data[col].mean()
            avg_freq /= len(gpu_freq_cols)
            
            windows.append(window_start)
            values.append(avg_freq)
    
    return windows, values


def compute_windowed_throughput(entry_times, input_lens, window_size=10.0):
    """
    Compute request throughput (tokens per second) for each time window.
    
    Throughput is calculated as the total number of prefill tokens that entered
    the system during each time window, divided by the window size.
    
    Args:
        entry_times: List of request entry times (in seconds from start)
        input_lens: List of input lengths (prefill token counts) for each request
        window_size: Size of time window in seconds
        
    Returns:
        windows: List of window start times
        values: List of throughput values (tokens per second) for each window
    """
    if not entry_times or not input_lens:
        return [], []
    
    max_time = max(entry_times)
    num_windows = int(np.ceil(max_time / window_size))
    
    windows = []
    values = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        
        # Find all requests that entered in this window
        total_tokens = sum(
            input_len for entry_time, input_len in zip(entry_times, input_lens)
            if window_start <= entry_time < window_end
        )
        
        # Calculate throughput (tokens per second)
        throughput = total_tokens / window_size
        
        windows.append(window_start)
        values.append(throughput)
    
    return windows, values


def load_s1_s2_data(traced_dataset, start_time):
    """
    Load S1 and S2 log data if available.
    
    Args:
        traced_dataset: Path to traced_dataset directory
        start_time: Start timestamp for the experiment
        
    Returns:
        Dictionary with S1 and S2 data (timestamps and values)
    """
    s1_file = os.path.join(traced_dataset, 's1_log.csv')
    s2_file = os.path.join(traced_dataset, 's2_log.csv')
    
    s1_data = {'timestamps': [], 'values': []}
    s2_data = {'timestamps': [], 'values': []}
    
    # Load S1 data (optimal_chunk_size)
    if os.path.exists(s1_file):
        try:
            s1_df = pd.read_csv(s1_file)
            s1_df = s1_df.dropna()
            if not s1_df.empty and 'timestamp' in s1_df.columns:
                # Convert timestamps to datetime if they're Unix timestamps
                df_start_time = s1_df['timestamp'].iloc[0]
                df_start_time = pd.to_datetime(df_start_time, unit='s')
                if s1_df['timestamp'].dtype in ['int64', 'float64']:
                    s1_df['timestamp'] = pd.to_datetime(s1_df['timestamp'], unit='s')
                else:
                    s1_df['timestamp'] = pd.to_datetime(s1_df['timestamp'])
                
                # Convert timestamps to time offsets relative to experiment start
                s1_df['time_offset'] = (s1_df['timestamp'] - df_start_time).dt.total_seconds()
                s1_data['timestamps'] = s1_df['time_offset'].tolist()
                s1_data['values'] = s1_df['optimal_chunk_size'].tolist()
        except Exception as e:
            print(f"Warning: Failed to load S1 data: {e}")
    
    # Load S2 data (optimal_num_microbatches)
    if os.path.exists(s2_file):
        try:
            s2_df = pd.read_csv(s2_file)
            s2_df = s2_df.dropna()
            if not s2_df.empty and 'timestamp' in s2_df.columns:
                # Convert timestamps to datetime if they're Unix timestamps
                df_start_time = s2_df['timestamp'].iloc[0]  
                df_start_time = pd.to_datetime(df_start_time, unit='s')
                if s2_df['timestamp'].dtype in ['int64', 'float64']:
                    s2_df['timestamp'] = pd.to_datetime(s2_df['timestamp'], unit='s')
                else:
                    s2_df['timestamp'] = pd.to_datetime(s2_df['timestamp'])
                
                # Convert timestamps to time offsets relative to experiment start
                s2_df['time_offset'] = (s2_df['timestamp'] - df_start_time).dt.total_seconds()
                s2_data['timestamps'] = s2_df['time_offset'].tolist()
                s2_data['values'] = s2_df['optimal_num_microbatches'].tolist()
        except Exception as e:
            print(f"Warning: Failed to load S2 data: {e}")
    
    return {'s1': s1_data, 's2': s2_data}


def load_additional_csv_data(traced_dataset, start_time):
    """
    Load additional CSV data: kv_usage_log.csv, s1_exec_time_log.csv, s2_exec_time_log.csv, schedule_exec_time_log.csv.
    
    Args:
        traced_dataset: Path to traced_dataset directory
        start_time: Start timestamp for the experiment
        
    Returns:
        Dictionary with kv_usage, s1_exec_time, s2_exec_time, and schedule_exec_time data
    """
    kv_usage_file = os.path.join(traced_dataset, 'kv_usage_log.csv')
    s1_exec_file = os.path.join(traced_dataset, 's1_exec_time_log.csv')
    s2_exec_file = os.path.join(traced_dataset, 's2_exec_time_log.csv')
    schedule_exec_file = os.path.join(traced_dataset, 'schedule_exec_time_log.csv')
    
    kv_usage_data = {'timestamps': [], 'values': []}
    s1_exec_data = {'timestamps': [], 'values': []}
    s2_exec_data = {'timestamps': [], 'values': []}
    schedule_exec_data = {'timestamps': [], 'values': []}
    
    # Load KV usage data
    if os.path.exists(kv_usage_file):
        try:
            kv_df = pd.read_csv(kv_usage_file)
            kv_df = kv_df.dropna()
            if not kv_df.empty and 'timestamp' in kv_df.columns and 'usage' in kv_df.columns:
                # Convert timestamps to datetime
                df_start_time = kv_df['timestamp'].iloc[0]
                df_start_time = pd.to_datetime(df_start_time, unit='s')
                if kv_df['timestamp'].dtype in ['int64', 'float64']:
                    kv_df['timestamp'] = pd.to_datetime(kv_df['timestamp'], unit='s')
                else:
                    kv_df['timestamp'] = pd.to_datetime(kv_df['timestamp'])
                
                # Convert timestamps to time offsets
                kv_df['time_offset'] = (kv_df['timestamp'] - df_start_time).dt.total_seconds()
                kv_usage_data['timestamps'] = kv_df['time_offset'].tolist()
                kv_usage_data['values'] = (kv_df['usage'] * 100).tolist()  # Convert to percentage
        except Exception as e:
            print(f"Warning: Failed to load KV usage data: {e}")
    
    # Load S1 execution time data
    if os.path.exists(s1_exec_file):
        try:
            s1_exec_df = pd.read_csv(s1_exec_file)
            s1_exec_df = s1_exec_df.dropna()
            if not s1_exec_df.empty and 'timestamp' in s1_exec_df.columns and 'execution_time' in s1_exec_df.columns:
                df_start_time = s1_exec_df['timestamp'].iloc[0]
                df_start_time = pd.to_datetime(df_start_time, unit='s')
                if s1_exec_df['timestamp'].dtype in ['int64', 'float64']:
                    s1_exec_df['timestamp'] = pd.to_datetime(s1_exec_df['timestamp'], unit='s')
                else:
                    s1_exec_df['timestamp'] = pd.to_datetime(s1_exec_df['timestamp'])
                
                s1_exec_df['time_offset'] = (s1_exec_df['timestamp'] - df_start_time).dt.total_seconds()
                s1_exec_data['timestamps'] = s1_exec_df['time_offset'].tolist()
                s1_exec_data['values'] = (s1_exec_df['execution_time'] * 1000).tolist()  # Convert to ms
        except Exception as e:
            print(f"Warning: Failed to load S1 exec time data: {e}")
    
    # Load S2 execution time data
    if os.path.exists(s2_exec_file):
        try:
            s2_exec_df = pd.read_csv(s2_exec_file)
            s2_exec_df = s2_exec_df.dropna()
            if not s2_exec_df.empty and 'timestamp' in s2_exec_df.columns and 'execution_time' in s2_exec_df.columns:
                df_start_time = s2_exec_df['timestamp'].iloc[0]
                df_start_time = pd.to_datetime(df_start_time, unit='s')
                if s2_exec_df['timestamp'].dtype in ['int64', 'float64']:
                    s2_exec_df['timestamp'] = pd.to_datetime(s2_exec_df['timestamp'], unit='s')
                else:
                    s2_exec_df['timestamp'] = pd.to_datetime(s2_exec_df['timestamp'])
                
                s2_exec_df['time_offset'] = (s2_exec_df['timestamp'] - df_start_time).dt.total_seconds()
                s2_exec_data['timestamps'] = s2_exec_df['time_offset'].tolist()
                s2_exec_data['values'] = (s2_exec_df['execution_time'] * 1000).tolist()  # Convert to ms
        except Exception as e:
            print(f"Warning: Failed to load S2 exec time data: {e}")
    
    # Load Schedule execution time data
    if os.path.exists(schedule_exec_file):
        try:
            schedule_exec_df = pd.read_csv(schedule_exec_file)
            schedule_exec_df = schedule_exec_df.dropna()
            if not schedule_exec_df.empty and 'timestamp' in schedule_exec_df.columns and 'execution_time' in schedule_exec_df.columns:
                df_start_time = schedule_exec_df['timestamp'].iloc[0]
                df_start_time = pd.to_datetime(df_start_time, unit='s')
                if schedule_exec_df['timestamp'].dtype in ['int64', 'float64']:
                    schedule_exec_df['timestamp'] = pd.to_datetime(schedule_exec_df['timestamp'], unit='s')
                else:
                    schedule_exec_df['timestamp'] = pd.to_datetime(schedule_exec_df['timestamp'])
                
                schedule_exec_df['time_offset'] = (schedule_exec_df['timestamp'] - df_start_time).dt.total_seconds()
                schedule_exec_data['timestamps'] = schedule_exec_df['time_offset'].tolist()
                schedule_exec_data['values'] = (schedule_exec_df['execution_time'] * 1000).tolist()  # Convert to ms
        except Exception as e:
            print(f"Warning: Failed to load Schedule exec time data: {e}")
    
    return {
        'kv_usage': kv_usage_data,
        's1_exec_time': s1_exec_data,
        's2_exec_time': s2_exec_data,
        'schedule_exec_time': schedule_exec_data
    }


def load_phase_info(base_dir):
    """
    Load phase information from phase_info.csv if available.
    
    Args:
        base_dir: Path to experiment directory
        
    Returns:
        List of phase start times (in seconds), or empty list if not found
    """
    phase_file = os.path.join(base_dir, 'phase_info.csv')
    
    if not os.path.exists(phase_file):
        return []
    
    try:
        phase_df = pd.read_csv(phase_file)
        phase_df = phase_df.dropna()
        if not phase_df.empty and 'start_time' in phase_df.columns:
            return phase_df['start_time'].tolist()
    except Exception as e:
        print(f"Warning: Failed to load phase info from {phase_file}: {e}")
    
    return []


def load_slo_info(base_dir):
    """
    Load SLO information from slo_info.txt if available.
    
    Args:
        base_dir: Path to experiment directory
        
    Returns:
        Dictionary with 'ttft_ms' and 'tbt_ms' SLO values, or None if not found
    """
    slo_file = os.path.join(base_dir, 'slo_info.txt')
    
    if not os.path.exists(slo_file):
        return None
    
    try:
        with open(slo_file, 'r') as f:
            content = f.read()
        
        ttft_slo = None
        tbt_slo = None
        
        # Parse the file
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('TBT:'):
                # Format: "TBT: 0.2s"
                tbt_str = line.split(':')[1].strip().rstrip('s')
                tbt_slo = float(tbt_str) * 1000  # Convert to ms
            elif line.startswith('TTFT:'):
                # Format: "TTFT: 1.0s"
                ttft_str = line.split(':')[1].strip().rstrip('s')
                ttft_slo = float(ttft_str) * 1000  # Convert to ms
        
        if ttft_slo is not None or tbt_slo is not None:
            return {'ttft_ms': ttft_slo, 'tbt_ms': tbt_slo}
        
    except Exception as e:
        print(f"Warning: Failed to parse SLO info from {slo_file}: {e}")
    
    return None


def get_experiment_name(directory):
    """Extract a readable name from the directory path."""
    dir_name = os.path.basename(directory)
    
    # Map common patterns to readable names
    name_map = {
        'A_vanilla_vllm': 'Vanilla vLLM',
        'B2_dynamollm_dvfs': 'DynamoLLM+DVFS',
        'C_s1_only': 'S1 Only',
        'D_s1_s2': 'S1+S2',
    }
    
    # Check for exact matches or partial matches
    for key, value in name_map.items():
        if key in dir_name:
            return value
    
    # Fall back to abbreviated directory name
    return dir_name[:30]


def process_experiment_directory(base_dir, window_size=10.0):
    """
    Process a single experiment directory and extract windowed metrics.
    
    Args:
        base_dir: Path to experiment directory (contains traced_dataset/)
        window_size: Size of time window in seconds
        
    Returns:
        Dictionary with windowed data for TTFT, TBT, and Energy
    """
    traced_dataset = os.path.join(base_dir, 'traced_dataset')
    
    if not os.path.exists(traced_dataset):
        raise ValueError(f"traced_dataset not found in {base_dir}")
    
    # Find JSON file
    json_files = glob.glob(os.path.join(traced_dataset, '*.json'))
    if not json_files:
        raise ValueError(f"No JSON file found in {traced_dataset}")
    
    # Find the file that does NOT have _estimated in the name
    json_file = None
    for f in json_files:
        if '_estimations' not in os.path.basename(f):
            json_file = f
            break
    if json_file is None:
        raise ValueError("No JSON file without '_estimations' in the name found in traced_dataset")
    # Find energy CSV file
    energy_file = os.path.join(traced_dataset, 'gpu_energy_and_frequency_huggyllama_llama-13b.csv')
    if not os.path.exists(energy_file):
        raise ValueError(f"Energy file not found: {energy_file}")
    
    # Find batch log file for accurate start/finish times
    batch_log_file = os.path.join(traced_dataset, 'batch_log_GPU_0.csv')
    if not os.path.exists(batch_log_file):
        raise ValueError(f"Batch log file not found: {batch_log_file}")
    
    # Load data
    trace_data = load_json_trace(json_file)
    energy_df = load_energy_data(energy_file)
    
    # Extract metrics
    entry_times = trace_data.get('entry_times', [])
    ttfts = trace_data.get('ttfts', [])
    itls = trace_data.get('itls', [])
    input_lens = trace_data.get('input_lens', [])
    
    # Get start and finish time from batch log for accurate timing
    start_time, finish_time = get_start_finish_time(batch_log_file)
    
    # Compute windowed metrics
    ttft_windows, ttft_values = compute_windowed_ttft(entry_times, ttfts, window_size)
    tbt_windows, tbt_values = compute_windowed_tbt(entry_times, ttfts, itls, window_size)
    energy_windows, energy_values = compute_windowed_energy(energy_df, start_time, window_size)
    freq_windows, freq_values = compute_windowed_frequency(energy_df, start_time, window_size)
    throughput_windows, throughput_values = compute_windowed_throughput(entry_times, input_lens, window_size)
    
    # Load S1 and S2 data if available
    s1_s2_data = load_s1_s2_data(traced_dataset, start_time)
    
    # Load additional CSV data
    additional_data = load_additional_csv_data(traced_dataset, start_time)
    
    # Load phase info if available
    phase_times = load_phase_info(base_dir)
    
    # Load SLO info if available
    slo_info = load_slo_info(base_dir)
    
    return {
        'name': get_experiment_name(base_dir),
        'ttft': {'windows': ttft_windows, 'values': ttft_values},
        'tbt': {'windows': tbt_windows, 'values': tbt_values},
        'energy': {'windows': energy_windows, 'values': energy_values},
        'frequency': {'windows': freq_windows, 'values': freq_values},
        'throughput': {'windows': throughput_windows, 'values': throughput_values},
        's1': s1_s2_data['s1'],
        's2': s1_s2_data['s2'],
        'kv_usage': additional_data['kv_usage'],
        's1_exec_time': additional_data['s1_exec_time'],
        's2_exec_time': additional_data['s2_exec_time'],
        'schedule_exec_time': additional_data['schedule_exec_time'],
        'phase_times': phase_times,
        'slo_info': slo_info,
    }


def visualize_timeline(experiments_data, window_size=10.0, save_path=None):
    """
    Create timeline visualization with subplots for TTFT, TBT, Energy, GPU Frequency, Throughput, and optionally S1 and S2.
    
    Args:
        experiments_data: List of experiment data dictionaries
        window_size: Size of time window in seconds
        save_path: Path to save the figure (optional)
    """
    # Check if any experiment has S1 or S2 data (independently)
    has_s1 = any(
        exp.get('s1', {}).get('timestamps', [])
        for exp in experiments_data
    )
    has_s2 = any(
        exp.get('s2', {}).get('timestamps', [])
        for exp in experiments_data
    )
    
    # Create appropriate number of subplots with shared x-axis
    # Base plots: TTFT, TBT, Energy, Frequency, Throughput (5 plots)
    num_plots = 5
    if has_s1:
        num_plots += 1
    if has_s2:
        num_plots += 1
    
    fig_height = 16 + (3 * (num_plots - 5))  # Base 16 inches + 3 inches per additional plot
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, fig_height), sharex=True)
    
    # Define colors for different experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot TTFT
    ax = axes[0]
    for i, exp_data in enumerate(experiments_data):
        windows = exp_data['ttft']['windows']
        values = exp_data['ttft']['values']
        if windows:
            ax.plot(windows, values, 
                   label=exp_data['name'],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_ylabel('Mean TTFT (ms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add phase boundary lines if available
    for i, exp_data in enumerate(experiments_data):
        phase_times = exp_data.get('phase_times', [])
        if phase_times:
            for phase_time in phase_times:
                ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Add SLO line if available (only add once from first experiment with SLO)
    slo_added = False
    for exp_data in experiments_data:
        slo_info = exp_data.get('slo_info')
        if slo_info and not slo_added:
            ttft_slo = slo_info.get('ttft_ms')
            if ttft_slo is not None:
                ax.axhline(y=ttft_slo, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label=f'SLO ({ttft_slo:.0f} ms)')
                slo_added = True
                break
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'Time To First Token - {window_size}s Windows', fontsize=14, fontweight='bold')
    
    # Plot TBT
    ax = axes[1]
    for i, exp_data in enumerate(experiments_data):
        windows = exp_data['tbt']['windows']
        values = exp_data['tbt']['values']
        if windows:
            ax.plot(windows, values,
                   label=exp_data['name'],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_ylabel('Mean TBT (ms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add phase boundary lines if available
    for i, exp_data in enumerate(experiments_data):
        phase_times = exp_data.get('phase_times', [])
        if phase_times:
            for phase_time in phase_times:
                ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Add SLO line if available (only add once from first experiment with SLO)
    slo_added = False
    for exp_data in experiments_data:
        slo_info = exp_data.get('slo_info')
        if slo_info and not slo_added:
            tbt_slo = slo_info.get('tbt_ms')
            if tbt_slo is not None:
                ax.axhline(y=tbt_slo, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label=f'SLO ({tbt_slo:.0f} ms)')
                slo_added = True
                break
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'Time Between Tokens - {window_size}s Windows', fontsize=14, fontweight='bold')
    
    # Plot Power (Energy)
    ax = axes[2]
    for i, exp_data in enumerate(experiments_data):
        windows = exp_data['energy']['windows']
        values = exp_data['energy']['values']
        if windows:
            ax.plot(windows, values,
                   label=exp_data['name'],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_ylabel('Power Consumption (W)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add phase boundary lines if available
    for i, exp_data in enumerate(experiments_data):
        phase_times = exp_data.get('phase_times', [])
        if phase_times:
            for phase_time in phase_times:
                ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'Power Consumption - {window_size}s Windows', fontsize=14, fontweight='bold')
    
    # Plot GPU Frequency
    ax = axes[3]
    for i, exp_data in enumerate(experiments_data):
        windows = exp_data['frequency']['windows']
        values = exp_data['frequency']['values']
        if windows:
            ax.plot(windows, values,
                   label=exp_data['name'],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_ylabel('Average GPU Frequency (MHz)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add phase boundary lines if available
    for i, exp_data in enumerate(experiments_data):
        phase_times = exp_data.get('phase_times', [])
        if phase_times:
            for phase_time in phase_times:
                ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'GPU Frequency - {window_size}s Windows', fontsize=14, fontweight='bold')
    
    # Plot Request Throughput
    ax = axes[4]
    for i, exp_data in enumerate(experiments_data):
        windows = exp_data['throughput']['windows']
        values = exp_data['throughput']['values']
        if windows:
            ax.plot(windows, values,
                   label=exp_data['name'],
                   color=colors[i % len(colors)],
                   linewidth=2,
                   alpha=0.8)
    
    ax.set_ylabel('Request Throughput (Tokens/s)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add phase boundary lines if available
    for i, exp_data in enumerate(experiments_data):
        phase_times = exp_data.get('phase_times', [])
        if phase_times:
            for phase_time in phase_times:
                ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'Request Throughput - {window_size}s Windows', fontsize=14, fontweight='bold')
    
    # Add x-axis label only if this is the last plot (no S1/S2 data)
    if not has_s1 and not has_s2:
        ax.set_xlabel('Time (s)', fontsize=12)
    
    # Determine current subplot index for S1/S2
    current_plot_idx = 5
    
    # Plot S1 (Optimal Chunk Size) if available
    if has_s1:
        ax = axes[current_plot_idx]
        for i, exp_data in enumerate(experiments_data):
            s1_data = exp_data.get('s1', {})
            timestamps = s1_data.get('timestamps', [])
            values = s1_data.get('values', [])
            if timestamps and values:
                ax.plot(timestamps, values,
                       marker='o', linestyle='-', linewidth=1.5, markersize=4,
                       label=exp_data['name'],
                       color=colors[i % len(colors)],
                       alpha=0.8)
        
        ax.set_ylabel('Optimal Chunk Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add phase boundary lines if available
        for i, exp_data in enumerate(experiments_data):
            phase_times = exp_data.get('phase_times', [])
            if phase_times:
                for phase_time in phase_times:
                    ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_title('S1: Optimal Chunk Size Over Time', fontsize=14, fontweight='bold')
        
        # Set y-axis to show discrete values clearly
        all_values = []
        for exp_data in experiments_data:
            s1_data = exp_data.get('s1', {})
            all_values.extend(s1_data.get('values', []))
        if all_values:
            unique_values = sorted(set(all_values))
            ax.set_yticks(unique_values)
            if len(unique_values) > 1:
                ax.set_ylim([min(unique_values) - 32, max(unique_values) + 32])
        
        # Add x-axis label if S1 is the last plot (no S2)
        if not has_s2:
            ax.set_xlabel('Time (s)', fontsize=12)
        
        current_plot_idx += 1
    
    # Plot S2 (Optimal Number of Microbatches) if available
    if has_s2:
        ax = axes[current_plot_idx]
        for i, exp_data in enumerate(experiments_data):
            s2_data = exp_data.get('s2', {})
            timestamps = s2_data.get('timestamps', [])
            values = s2_data.get('values', [])
            if timestamps and values:
                ax.plot(timestamps, values,
                       marker='o', linestyle='-', linewidth=1.5, markersize=4,
                       label=exp_data['name'],
                       color=colors[i % len(colors)],
                       alpha=0.8)
        
        ax.set_ylabel('Optimal Number of Microbatches', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add phase boundary lines if available
        for i, exp_data in enumerate(experiments_data):
            phase_times = exp_data.get('phase_times', [])
            if phase_times:
                for phase_time in phase_times:
                    ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_title('S2: Optimal Number of Microbatches Over Time', fontsize=14, fontweight='bold')
        
        # Set y-axis to show discrete values clearly
        all_values = []
        for exp_data in experiments_data:
            s2_data = exp_data.get('s2', {})
            all_values.extend(s2_data.get('values', []))
        if all_values:
            unique_values = sorted(set(all_values))
            ax.set_yticks(unique_values)
            if len(unique_values) > 1:
                ax.set_ylim([min(unique_values) - 0.5, max(unique_values) + 0.5])
            else:
                val = unique_values[0]
                ax.set_ylim([val - 0.5, val + 0.5])
        
        # Add x-axis label (S2 is always the last plot when it exists)
        ax.set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def visualize_additional_csv_data(experiments_data, save_path=None):
    """
    Create a separate timeline visualization for additional CSV data:
    KV cache usage, S1 execution time, S2 execution time, and Schedule execution time.
    
    Args:
        experiments_data: List of experiment data dictionaries
        save_path: Path to save the figure (optional)
    """
    # Check which data is available
    has_kv_usage = any(
        exp.get('kv_usage', {}).get('timestamps', [])
        for exp in experiments_data
    )
    has_s1_exec = any(
        exp.get('s1_exec_time', {}).get('timestamps', [])
        for exp in experiments_data
    )
    has_s2_exec = any(
        exp.get('s2_exec_time', {}).get('timestamps', [])
        for exp in experiments_data
    )
    has_schedule_exec = any(
        exp.get('schedule_exec_time', {}).get('timestamps', [])
        for exp in experiments_data
    )
    
    # Count how many plots we need
    num_plots = sum([has_kv_usage, has_s1_exec, has_s2_exec, has_schedule_exec])
    
    if num_plots == 0:
        print("Warning: No additional CSV data found to visualize")
        return None
    
    # Create subplots
    fig_height = 4 + (3 * (num_plots - 1))  # Base 4 inches + 3 inches per additional plot
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, fig_height), sharex=True, squeeze=False)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Define colors for different experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    current_plot_idx = 0
    
    # Plot KV Cache Usage if available
    if has_kv_usage:
        ax = axes[current_plot_idx]
        for i, exp_data in enumerate(experiments_data):
            kv_data = exp_data.get('kv_usage', {})
            timestamps = kv_data.get('timestamps', [])
            values = kv_data.get('values', [])
            if timestamps and values:
                ax.plot(timestamps, values,
                       label=exp_data['name'],
                       color=colors[i % len(colors)],
                       linewidth=1.5,
                       alpha=0.7)
        
        ax.set_ylabel('KV Cache Usage (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add phase boundary lines if available
        for i, exp_data in enumerate(experiments_data):
            phase_times = exp_data.get('phase_times', [])
            if phase_times:
                for phase_time in phase_times:
                    ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_title('KV Cache Usage Over Time', fontsize=14, fontweight='bold')
        
        current_plot_idx += 1
    
    # Plot S1 Execution Time if available
    if has_s1_exec:
        ax = axes[current_plot_idx]
        for i, exp_data in enumerate(experiments_data):
            s1_exec_data = exp_data.get('s1_exec_time', {})
            timestamps = s1_exec_data.get('timestamps', [])
            values = s1_exec_data.get('values', [])
            if timestamps and values:
                ax.plot(timestamps, values,
                       marker='o', linestyle='-', linewidth=1.5, markersize=3,
                       label=exp_data['name'],
                       color=colors[i % len(colors)],
                       alpha=0.7)
        
        ax.set_ylabel('S1 Execution Time (ms)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add phase boundary lines if available
        for i, exp_data in enumerate(experiments_data):
            phase_times = exp_data.get('phase_times', [])
            if phase_times:
                for phase_time in phase_times:
                    ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_title('S1 (Chunk Size) Execution Time', fontsize=14, fontweight='bold')
        
        current_plot_idx += 1
    
    # Plot S2 Execution Time if available
    if has_s2_exec:
        ax = axes[current_plot_idx]
        for i, exp_data in enumerate(experiments_data):
            s2_exec_data = exp_data.get('s2_exec_time', {})
            timestamps = s2_exec_data.get('timestamps', [])
            values = s2_exec_data.get('values', [])
            if timestamps and values:
                ax.plot(timestamps, values,
                       marker='o', linestyle='-', linewidth=1.5, markersize=3,
                       label=exp_data['name'],
                       color=colors[i % len(colors)],
                       alpha=0.7)
        
        ax.set_ylabel('S2 Execution Time (ms)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add phase boundary lines if available
        for i, exp_data in enumerate(experiments_data):
            phase_times = exp_data.get('phase_times', [])
            if phase_times:
                for phase_time in phase_times:
                    ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_title('S2 (Microbatches) Execution Time', fontsize=14, fontweight='bold')
        
        current_plot_idx += 1
    
    # Plot Schedule Execution Time if available
    if has_schedule_exec:
        ax = axes[current_plot_idx]
        for i, exp_data in enumerate(experiments_data):
            schedule_exec_data = exp_data.get('schedule_exec_time', {})
            timestamps = schedule_exec_data.get('timestamps', [])
            values = schedule_exec_data.get('values', [])
            if timestamps and values:
                ax.plot(timestamps, values,
                       marker='o', linestyle='-', linewidth=1.5, markersize=3,
                       label=exp_data['name'],
                       color=colors[i % len(colors)],
                       alpha=0.7)
        
        ax.set_ylabel('Schedule Execution Time (ms)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add phase boundary lines if available
        for i, exp_data in enumerate(experiments_data):
            phase_times = exp_data.get('phase_times', [])
            if phase_times:
                for phase_time in phase_times:
                    ax.axvline(x=phase_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_title('Schedule Execution Time', fontsize=14, fontweight='bold')
    
    # Add x-axis label to the last plot
    axes[num_plots - 1].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Additional CSV data figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize windowed timeline of TTFT, TBT, and Energy metrics'
    )
    parser.add_argument(
        'parent_directory',
        type=str,
        help='Parent directory containing experiment subdirectories (A, B, C, D, etc.)'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=3.0,
        help='Time window size in seconds (default: 3.0)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the figure (optional, default: auto-generated in parent directory)'
    )
    
    args = parser.parse_args()
    
    # Find all experiment directories
    parent_dir = os.path.abspath(args.parent_directory)
    if not os.path.exists(parent_dir):
        print(f"Error: Directory not found: {parent_dir}")
        return
    
    # Get all subdirectories
    subdirs = sorted([
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ])
    
    if not subdirs:
        print(f"Error: No subdirectories found in {parent_dir}")
        return
    
    print(f"Found {len(subdirs)} experiment directories")
    print(f"Window size: {args.window_size}s\n")
    
    # Process each experiment
    experiments_data = []
    for subdir in subdirs:
        try:
            print(f"Processing: {os.path.basename(subdir)}")
            exp_data = process_experiment_directory(subdir, args.window_size)
            experiments_data.append(exp_data)
            print(f"  ✓ Loaded: {exp_data['name']}")
            
            # Print SLO info if available
            slo_info = exp_data.get('slo_info')
            if slo_info:
                print(f"  → SLO Info:")
                if slo_info.get('ttft_ms') is not None:
                    print(f"    TTFT: {slo_info['ttft_ms']:.0f} ms")
                if slo_info.get('tbt_ms') is not None:
                    print(f"    TBT: {slo_info['tbt_ms']:.0f} ms")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if not experiments_data:
        print("\nError: No valid experiment data loaded")
        return
    
    print(f"\nCreating timeline visualization...")
    
    # Generate default save path if not provided
    save_path = args.save
    if save_path is None:
        # Auto-generate filename based on parent directory name and window size
        parent_name = os.path.basename(parent_dir)
        filename = f"timeline_{parent_name}.png"
        save_path = os.path.join(parent_dir, filename)
    
    # Create main timeline visualization
    visualize_timeline(experiments_data, args.window_size, save_path)
    
    # Create additional CSV data visualization
    print(f"\nCreating additional CSV data visualization...")
    additional_save_path = save_path.replace('.png', '_additional_csv_data.png')
    visualize_additional_csv_data(experiments_data, additional_save_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

