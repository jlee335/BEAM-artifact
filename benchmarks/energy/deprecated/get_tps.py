#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Script to analyze profiling results and find maximum tokens-per-second (TPS) 
for each configuration and clock frequency.

For each CSV file:
1. Calculate TPS = batch_size / (time_taken * num_pp) (accounting for pipeline parallelism)
2. For each clock frequency, find the batch size with maximum TPS
3. Generate a comprehensive summary table
"""

import glob
import os
import re

import pandas as pd


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

    # Fallback parsing for complex cases
    parts = basename.replace('dvfs_profile_', '').replace('_one.csv',
                                                          '').split('_')

    # Find tp and pp indices
    tp_idx = pp_idx = -1
    for i, part in enumerate(parts):
        if part.startswith('tp') and part[2:].isdigit():
            tp_idx = i
            tp = int(part[2:])
        elif part.startswith('pp') and part[2:].isdigit():
            pp_idx = i
            pp = int(part[2:])

    if tp_idx > 0 and pp_idx > 0:
        gpu_parts = parts[:min(tp_idx, pp_idx) -
                          1] if min(tp_idx, pp_idx) > 0 else []
        model_parts = parts[min(tp_idx, pp_idx) - 1:min(tp_idx, pp_idx)]

        gpu = '_'.join(gpu_parts) if gpu_parts else 'Unknown'
        model = '_'.join(model_parts) if model_parts else 'Unknown'

        return gpu, model, tp, pp

    return 'Unknown', 'Unknown', 0, 0


def calculate_tps_for_file(filepath):
    """Calculate TPS for each row in a CSV file"""
    try:
        df = pd.read_csv(filepath)

        # Check if required columns exist
        required_cols = ['clock', 'batch_size', 'time_taken', 'valid']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in {filepath}")
            return None

        # Filter only valid measurements and exclude zero time measurements
        df = df[(df['valid'] == 1) & (df['time_taken'] > 0)].copy()

        if df.empty:
            print(f"No valid measurements in {filepath}")
            return None

        # Keep track of tokens_per_request for reference
        if 'total_ctx_len' in df.columns:
            df['tokens_per_request'] = df['total_ctx_len']
        else:
            df['tokens_per_request'] = 64  # Default assumption

        return df[[
            'clock', 'batch_size', 'time_taken', 'energy_consumption',
            'tokens_per_request'
        ]]

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def find_max_tps_per_clock(df, pp_factor):
    """For each clock frequency, find the batch size with maximum TPS"""
    if df is None or df.empty:
        return None

    # Calculate TPS accounting for pipeline parallelism
    # With PP, effective latency = time_taken * num_pp, so TPS = batch_size / (time_taken * num_pp)
    df = df.copy()
    df['tps'] = df['batch_size'] / (df['time_taken'] * pp_factor)

    # Group by clock and find the row with maximum TPS for each clock
    max_tps_rows = df.loc[df.groupby('clock')['tps'].idxmax()]

    return max_tps_rows[[
        'clock', 'batch_size', 'tps', 'time_taken', 'energy_consumption',
        'tokens_per_request'
    ]]


def main():
    # Find all _one.csv files
    csv_files = glob.glob(
        '/workspace/vllm-energy-v1/vllm/benchmarks/energy/offline_profile_results/*_one.csv'
    )

    print(f"Found {len(csv_files)} CSV files to analyze")

    all_results = []

    for filepath in sorted(csv_files):
        print(f"Processing: {os.path.basename(filepath)}")

        # Extract configuration
        gpu, model, tp, pp = extract_config_from_filename(filepath)
        print(f"  Configuration: {gpu}, {model}, TP={tp}, PP={pp}")

        # Calculate TPS for this file
        df = calculate_tps_for_file(filepath)

        if df is not None:
            print(f"  Valid measurements: {len(df)}")
            # Find max TPS per clock (accounting for PP factor)
            max_tps_df = find_max_tps_per_clock(df, pp)

            if max_tps_df is not None:
                # Add configuration information
                max_tps_df = max_tps_df.copy()
                max_tps_df['gpu'] = gpu
                max_tps_df['model'] = model
                max_tps_df['tp'] = tp
                max_tps_df['pp'] = pp
                max_tps_df['filename'] = os.path.basename(filepath)

                all_results.append(max_tps_df)

    if all_results:
        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)

        # Reorder columns for better readability
        column_order = [
            'gpu', 'model', 'tp', 'pp', 'clock', 'batch_size', 'tps',
            'time_taken', 'energy_consumption', 'tokens_per_request',
            'filename'
        ]
        final_df = final_df[column_order]

        # Sort by GPU, model, tp, pp, and clock
        final_df = final_df.sort_values(['gpu', 'model', 'tp', 'pp', 'clock'])

        # Save results
        output_file = '/workspace/vllm-energy-v1/vllm/benchmarks/energy/offline_profile_results/max_tps_summary.csv'
        final_df.to_csv(output_file, index=False, float_format='%.6f')

        print(f"\nSummary saved to: {output_file}")
        print(
            f"Total configurations analyzed: {len(final_df['filename'].unique())}"
        )
        print(f"Total measurements: {len(final_df)}")

        # Display summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"GPU types: {final_df['gpu'].nunique()}")
        print(f"Models: {final_df['model'].nunique()}")
        print(
            f"TP/PP configurations: {final_df[['tp', 'pp']].drop_duplicates().shape[0]}"
        )
        print(
            f"Clock frequencies range: {final_df['clock'].min()} - {final_df['clock'].max()}"
        )
        print(f"Max TPS achieved: {final_df['tps'].max():.2f} tokens/sec")
        print(
            f"Configuration with max TPS: {final_df.loc[final_df['tps'].idxmax()][['gpu', 'model', 'tp', 'pp', 'clock', 'batch_size']].to_dict()}"
        )

        # Show top 10 configurations by TPS
        print("\n=== TOP 10 CONFIGURATIONS BY TPS ===")
        top_configs = final_df.nlargest(10, 'tps')[[
            'gpu', 'model', 'tp', 'pp', 'clock', 'batch_size', 'tps'
        ]]
        print(
            top_configs.to_string(index=False,
                                  float_format=lambda x: f'{x:.2f}'))

    else:
        print("No valid results found!")


if __name__ == "__main__":
    main()
