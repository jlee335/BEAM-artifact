#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Script to parse CSV files and plot Energy (J) vs Clock Frequency (MHz).
Energy consumption is plotted for specific batch sizes (16, 512, 2048).
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_csv_files(directory="."):
    """
    Parse all CSV files ending with '_one.csv' in the given directory.
    
    Args:
        directory (str): Directory to search for CSV files
        
    Returns:
        dict: Dictionary with filename as key and DataFrame as value
    """
    # Find all CSV files ending with '_one.csv'
    pattern = os.path.join(directory, "*_one.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print("No CSV files found with '_one.csv' suffix in the directory")
        return {}

    print(f"Found {len(csv_files)} CSV files:")
    data_dict = {}

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"  - {filename}")

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Check if required columns exist
            required_columns = ['clock', 'batch_size', 'energy_consumption', 'time_taken']
            if not all(col in df.columns for col in required_columns):
                print(
                    f"Warning: {filename} missing required columns. Available columns: {list(df.columns)}"
                )
                continue

            # Extract pp from filename to calculate latency
            # filename format expected: ..._tpX_ppY_...
            pp_match = re.search(r'_pp(\d+)', filename)
            pp = int(pp_match.group(1)) if pp_match else 1
            if not pp_match:
                print(f"  Warning: Could not extract pp from {filename}, assuming pp=1")

            # Calculate latency
            df['latency'] = df['time_taken'] * pp

            # Convert energy_consumption from mJ to J if needed
            if df['energy_consumption'].max() > 10000:  # Likely in mJ
                df['energy_consumption'] = df['energy_consumption'] / 1000
                print(
                    f"  Converted energy_consumption from mJ to J for {filename}"
                )

            data_dict[filename] = df

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return data_dict


def plot_clock_versus_energy(data_dict, target_batch_sizes=[16, 512, 2048]):
    """
    Create plots of Energy (J) vs Clock Frequency (MHz) for specific batch sizes.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
        target_batch_sizes (list): List of batch sizes to plot
    """
    if not data_dict:
        print("No data to plot")
        return

    for filename, df in data_dict.items():
        # Extract model name and GPU from filename for cleaner title
        # Example: dvfs_profile_NVIDIA RTX A6000_ByteResearch_Llama-3-8B-Instruct_tp1_pp4_one.csv
        parts = filename.replace('dvfs_profile_', '').replace('_one.csv',
                                                              '').split('_')

        # Try to extract meaningful title
        if len(parts) >= 3:
            gpu = parts[0] + ' ' + parts[1] + ' ' + parts[2] if parts[
                1] not in ['A100-SXM4-80GB'] else parts[0] + ' ' + parts[1]
            model_parts = [
                p for p in parts[2:] if not any(x in p for x in ['tp', 'pp'])
            ]
            model = '_'.join(model_parts) if model_parts else 'Unknown'
            title = f"{gpu} - {model}"
        else:
            title = filename.replace('_one.csv', '')

        # Create subplot for each batch size
        # 2 rows: Top for Energy, Bottom for Latency
        fig, axes = plt.subplots(2,
                                 len(target_batch_sizes),
                                 figsize=(6 * len(target_batch_sizes), 8))
        
        # Get available batch sizes in the data
        available_batch_sizes = sorted(df['batch_size'].unique())
        print(
            f"\nAvailable batch sizes for {filename}: {available_batch_sizes}")

        for i, target_batch_size in enumerate(target_batch_sizes):
            # Select axes
            if len(target_batch_sizes) == 1:
                ax_energy = axes[0]
                ax_latency = axes[1]
            else:
                ax_energy = axes[0][i]
                ax_latency = axes[1][i]

            # Find the closest available batch size
            closest_batch_size = min(available_batch_sizes,
                                     key=lambda x: abs(x - target_batch_size))

            if abs(
                    closest_batch_size - target_batch_size
            ) > target_batch_size * 0.5:  # If difference is more than 50%
                print(
                    f"Warning: No close match for batch_size {target_batch_size}, using {closest_batch_size}"
                )

            # Filter data for this batch size
            batch_data = df[df['batch_size'] == closest_batch_size].copy()

            if batch_data.empty:
                print(f"No data found for batch_size {closest_batch_size}")
                continue

            # Group by clock and calculate mean energy and latency
            grouped = batch_data.groupby('clock').agg({
                'energy_consumption': ['mean', 'std'],
                'latency': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            grouped.columns = ['clock', 'energy_mean', 'energy_std', 'latency_mean', 'latency_std']
            grouped = grouped.sort_values('clock')

            # --- Plot Energy ---
            ax_energy.plot(grouped['clock'],
                         grouped['energy_mean'],
                         marker='o',
                         linewidth=3,
                         markersize=8,
                         color='red',
                         markeredgewidth=1.5,
                         markeredgecolor='black')

            # Add error bars if there's variation
            if len(grouped) > 1 and grouped['energy_std'].max() > 0:
                ax_energy.errorbar(grouped['clock'],
                                 grouped['energy_mean'],
                                 yerr=grouped['energy_std'],
                                 fmt='none',
                                 color='red',
                                 alpha=0.5,
                                 capsize=5)

            # Highlight minimum energy point with a thick dot
            min_energy_idx = grouped['energy_mean'].idxmin()
            min_energy_clock = grouped.loc[min_energy_idx, 'clock']
            min_energy_value = grouped.loc[min_energy_idx, 'energy_mean']
            ax_energy.plot(min_energy_clock,
                         min_energy_value,
                         marker='o',
                         markersize=15,
                         color='green',
                         markeredgewidth=2,
                         markeredgecolor='black',
                         zorder=5,
                         label='Minimum Energy')

            ax_energy.set_xlabel('Clock Frequency (MHz)',
                               fontsize=11,
                               fontweight='bold')
            ax_energy.set_ylabel('Energy Consumption (J)',
                               fontsize=11,
                               fontweight='bold')
            ax_energy.set_title(f'Energy - Batch Size = {closest_batch_size}',
                              fontsize=12,
                              fontweight='bold')
            ax_energy.grid(True, alpha=0.4, linewidth=0.8)
            ax_energy.set_ylim(bottom=0)

            # Set x-axis range based on available clock frequencies
            min_clock = grouped['clock'].min()
            max_clock = grouped['clock'].max()
            clock_range = max_clock - min_clock
            x_limits = (min_clock - clock_range * 0.05, max_clock + clock_range * 0.05)
            ax_energy.set_xlim(x_limits)

            # --- Plot Latency ---
            ax_latency.plot(grouped['clock'],
                         grouped['latency_mean'],
                         marker='s',
                         linewidth=3,
                         markersize=8,
                         color='blue',
                         markeredgewidth=1.5,
                         markeredgecolor='black')

            # Add error bars for latency
            if len(grouped) > 1 and grouped['latency_std'].max() > 0:
                ax_latency.errorbar(grouped['clock'],
                                  grouped['latency_mean'],
                                  yerr=grouped['latency_std'],
                                  fmt='none',
                                  color='blue',
                                  alpha=0.5,
                                  capsize=5)

            ax_latency.set_xlabel('Clock Frequency (MHz)',
                                fontsize=11,
                                fontweight='bold')
            ax_latency.set_ylabel('Latency (s)',
                                fontsize=11,
                                fontweight='bold')
            ax_latency.set_title(f'Latency - Batch Size = {closest_batch_size}',
                               fontsize=12,
                               fontweight='bold')
            ax_latency.grid(True, alpha=0.4, linewidth=0.8)
            ax_latency.set_ylim(bottom=0)
            ax_latency.set_xlim(x_limits)

        # Add overall title
        fig.suptitle(f'Energy and Latency vs Clock Frequency - {title}',
                     fontsize=14,
                     fontweight='bold')

        # Adjust layout
        plt.tight_layout(pad=2.0)

        # Save the plot with a clean filename
        clean_filename = filename.replace('dvfs_profile_',
                                          '').replace('_one.csv',
                                                      '').replace(' ', '_')
        output_file = f'clock_vs_energy_{clean_filename}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")

        # Close the figure to free memory
        plt.close()


def print_summary_stats(data_dict, target_batch_sizes=[16, 512, 2048]):
    """
    Print summary statistics for all datasets.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
        target_batch_sizes (list): List of batch sizes to analyze
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for filename, df in data_dict.items():
        print(f"\nFile: {filename}")
        print(f"  Total records: {len(df)}")
        print(f"  Clock range: {df['clock'].min()} - {df['clock'].max()} MHz")
        print(
            f"  Energy range: {df['energy_consumption'].min():.3f} - {df['energy_consumption'].max():.3f} J"
        )

        # Show available batch sizes
        batch_sizes = sorted(df['batch_size'].unique())
        print(f"  Available batch sizes: {batch_sizes}")

        # Analyze each target batch size
        for target_batch_size in target_batch_sizes:
            closest_batch_size = min(batch_sizes,
                                     key=lambda x: abs(x - target_batch_size))
            batch_data = df[df['batch_size'] == closest_batch_size]

            if not batch_data.empty:
                min_energy = batch_data['energy_consumption'].min()
                max_energy = batch_data['energy_consumption'].max()
                min_clock = batch_data.loc[
                    batch_data['energy_consumption'].idxmin(), 'clock']
                max_clock = batch_data.loc[
                    batch_data['energy_consumption'].idxmax(), 'clock']

                print(f"  Batch size {closest_batch_size}:")
                print(
                    f"    Energy range: {min_energy:.3f} - {max_energy:.3f} J")
                print(f"    Min energy at: {min_clock} MHz")
                print(f"    Max energy at: {max_clock} MHz")


def print_energy_analysis(data_dict, target_batch_sizes=[16, 512, 2048]):
    """
    Print analysis of energy consumption patterns for each dataset.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
        target_batch_sizes (list): List of batch sizes to analyze
    """
    print("\n" + "=" * 80)
    print("ENERGY CONSUMPTION ANALYSIS")
    print("=" * 80)

    for filename, df in data_dict.items():
        # Extract model name for cleaner output
        parts = filename.replace('dvfs_profile_', '').replace('_one.csv',
                                                              '').split('_')
        if len(parts) >= 3:
            gpu = parts[0] + ' ' + parts[1] + ' ' + parts[2] if parts[
                1] not in ['A100-SXM4-80GB'] else parts[0] + ' ' + parts[1]
            model_parts = [
                p for p in parts[2:] if not any(x in p for x in ['tp', 'pp'])
            ]
            model = '_'.join(model_parts) if model_parts else 'Unknown'
            label = f"{gpu} - {model}"
        else:
            label = filename.replace('_one.csv', '')

        print(f"\n{label}")
        print("-" * len(label))

        available_batch_sizes = sorted(df['batch_size'].unique())

        for target_batch_size in target_batch_sizes:
            closest_batch_size = min(available_batch_sizes,
                                     key=lambda x: abs(x - target_batch_size))
            batch_data = df[df['batch_size'] == closest_batch_size]

            if batch_data.empty:
                continue

            # Group by clock and calculate statistics
            grouped = batch_data.groupby(
                'clock')['energy_consumption'].mean().reset_index()
            min_energy = grouped['energy_consumption'].min()
            max_energy = grouped['energy_consumption'].max()
            min_clock = grouped.loc[grouped['energy_consumption'].idxmin(),
                                    'clock']
            max_clock = grouped.loc[grouped['energy_consumption'].idxmax(),
                                    'clock']

            print(f"  Batch size {closest_batch_size}:")
            print(
                f"    Minimum energy: {min_energy:.3f} J (at {min_clock} MHz)")
            print(
                f"    Maximum energy: {max_energy:.3f} J (at {max_clock} MHz)")
            print(f"    Energy range: {max_energy/min_energy:.1f}x difference")

            # Find the most efficient clock frequencies (within 10% of minimum)
            efficient_threshold = min_energy * 1.1
            efficient_clocks = grouped[grouped['energy_consumption'] <=
                                       efficient_threshold]
            print(
                f"    Efficient clocks (≤{efficient_threshold:.1f} J): {sorted(efficient_clocks['clock'].tolist())}"
            )

            # Show top 5 most efficient configurations
            top_efficient = grouped.nsmallest(5, 'energy_consumption')
            print("    Top 5 most efficient configurations:")
            for _, row in top_efficient.iterrows():
                multiplier = row['energy_consumption'] / min_energy
                print(
                    f"      {int(row['clock']):4d} MHz: {row['energy_consumption']:8.3f} J (x{multiplier:.2f})"
                )


def main():
    """
    Main function to execute the analysis.
    """
    print("Clock vs Energy Analysis")
    print("=" * 50)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define target batch sizes
    target_batch_sizes = [16, 512, 2048]

    # Parse CSV files
    data_dict = parse_csv_files(script_dir)

    if not data_dict:
        print("No valid data found. Exiting.")
        return

    # Print summary statistics
    print_summary_stats(data_dict, target_batch_sizes)

    # Print energy analysis
    print_energy_analysis(data_dict, target_batch_sizes)

    # Create the plots
    plot_clock_versus_energy(data_dict, target_batch_sizes)


if __name__ == "__main__":
    main()
