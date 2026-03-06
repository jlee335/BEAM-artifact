#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Script to parse CSV files and plot Time Taken (s) vs Batch Size.
Time taken is plotted for batch sizes ≤ 256 at specific clock frequencies (1230, 1530, 1830 MHz).
"""

import glob
import os

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
            required_columns = ['clock', 'batch_size', 'time_taken']
            if not all(col in df.columns for col in required_columns):
                print(
                    f"Warning: {filename} missing required columns. Available columns: {list(df.columns)}"
                )
                continue

            data_dict[filename] = df

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return data_dict


def plot_time_versus_batch_size(data_dict, target_clocks=[1230, 1530, 1830]):
    """
    Create plots of Time Taken (s) vs Batch Size for specific clock frequencies.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
        target_clocks (list): List of clock frequencies to plot
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

        # Create single plot with 3 lines
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Filter data for batch sizes ≤ 512
        df_filtered = df[df['batch_size'] <= 512].copy()

        # Get available clock frequencies in the data
        available_clocks = sorted(df_filtered['clock'].unique())
        print(
            f"\nAvailable clock frequencies for {filename}: {available_clocks}"
        )

        # Define colors for each clock frequency
        colors = ['blue', 'red', 'green']

        for i, target_clock in enumerate(target_clocks):
            # Find the closest available clock frequency
            closest_clock = min(available_clocks,
                                key=lambda x: abs(x - target_clock))

            if abs(closest_clock - target_clock
                   ) > target_clock * 0.1:  # If difference is more than 10%
                print(
                    f"Warning: No close match for clock {target_clock}, using {closest_clock}"
                )

            # Filter data for this clock frequency
            clock_data = df_filtered[df_filtered['clock'] ==
                                     closest_clock].copy()

            if clock_data.empty:
                print(f"No data found for clock {closest_clock}")
                continue

            # Group by batch_size and calculate mean time_taken (in case of multiple entries)
            grouped = clock_data.groupby('batch_size')['time_taken'].agg(
                ['mean', 'std']).reset_index()
            grouped = grouped.sort_values('batch_size')

            # Plot with connecting lines and error bars
            ax.plot(grouped['batch_size'],
                    grouped['mean'],
                    marker='o',
                    linewidth=3,
                    markersize=8,
                    color=colors[i],
                    markeredgewidth=1.5,
                    markeredgecolor='black',
                    label=f'{closest_clock} MHz')

            # Add error bars if there's variation
            if len(grouped) > 1 and grouped['std'].max() > 0:
                ax.errorbar(grouped['batch_size'],
                            grouped['mean'],
                            yerr=grouped['std'],
                            fmt='none',
                            color=colors[i],
                            alpha=0.5,
                            capsize=5)

            # Highlight minimum time point with a thick dot
            min_time_idx = grouped['mean'].idxmin()
            min_time_batch = grouped.loc[min_time_idx, 'batch_size']
            min_time_value = grouped.loc[min_time_idx, 'mean']
            ax.plot(min_time_batch,
                    min_time_value,
                    marker='o',
                    markersize=15,
                    color='orange',
                    markeredgewidth=2,
                    markeredgecolor='black',
                    zorder=5)

        ax.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax.set_ylabel('Time Taken (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'Time vs Batch Size - {title}',
                     fontsize=12,
                     fontweight='bold')
        ax.grid(True, alpha=0.4, linewidth=0.8)
        ax.legend(fontsize=10)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Set x-axis range based on available batch sizes
        all_batch_sizes = df_filtered['batch_size'].unique()
        min_batch = all_batch_sizes.min()
        max_batch = all_batch_sizes.max()
        batch_range = max_batch - min_batch
        ax.set_xlim(min_batch - batch_range * 0.05,
                    max_batch + batch_range * 0.05)

        # Adjust layout
        plt.tight_layout(pad=1.0)

        # Save the plot with a clean filename
        clean_filename = filename.replace('dvfs_profile_',
                                          '').replace('_one.csv',
                                                      '').replace(' ', '_')
        output_file = f'time_vs_batch_size_{clean_filename}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")

        # Close the figure to free memory
        plt.close()


def print_summary_stats(data_dict, target_clocks=[1230, 1530, 1830]):
    """
    Print summary statistics for all datasets.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
        target_clocks (list): List of clock frequencies to analyze
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for filename, df in data_dict.items():
        print(f"\nFile: {filename}")
        # Filter data for batch sizes ≤ 512
        df_filtered = df[df['batch_size'] <= 512]

        print(f"  Total records: {len(df)}")
        print(f"  Records with batch_size ≤ 512: {len(df_filtered)}")
        print(f"  Clock range: {df['clock'].min()} - {df['clock'].max()} MHz")
        print(
            f"  Time range: {df['time_taken'].min():.6f} - {df['time_taken'].max():.6f} s"
        )

        # Show available batch sizes ≤ 512
        batch_sizes = sorted(df_filtered['batch_size'].unique())
        print(f"  Available batch sizes ≤ 512: {batch_sizes}")

        # Analyze each target clock frequency
        for target_clock in target_clocks:
            available_clocks = sorted(df_filtered['clock'].unique())
            closest_clock = min(available_clocks,
                                key=lambda x: abs(x - target_clock))
            clock_data = df_filtered[df_filtered['clock'] == closest_clock]

            if not clock_data.empty:
                min_time = clock_data['time_taken'].min()
                max_time = clock_data['time_taken'].max()
                min_batch = clock_data.loc[clock_data['time_taken'].idxmin(),
                                           'batch_size']
                max_batch = clock_data.loc[clock_data['time_taken'].idxmax(),
                                           'batch_size']

                print(f"  Clock {closest_clock} MHz:")
                print(f"    Time range: {min_time:.6f} - {max_time:.6f} s")
                print(f"    Min time at: batch_size {min_batch}")
                print(f"    Max time at: batch_size {max_batch}")


def print_time_analysis(data_dict, target_clocks=[1230, 1530, 1830]):
    """
    Print analysis of time consumption patterns for each dataset.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
        target_clocks (list): List of clock frequencies to analyze
    """
    print("\n" + "=" * 80)
    print("TIME CONSUMPTION ANALYSIS")
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

        # Filter data for batch sizes ≤ 512
        df_filtered = df[df['batch_size'] <= 512]
        available_clocks = sorted(df_filtered['clock'].unique())

        for target_clock in target_clocks:
            closest_clock = min(available_clocks,
                                key=lambda x: abs(x - target_clock))
            clock_data = df_filtered[df_filtered['clock'] == closest_clock]

            if clock_data.empty:
                continue

            # Group by batch_size and calculate statistics
            grouped = clock_data.groupby(
                'batch_size')['time_taken'].mean().reset_index()
            min_time = grouped['time_taken'].min()
            max_time = grouped['time_taken'].max()
            min_batch = grouped.loc[grouped['time_taken'].idxmin(),
                                    'batch_size']
            max_batch = grouped.loc[grouped['time_taken'].idxmax(),
                                    'batch_size']

            print(f"  Clock {closest_clock} MHz:")
            print(
                f"    Minimum time: {min_time:.6f} s (at batch_size {min_batch})"
            )
            print(
                f"    Maximum time: {max_time:.6f} s (at batch_size {max_batch})"
            )
            print(f"    Time range: {max_time/min_time:.1f}x difference")

            # Find the most efficient batch sizes (within 10% of minimum)
            efficient_threshold = min_time * 1.1
            efficient_batches = grouped[grouped['time_taken'] <=
                                        efficient_threshold]
            print(
                f"    Efficient batch sizes (≤{efficient_threshold:.6f} s): {sorted(efficient_batches['batch_size'].tolist())}"
            )

            # Show top 5 most efficient configurations
            top_efficient = grouped.nsmallest(5, 'time_taken')
            print("    Top 5 most efficient configurations:")
            for _, row in top_efficient.iterrows():
                multiplier = row['time_taken'] / min_time
                print(
                    f"      batch_size={int(row['batch_size']):4d}: {row['time_taken']:10.6f} s (x{multiplier:.2f})"
                )


def main():
    """
    Main function to execute the analysis.
    """
    print("Time vs Batch Size Analysis")
    print("=" * 50)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define target clock frequencies
    target_clocks = [1230, 1530, 1830]

    # Parse CSV files
    data_dict = parse_csv_files(script_dir)

    if not data_dict:
        print("No valid data found. Exiting.")
        return

    # Print summary statistics
    print_summary_stats(data_dict, target_clocks)

    # Print time analysis
    print_time_analysis(data_dict, target_clocks)

    # Create the plots
    plot_time_versus_batch_size(data_dict, target_clocks)


if __name__ == "__main__":
    main()
