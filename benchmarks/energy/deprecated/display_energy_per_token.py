#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Script to parse CSV files and plot Energy (J) per token vs batch_size.
Energy per token is calculated as energy_consumption / batch_size.
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
            required_columns = ['batch_size', 'energy_consumption']
            if not all(col in df.columns for col in required_columns):
                print(
                    f"Warning: {filename} missing required columns. Available columns: {list(df.columns)}"
                )
                continue

            # Calculate energy per token
            df['energy_per_token'] = df['energy_consumption'] / df['batch_size']

            # Filter out rows where batch_size is 0 to avoid division by zero
            df = df[df['batch_size'] > 0]

            data_dict[filename] = df

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return data_dict


def plot_energy_per_token(data_dict):
    """
    Create individual plots of Energy per token vs batch_size for each dataset.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
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

        # Create individual plot for this dataset (3:1 ratio, compact)
        plt.figure(figsize=(9, 3))

        # Group by batch_size and calculate mean energy per token (in case of multiple entries)
        grouped = df.groupby('batch_size')['energy_per_token'].agg(
            ['mean', 'std']).reset_index()

        # Find the minimum energy per token for multiplier calculation
        min_energy = grouped['mean'].min()

        # Calculate multipliers
        grouped['multiplier'] = grouped['mean'] / min_energy

        # Plot with connecting lines (no error bars) - bold styling
        plt.plot(grouped['batch_size'],
                 grouped['mean'],
                 marker='o',
                 linewidth=3,
                 markersize=8,
                 color='blue',
                 markeredgewidth=1.5,
                 markeredgecolor='black')

        # Add multiplier labels for each datapoint
        for idx, row in grouped.iterrows():
            batch_size = row['batch_size']
            energy = row['mean']
            multiplier = row['multiplier']

            # Format multiplier text
            if multiplier < 1.1 or multiplier < 10:
                multiplier_text = f"x{multiplier:.1f}"
            else:
                multiplier_text = f"x{multiplier:.0f}"

            # Position the label above the point
            plt.annotate(
                multiplier_text,
                xy=(batch_size, energy),
                xytext=(0, 8),  # 8 points above
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold')

        plt.xlabel('Batch Size', fontsize=11, fontweight='bold')
        plt.ylabel('Energy (J) per Token', fontsize=11, fontweight='bold')
        plt.title('Energy Consumption per Token vs Batch Size',
                  fontsize=12,
                  fontweight='bold')
        plt.grid(True, alpha=0.4, linewidth=0.8)

        # Set log scale for y-axis only (x-axis is linear)
        plt.yscale('log')

        # Set x-axis range to 0-2048
        ax = plt.gca()
        plt.xlim(0, 2048)

        # Adjust layout for compact design
        plt.tight_layout(pad=0.5)

        # Save the plot with a clean filename
        clean_filename = filename.replace('dvfs_profile_',
                                          '').replace('_one.csv',
                                                      '').replace(' ', '_')
        output_file = f'energy_per_token_{clean_filename}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")

        # Close the figure to free memory
        plt.close()


def print_summary_stats(data_dict):
    """
    Print summary statistics for all datasets.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for filename, df in data_dict.items():
        print(f"\nFile: {filename}")
        print(f"  Total records: {len(df)}")
        print(
            f"  Batch size range: {df['batch_size'].min()} - {df['batch_size'].max()}"
        )
        print(
            f"  Energy per token range: {df['energy_per_token'].min():.3f} - {df['energy_per_token'].max():.3f} J"
        )
        print(
            f"  Mean energy per token: {df['energy_per_token'].mean():.3f} J")

        # Show batch size distribution
        batch_sizes = sorted(df['batch_size'].unique())
        print(f"  Available batch sizes: {batch_sizes}")


def print_multiplier_analysis(data_dict):
    """
    Print analysis of energy multipliers for each dataset.
    
    Args:
        data_dict (dict): Dictionary with filename as key and DataFrame as value
    """
    print("\n" + "=" * 80)
    print("ENERGY MULTIPLIER ANALYSIS")
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

        # Group by batch_size and calculate statistics
        grouped = df.groupby('batch_size')['energy_per_token'].agg(
            ['mean', 'std']).reset_index()
        min_energy = grouped['mean'].min()
        max_energy = grouped['mean'].max()
        min_batch = grouped.loc[grouped['mean'].idxmin(), 'batch_size']
        max_batch = grouped.loc[grouped['mean'].idxmax(), 'batch_size']

        print(
            f"  Minimum energy: {min_energy:.3f} J/token (batch_size={min_batch})"
        )
        print(
            f"  Maximum energy: {max_energy:.3f} J/token (batch_size={max_batch})"
        )
        print(f"  Energy range: {max_energy/min_energy:.1f}x difference")

        # Find the most efficient batch sizes (within 10% of minimum)
        efficient_threshold = min_energy * 1.1
        efficient_batches = grouped[grouped['mean'] <= efficient_threshold]
        print(
            f"  Efficient batch sizes (≤{efficient_threshold:.1f} J/token): {sorted(efficient_batches['batch_size'].tolist())}"
        )

        # Show top 5 most efficient configurations
        top_efficient = grouped.nsmallest(5, 'mean')
        print("  Top 5 most efficient configurations:")
        for _, row in top_efficient.iterrows():
            multiplier = row['mean'] / min_energy
            print(
                f"    batch_size={int(row['batch_size']):4d}: {row['mean']:8.3f} J/token (x{multiplier:.2f})"
            )


def main():
    """
    Main function to execute the analysis.
    """
    print("Energy per Token Analysis")
    print("=" * 50)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse CSV files
    data_dict = parse_csv_files(script_dir)

    if not data_dict:
        print("No valid data found. Exiting.")
        return

    # Print summary statistics
    print_summary_stats(data_dict)

    # Print multiplier analysis
    print_multiplier_analysis(data_dict)

    # Create the plot
    plot_energy_per_token(data_dict)


if __name__ == "__main__":
    main()
