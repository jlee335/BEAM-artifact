# SPDX-License-Identifier: Apache-2.0
"""
Detailed version of the tradeoff analysis script.

This script creates individual plots for each CSV file to better visualize
the batching effects on latency and energy per request.
"""

import matplotlib
import pandas as pd

matplotlib.use('Agg')  # Use non-interactive backend
import glob
import os

import matplotlib.pyplot as plt


def load_and_process_csv(file_path):
    """Load CSV file and calculate per-request metrics."""
    df = pd.read_csv(file_path)

    # Filter valid entries
    df = df[df['valid'] == 1]

    # Calculate per-request metrics
    df['latency_per_request'] = (
        df['time_taken'] / df['batch_size']) * 1000  # Convert to milliseconds
    df['energy_per_request'] = (df['energy_consumption'] /
                                df['batch_size']) / 1000

    return df


def create_individual_csv_plots(csv_files):
    """Create individual plots for each CSV file."""

    # Create output directory
    output_dir = "individual_csv_analysis"
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in csv_files:
        try:
            # Load and process data
            df = load_and_process_csv(csv_file)

            # Extract model name from filename
            filename = os.path.basename(csv_file)
            model_name = filename.replace('dvfs_profile_',
                                          '').replace('_one.csv', '')

            # Group by batch size and calculate mean values
            grouped = df.groupby('batch_size').agg({
                'latency_per_request': 'mean',
                'energy_per_request': 'mean',
                'time_taken': 'mean',
                'energy_consumption': 'mean'
            }).reset_index()

            # Set publication-quality style
            plt.style.use('default')
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.linewidth'] = 1.5
            plt.rcParams['xtick.major.width'] = 1.5
            plt.rcParams['ytick.major.width'] = 1.5
            plt.rcParams['xtick.major.size'] = 6
            plt.rcParams['ytick.major.size'] = 6

            # Create figure with vertically stacked subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            fig.suptitle(f'{model_name}',
                         fontsize=14,
                         fontweight='bold',
                         y=0.98)

            # Plot 1: Latency per request vs batch size
            ax1.plot(grouped['batch_size'],
                     grouped['latency_per_request'],
                     marker='o',
                     linewidth=3,
                     markersize=10,
                     color='#1f77b4',
                     markerfacecolor='#1f77b4',
                     markeredgewidth=1.5,
                     markeredgecolor='#1f77b4')
            ax1.set_ylabel('Latency per Token (ms/token)',
                           fontsize=11,
                           fontweight='bold')
            ax1.set_title('Latency per Request',
                          fontsize=12,
                          fontweight='bold',
                          pad=10)
            ax1.grid(True, alpha=0.4, linewidth=0.8)
            ax1.set_yscale('log')
            ax1.tick_params(axis='both',
                            which='major',
                            labelsize=10,
                            width=1.5,
                            length=6)

            # Add value annotations for specific datapoints on latency plot
            target_batch_sizes = [1, 128, 512, 1024, 2048, 4096]
            for batch_size in target_batch_sizes:
                if batch_size in grouped['batch_size'].values:
                    value = grouped[grouped['batch_size'] ==
                                    batch_size]['latency_per_request'].iloc[0]
                    ax1.annotate(f'{value:.3f}',
                                 xy=(batch_size, value),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center',
                                 va='bottom',
                                 fontsize=9,
                                 fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white',
                                           alpha=0.8,
                                           edgecolor='#1f77b4'))

            # Plot 2: Energy per request vs batch size
            ax2.plot(grouped['batch_size'],
                     grouped['energy_per_request'],
                     marker='s',
                     linewidth=3,
                     markersize=10,
                     color='#d62728',
                     markerfacecolor='#d62728',
                     markeredgewidth=1.5,
                     markeredgecolor='#d62728')
            ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Energy per Token (J/token)',
                           fontsize=11,
                           fontweight='bold')
            ax2.set_title('Energy per Request',
                          fontsize=12,
                          fontweight='bold',
                          pad=10)
            ax2.grid(True, alpha=0.4, linewidth=0.8)
            ax2.set_yscale('log')
            ax2.tick_params(axis='both',
                            which='major',
                            labelsize=10,
                            width=1.5,
                            length=6)

            # Add value annotations for specific datapoints on energy plot
            for batch_size in target_batch_sizes:
                if batch_size in grouped['batch_size'].values:
                    value = grouped[grouped['batch_size'] ==
                                    batch_size]['energy_per_request'].iloc[0]
                    ax2.annotate(f'{value:.3f}',
                                 xy=(batch_size, value),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center',
                                 va='bottom',
                                 fontsize=9,
                                 fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white',
                                           alpha=0.8,
                                           edgecolor='#d62728'))

            # Set x-axis limits for both plots
            ax1.set_xlim(0, 4096)
            ax2.set_xlim(0, 4096)

            plt.tight_layout()

            # Save the plot
            safe_model_name = model_name.replace(' ', '_').replace('/', '_')
            output_path = os.path.join(
                output_dir, f'batching_analysis_{safe_model_name}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved as: {output_path}")

            plt.close()  # Close to free memory

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue


def create_summary_statistics(csv_files):
    """Create a summary table of key statistics."""

    summary_data = []

    for csv_file in csv_files:
        try:
            df = load_and_process_csv(csv_file)

            filename = os.path.basename(csv_file)
            model_name = filename.replace('dvfs_profile_',
                                          '').replace('_one.csv', '')

            # Calculate statistics for each batch size
            for batch_size in df['batch_size'].unique():
                batch_data = df[df['batch_size'] == batch_size]

                summary_data.append({
                    'Model':
                    model_name,
                    'Batch_Size':
                    batch_size,
                    'Avg_Latency_per_Request':
                    batch_data['latency_per_request'].mean(),
                    'Std_Latency_per_Request':
                    batch_data['latency_per_request'].std(),
                    'Avg_Energy_per_Request_kJ':
                    batch_data['energy_per_request'].mean(),
                    'Std_Energy_per_Request_kJ':
                    batch_data['energy_per_request'].std(),
                    'Min_Latency_per_Request':
                    batch_data['latency_per_request'].min(),
                    'Max_Latency_per_Request':
                    batch_data['latency_per_request'].max(),
                    'Min_Energy_per_Request_kJ':
                    batch_data['energy_per_request'].min(),
                    'Max_Energy_per_Request_kJ':
                    batch_data['energy_per_request'].max()
                })

        except Exception as e:
            print(f"Error processing {csv_file} for summary: {e}")
            continue

    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    output_dir = "individual_csv_analysis"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'batching_summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved as: {summary_path}")

    return summary_df


def main():
    """Main function to find CSV files and create plots."""

    print("Starting analysis...")

    # Find all _one.csv files in the current directory
    csv_files = glob.glob('*_one.csv')

    if not csv_files:
        print("No *_one.csv files found in the current directory.")
        return

    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {file}")

    # Create individual CSV plots
    print("\nCreating individual CSV plots...")
    create_individual_csv_plots(csv_files)

    # Create summary statistics
    print("\nCreating summary statistics...")
    summary_df = create_summary_statistics(csv_files)

    # Display some key insights
    print("\n=== Key Insights ===")
    print(f"Total models analyzed: {len(csv_files)}")
    print(f"Total data points: {len(summary_df)}")

    # Find optimal batch sizes for different metrics
    if not summary_df.empty:
        min_latency_idx = summary_df['Avg_Latency_per_Request'].idxmin()
        min_energy_idx = summary_df['Avg_Energy_per_Request_kJ'].idxmin()

        print("\nBest latency per request:")
        print(f"  Model: {summary_df.loc[min_latency_idx, 'Model']}")
        print(f"  Batch Size: {summary_df.loc[min_latency_idx, 'Batch_Size']}")
        print(
            f"  Latency: {summary_df.loc[min_latency_idx, 'Avg_Latency_per_Request']:.1f} ms"
        )

        print("\nBest energy per request:")
        print(f"  Model: {summary_df.loc[min_energy_idx, 'Model']}")
        print(f"  Batch Size: {summary_df.loc[min_energy_idx, 'Batch_Size']}")
        print(
            f"  Energy: {summary_df.loc[min_energy_idx, 'Avg_Energy_per_Request_kJ']:.2f} kJ"
        )

    print("Analysis complete!")


if __name__ == "__main__":
    main()
