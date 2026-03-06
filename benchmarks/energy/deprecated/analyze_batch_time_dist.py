#!/usr/bin/env python3
"""
Analyze batch elapsed time distribution from batch_log_GPU_0.csv files.
Calculates mean, P50, and P99 statistics and generates a histogram.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_batch_times(csv_path: str, output_dir: str = None):
    """
    Analyze elapsed times from batch log CSV file.
    
    Args:
        csv_path: Path to the batch_log_GPU_0.csv file
        output_dir: Directory to save histogram (defaults to same dir as CSV)
    """
    # Read CSV file
    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract elapsed_time column
    if 'elapsed_time' not in df.columns:
        raise ValueError(f"CSV file does not contain 'elapsed_time' column. Available columns: {df.columns.tolist()}")
    
    elapsed_times = df['elapsed_time'].values
    
    # Calculate statistics
    mean_time = np.mean(elapsed_times)
    p50_time = np.percentile(elapsed_times, 50)
    p99_time = np.percentile(elapsed_times, 99)
    min_time = np.min(elapsed_times)
    max_time = np.max(elapsed_times)
    std_time = np.std(elapsed_times)
    
    # Print statistics
    print("\n" + "="*70)
    print("Batch Elapsed Time Statistics")
    print("="*70)
    print(f"Total samples: {len(elapsed_times)}")
    print(f"Mean:          {mean_time:.4f} ms")
    print(f"P50 (Median):  {p50_time:.4f} ms")
    print(f"P99:           {p99_time:.4f} ms")
    print(f"Min:           {min_time:.4f} ms")
    print(f"Max:           {max_time:.4f} ms")
    print(f"Std Dev:       {std_time:.4f} ms")
    print("="*70)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histogram
    n, bins, patches = ax.hist(elapsed_times, bins=100, alpha=0.7, color='blue', edgecolor='black')
    
    # Add vertical lines for statistics
    ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f} ms')
    ax.axvline(p50_time, color='green', linestyle='--', linewidth=2, label=f'P50: {p50_time:.2f} ms')
    ax.axvline(p99_time, color='orange', linestyle='--', linewidth=2, label=f'P99: {p99_time:.2f} ms')
    
    # Labels and title
    ax.set_xlabel('Elapsed Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Batch Elapsed Time Distribution\n{Path(csv_path).parent.name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with additional stats
    textstr = f'Samples: {len(elapsed_times)}\nMin: {min_time:.2f} ms\nMax: {max_time:.2f} ms\nStd: {std_time:.2f} ms'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # Save figure
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    output_filename = f"batch_elapsed_time_histogram_{Path(csv_path).parent.name}.png"
    output_path = output_dir / output_filename
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nHistogram saved to: {output_path}")
    
    # Also create a zoomed-in version excluding outliers (below P99)
    elapsed_times_zoom = elapsed_times[elapsed_times <= p99_time]
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # Plot histogram (zoomed)
    n2, bins2, patches2 = ax2.hist(elapsed_times_zoom, bins=100, alpha=0.7, color='blue', edgecolor='black')
    
    # Add vertical lines for statistics
    ax2.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f} ms')
    ax2.axvline(p50_time, color='green', linestyle='--', linewidth=2, label=f'P50: {p50_time:.2f} ms')
    ax2.axvline(p99_time, color='orange', linestyle='--', linewidth=2, label=f'P99: {p99_time:.2f} ms')
    
    # Labels and title
    ax2.set_xlabel('Elapsed Time (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Batch Elapsed Time Distribution (Zoomed to P99)\n{Path(csv_path).parent.name}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add text box
    textstr2 = f'Samples shown: {len(elapsed_times_zoom)} ({len(elapsed_times_zoom)/len(elapsed_times)*100:.1f}%)\nExcluding outliers > P99'
    ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save zoomed figure
    output_filename_zoom = f"batch_elapsed_time_histogram_{Path(csv_path).parent.name}_zoomed.png"
    output_path_zoom = output_dir / output_filename_zoom
    
    plt.tight_layout()
    plt.savefig(output_path_zoom, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Zoomed histogram saved to: {output_path_zoom}")
    
    return {
        'mean': mean_time,
        'p50': p50_time,
        'p99': p99_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'count': len(elapsed_times)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze batch elapsed time distribution from batch_log CSV files'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to batch_log_GPU_0.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save histogram (defaults to same directory as CSV)'
    )
    
    args = parser.parse_args()
    
    analyze_batch_times(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()


