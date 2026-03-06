#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visualize mixed profile data: latency vs clock speed
Multiple lines per context length, separate graphs per batch size

Usage:
    python test_visualize_mixed.py <csv_file_path>
    
Example:
    python test_visualize_mixed.py "mixed_profile_NVIDIA RTX A6000_Qwen_Qwen2.5-32B_tp1_pp4.csv"
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Output directory
OUTPUT_DIR = "temp_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(csv_file):
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    # Read the CSV file
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Extract base filename without extension for output files
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Filter valid entries
    df = df[df['valid'] == 1]
    
    # Get unique batch sizes and context lengths
    batch_sizes = sorted(df['batch_size'].unique())
    ctx_lens = sorted(df['total_ctx_len'].unique())
    
    print(f"Batch sizes: {batch_sizes}")
    print(f"Context lengths: {ctx_lens}")
    
    # Create a color map for different context lengths
    colors = plt.cm.tab10(np.linspace(0, 1, len(ctx_lens)))
    
    # Calculate grid layout for subplots
    n_batch_sizes = len(batch_sizes)
    n_cols = 4  # 4 columns
    n_rows = (n_batch_sizes + n_cols - 1) // n_cols  # Ceiling division
    
    # ========== PLOT 1: LATENCY VS CLOCK SPEED ==========
    print("\nGenerating latency plot...")
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes1 = axes1.reshape(1, -1)
    axes1_flat = axes1.flatten()
    
    # Plot for each batch size
    for idx, batch_size in enumerate(batch_sizes):
        ax = axes1_flat[idx]
        
        # Filter data for this batch size
        df_batch = df[df['batch_size'] == batch_size]
        
        # Plot a line for each context length
        for ctx_idx, ctx_len in enumerate(ctx_lens):
            df_ctx = df_batch[df_batch['total_ctx_len'] == ctx_len]
            
            if len(df_ctx) > 0:
                # Sort by clock speed
                df_ctx = df_ctx.sort_values('clock')
                
                # Plot
                ax.plot(df_ctx['clock'], df_ctx['time_taken'], 
                       marker='o', label=f'ctx_len={ctx_len}',
                       color=colors[ctx_idx], linewidth=2, markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Clock Speed (MHz)', fontsize=10)
        ax.set_ylabel('Latency (seconds)', fontsize=10)
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)  # Start y-axis from 0
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    # Hide unused subplots
    for idx in range(len(batch_sizes), len(axes1_flat)):
        axes1_flat[idx].set_visible(False)
    
    # Overall title
    fig1.suptitle('Latency vs Clock Speed (Multiple Context Lengths)', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_path1 = os.path.join(OUTPUT_DIR, f'{base_filename}_latency_vs_clock.png')
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"Saved latency plot to: {output_path1}")
    
    plt.close(fig1)
    
    # ========== PLOT 2: ENERGY VS CLOCK SPEED ==========
    print("\nGenerating energy plot...")
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    axes2_flat = axes2.flatten()
    
    # Plot for each batch size
    for idx, batch_size in enumerate(batch_sizes):
        ax = axes2_flat[idx]
        
        # Filter data for this batch size
        df_batch = df[df['batch_size'] == batch_size]
        
        # Plot a line for each context length
        for ctx_idx, ctx_len in enumerate(ctx_lens):
            df_ctx = df_batch[df_batch['total_ctx_len'] == ctx_len]
            
            if len(df_ctx) > 0:
                # Sort by clock speed
                df_ctx = df_ctx.sort_values('clock')
                
                # Plot
                ax.plot(df_ctx['clock'], df_ctx['energy_consumption'], 
                       marker='o', label=f'ctx_len={ctx_len}',
                       color=colors[ctx_idx], linewidth=2, markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Clock Speed (MHz)', fontsize=10)
        ax.set_ylabel('Energy (mJ)', fontsize=10)
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)  # Start y-axis from 0
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    # Hide unused subplots
    for idx in range(len(batch_sizes), len(axes2_flat)):
        axes2_flat[idx].set_visible(False)
    
    # Overall title
    fig2.suptitle('Energy Consumption vs Clock Speed (Multiple Context Lengths)', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_path2 = os.path.join(OUTPUT_DIR, f'{base_filename}_energy_vs_clock.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved energy plot to: {output_path2}")
    
    plt.close(fig2)
    
    print(f"\nAll plots created successfully!")
    print(f"Each plot contains {len(batch_sizes)} batch sizes with {len(ctx_lens)} context lengths")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_visualize_mixed.py <csv_file_path>")
        print("\nExample:")
        print('  python test_visualize_mixed.py "mixed_profile_NVIDIA RTX A6000_Qwen_Qwen2.5-32B_tp1_pp4.csv"')
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    main(csv_file_path)

