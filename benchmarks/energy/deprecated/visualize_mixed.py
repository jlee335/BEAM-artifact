import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_mixed_profile(csv_file):
    """
    Visualize mixed profile results with dual y-axes.
    
    Args:
        csv_file: Path to the CSV file containing profile results
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter only valid entries
    df = df[df['valid'] == 1]
    
    # Filter for only prefill_tokens 0 and 200
    df = df[df['prefill_tokens'].isin([1])]
    
    
    # Get unique clock and prefill_tokens values
    clock_values = sorted(df['clock'].unique(), reverse=True)
    prefill_tokens_values = sorted(df['prefill_tokens'].unique())
    
    print(f"Clock frequencies: {clock_values} MHz")
    print(f"Prefill token values: {prefill_tokens_values}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Define colors for different clock values (gradient from dark to light)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(clock_values)))
    
    # Define line styles for different prefill_tokens values
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    # Plot 1: Latency
    for i, clock_val in enumerate(clock_values):
        for j, prefill_val in enumerate(prefill_tokens_values):
            subset = df[(df['clock'] == clock_val) & (df['prefill_tokens'] == prefill_val)].sort_values('total_ctx_len')
            
            if len(subset) > 0:
                ax1.plot(subset['total_ctx_len'], subset['time_taken'], 
                        marker=markers[j % len(markers)], 
                        linestyle=line_styles[j % len(line_styles)], 
                        linewidth=2.5, markersize=7,
                        color=colors[i], alpha=0.8,
                        label=f'{clock_val} MHz, Prefill {prefill_val} tokens')
    
    ax1.set_xlabel('Total Context Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency vs Total Context Length', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, framealpha=0.9, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Energy Consumption
    for i, clock_val in enumerate(clock_values):
        for j, prefill_val in enumerate(prefill_tokens_values):
            subset = df[(df['clock'] == clock_val) & (df['prefill_tokens'] == prefill_val)].sort_values('total_ctx_len')
            
            if len(subset) > 0:
                ax2.plot(subset['total_ctx_len'], subset['energy_consumption'], 
                        marker=markers[j % len(markers)], 
                        linestyle=line_styles[j % len(line_styles)], 
                        linewidth=2.5, markersize=7,
                        color=colors[i], alpha=0.8,
                        label=f'{clock_val} MHz, Prefill {prefill_val} tokens')
    
    ax2.set_xlabel('Total Context Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy Consumption (mJ)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Consumption vs Total Context Length', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, framealpha=0.9, loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    # Add overall title
    model_name = os.path.basename(csv_file).replace('mixed_profile_', '').replace('.csv', '')
    fig.suptitle(f'Performance Metrics vs Total Context Length\n{model_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    output_file = csv_file.replace('.csv', '_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Also save as PDF for better quality
    output_pdf = csv_file.replace('.csv', '_visualization.pdf')
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"PDF version saved to: {output_pdf}")
    
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the current file
        csv_file = "/workspace/disagg/energy-inf-v1-disagg/benchmarks/energy/offline_profile_results/mixed_profile_NVIDIA RTX A6000_Qwen_Qwen2.5-32B_tp2_pp2.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    
    visualize_mixed_profile(csv_file)

