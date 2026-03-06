# import analyze_.. 
from analyze_single_trace import TraceData
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_trace(trace_data: TraceData, save_path=None):
    """
    Visualize trace data with two rows of plots.
    
    Top row: Additional plots (can be customized)
    Bottom row: Density plots for TTFT, TBT, and TPOT with SLO lines
    
    Args:
        trace_data: TraceData object containing the trace information
        save_path: Path to save the figure (optional)
    """
    # Prepare data
    ttfts = np.array(trace_data.ttfts)
    tpots = np.array(trace_data.tpots)
    
    # Flatten tbts (list of lists) to get all inter-token latencies
    tbts_flat = []
    for tbt_list in trace_data.tbts:
        tbts_flat.extend(tbt_list)
    tbts = np.array(tbts_flat)
    
    # Convert SLO values to float (they're strings)
    ttft_slo = float(trace_data.ttft_slo)
    tbt_slo = float(trace_data.tbt_slo)
    tpot_slo = tbt_slo  # TPOT shares SLO with TBT
    
    # Calculate power
    power_watts = trace_data.energy_consumption / trace_data.total_duration
    
    # Calculate percentages below SLO
    ttft_below_slo = (ttfts <= ttft_slo).sum() / len(ttfts) * 100
    tbt_below_slo = (tbts <= tbt_slo).sum() / len(tbts) * 100
    tpot_below_slo = (tpots <= tpot_slo).sum() / len(tpots) * 100
    
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Top row - Summary statistics
    # Left: Request statistics
    ax_top_left = axes[0, 0]
    ax_top_left.axis('off')
    summary_text = f"""
    Request Statistics
    ━━━━━━━━━━━━━━━━━━━━━━
    Total Requests: {len(ttfts)}
    Total Duration: {trace_data.total_duration:.2f} s
    
    TTFT below SLO: {ttft_below_slo:.1f}%
    TBT below SLO: {tbt_below_slo:.1f}%
    TPOT below SLO: {tpot_below_slo:.1f}%
    """
    ax_top_left.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                     verticalalignment='center')
    
    # Middle: Energy and Power
    ax_top_middle = axes[0, 1]
    ax_top_middle.axis('off')
    energy_text = f"""
    Energy & Power
    ━━━━━━━━━━━━━━━━━━━━━━
    Total Energy: {trace_data.energy_consumption:.2f} J
    Average Power: {power_watts:.2f} W
    Number of GPUs: {trace_data.num_gpus}
    
    Per-GPU Power: {power_watts / trace_data.num_gpus:.2f} W
    """
    ax_top_middle.text(0.1, 0.5, energy_text, fontsize=12, family='monospace',
                       verticalalignment='center')
    
    # Right: SLO Configuration
    ax_top_right = axes[0, 2]
    ax_top_right.axis('off')
    slo_text = f"""
    SLO Configuration
    ━━━━━━━━━━━━━━━━━━━━━━
    TTFT SLO: {ttft_slo:.3f} s
    TBT SLO: {tbt_slo:.3f} s
    TPOT SLO: {tpot_slo:.3f} s
    
    Start Time: {trace_data.start_time}
    End Time: {trace_data.finish_time}
    """
    ax_top_right.text(0.1, 0.5, slo_text, fontsize=12, family='monospace',
                      verticalalignment='center')
    
    # Bottom row - Histograms
    # TTFT histogram
    ax_ttft = axes[1, 0]
    if len(ttfts) > 0:
        ttft_mean = np.mean(ttfts)
        ttft_p90 = np.percentile(ttfts, 90)
        
        n, bins, patches = ax_ttft.hist(ttfts, bins=50, alpha=0.7, color='skyblue', 
                                         edgecolor='black', label='Distribution')
        ax_ttft.axvline(ttft_slo, color='red', linestyle='--', linewidth=2, 
                        label=f'SLO={ttft_slo:.3f}s')
        ax_ttft.axvline(ttft_mean, color='blue', linestyle='-.', linewidth=2,
                        label=f'Mean={ttft_mean:.3f}s')
        ax_ttft.axvline(ttft_p90, color='orange', linestyle=':', linewidth=2,
                        label=f'P90={ttft_p90:.3f}s')
        ax_ttft.text(ttft_slo, ax_ttft.get_ylim()[1] * 0.95, 
                     f'{ttft_below_slo:.1f}% below SLO',
                     ha='left', va='top', fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_ttft.set_xlabel('Time to First Token (s)', fontsize=12)
    ax_ttft.set_ylabel('Frequency', fontsize=12)
    ax_ttft.set_title('TTFT Distribution', fontsize=14, fontweight='bold')
    ax_ttft.legend()
    ax_ttft.grid(True, alpha=0.3, axis='y')
    
    # TBT histogram
    ax_tbt = axes[1, 1]
    if len(tbts) > 0:
        tbt_mean = np.mean(tbts)
        tbt_p90 = np.percentile(tbts, 90)
        
        n, bins, patches = ax_tbt.hist(tbts, bins=50, alpha=0.7, color='lightgreen', 
                                        edgecolor='black', label='Distribution')
        ax_tbt.axvline(tbt_slo, color='red', linestyle='--', linewidth=2,
                       label=f'SLO={tbt_slo:.3f}s')
        ax_tbt.axvline(tbt_mean, color='blue', linestyle='-.', linewidth=2,
                       label=f'Mean={tbt_mean:.3f}s')
        ax_tbt.axvline(tbt_p90, color='orange', linestyle=':', linewidth=2,
                       label=f'P90={tbt_p90:.3f}s')
        ax_tbt.text(tbt_slo, ax_tbt.get_ylim()[1] * 0.95,
                    f'{tbt_below_slo:.1f}% below SLO',
                    ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_tbt.set_xlabel('Time Between Tokens (s)', fontsize=12)
    ax_tbt.set_ylabel('Frequency', fontsize=12)
    ax_tbt.set_title('TBT Distribution', fontsize=14, fontweight='bold')
    ax_tbt.legend()
    ax_tbt.grid(True, alpha=0.3, axis='y')
    
    # TPOT histogram
    ax_tpot = axes[1, 2]
    if len(tpots) > 0:
        tpot_mean = np.mean(tpots)
        tpot_p90 = np.percentile(tpots, 90)
        
        n, bins, patches = ax_tpot.hist(tpots, bins=50, alpha=0.7, color='salmon', 
                                         edgecolor='black', label='Distribution')
        ax_tpot.axvline(tpot_slo, color='red', linestyle='--', linewidth=2,
                        label=f'SLO={tpot_slo:.3f}s')
        ax_tpot.axvline(tpot_mean, color='blue', linestyle='-.', linewidth=2,
                        label=f'Mean={tpot_mean:.3f}s')
        ax_tpot.axvline(tpot_p90, color='orange', linestyle=':', linewidth=2,
                        label=f'P90={tpot_p90:.3f}s')
        ax_tpot.text(tpot_slo, ax_tpot.get_ylim()[1] * 0.95,
                     f'{tpot_below_slo:.1f}% below SLO',
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_tpot.set_xlabel('Time Per Output Token (s)', fontsize=12)
    ax_tpot.set_ylabel('Frequency', fontsize=12)
    ax_tpot.set_title('TPOT Distribution', fontsize=14, fontweight='bold')
    ax_tpot.legend()
    ax_tpot.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Print power information
    print(f"\n{'='*60}")
    print(f"POWER CONSUMPTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Energy Consumption: {trace_data.energy_consumption:.2f} J")
    print(f"Total Duration: {trace_data.total_duration:.2f} s")
    print(f"Average Power: {power_watts:.2f} W")
    print(f"Number of GPUs: {trace_data.num_gpus}")
    print(f"Per-GPU Average Power: {power_watts / trace_data.num_gpus:.2f} W")
    print(f"{'='*60}\n")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize trace data from benchmark results"
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="Base directory containing the traced_dataset folder (relative or absolute path)"
    )
 
    args = parser.parse_args()
    
    # Load trace data
    trace_data = TraceData(args.base_directory)
    
    # Create visualization directory
    viz_dir = os.path.join(args.base_directory, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate filename based on the trace data
    # Extract the experiment name from the base directory
    dir_name = os.path.basename(os.path.normpath(args.base_directory))
    output_filename = f"trace_visualization_{dir_name}.png"
    save_path = os.path.join(viz_dir, output_filename)
    
    # Generate visualization
    visualize_trace(trace_data, save_path=save_path)
