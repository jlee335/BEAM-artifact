import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_single_trace import TraceData
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re


def get_trace_alias(trace_name):
    """
    Convert trace name to a shorter alias for display.
    
    Args:
        trace_name: Full trace directory name
        
    Returns:
        Shorter alias for display
    """
    # Create aliases based on common patterns
    aliases = {
        's1_s2_fixed_chunk_512_clock_1830': 'TBT-based',
        'tpot_based_fixed_chunk_512_clock_1830': 'TPOT-based',
        'vanilla_fixed_chunk_512_clock_1830': 'Vanilla',
    }
    
    # Return alias if exists, otherwise try to create one
    if trace_name in aliases:
        return aliases[trace_name]
    
    # Try to extract chunk size from mbt<number> pattern
    chunk_size_match = re.search(r'mbt(\d+)', trace_name.lower())
    chunk_size = chunk_size_match.group(1) if chunk_size_match else None
    
    # Try to extract meaningful parts
    parts = trace_name.split('_')
    if 's1' in parts and 's2' in parts:
        if chunk_size:
            return f'TBT-based_{chunk_size}'
        return 'TBT-based'
    elif 'tpot' in trace_name.lower():
        if chunk_size:
            return f'TPOT-based_{chunk_size}'
        return 'TPOT-based'
    elif 'vanilla' in trace_name.lower():
        if chunk_size:
            return f'Vanilla_{chunk_size}'
        return 'Vanilla'
    
    # Fall back to abbreviated version
    return trace_name[:15]


def load_multiple_traces(parent_directory):
    """
    Load all trace data from subdirectories in the parent directory.
    
    Args:
        parent_directory: Parent directory containing multiple trace subdirectories
        
    Returns:
        Dictionary mapping trace names to TraceData objects
    """
    traces = {}
    
    # Find all subdirectories that contain traced_dataset folder
    for item in os.listdir(parent_directory):
        full_path = os.path.join(parent_directory, item)
        if os.path.isdir(full_path):
            traced_dataset_path = os.path.join(full_path, "traced_dataset")
            if os.path.exists(traced_dataset_path):
                # This is a valid trace directory
                trace_name = item
                try:
                    traces[trace_name] = TraceData(full_path)
                    alias = get_trace_alias(trace_name)
                    print(f"Loaded trace: {trace_name} (alias: {alias})")
                except Exception as e:
                    print(f"Failed to load trace {trace_name}: {e}")
    
    return traces


def visualize_multiple_distributions(traces_dict, save_path=None):
    """
    Create a matrix of distribution plots.
    Rows: Different traces
    Columns: TTFT, TBT, TPOT distributions
    
    Args:
        traces_dict: Dictionary mapping trace names to TraceData objects
        save_path: Path to save the figure (optional)
    """
    num_traces = len(traces_dict)
    trace_names = sorted(traces_dict.keys())
    
    # Create figure with num_traces rows and 3 columns - more compact
    fig, axes = plt.subplots(num_traces, 3, figsize=(15, 3.5 * num_traces))
    
    # If only one trace, make axes 2D for consistent indexing
    if num_traces == 1:
        axes = axes.reshape(1, -1)
    
    # First pass: collect all data to determine common x-axis limits
    all_ttfts = []
    all_tbts = []
    all_tpots = []
    
    for trace_name in trace_names:
        trace_data = traces_dict[trace_name]
        all_ttfts.extend(trace_data.ttfts)
        all_tpots.extend(trace_data.tpots)
        for tbt_list in trace_data.tbts:
            all_tbts.extend(tbt_list)
    
    # Get SLO values (should be same across all traces)
    first_trace_data = traces_dict[trace_names[0]]
    ttft_slo = float(first_trace_data.ttft_slo)
    tbt_slo = float(first_trace_data.tbt_slo)
    tpot_slo = tbt_slo
    
    # Calculate common x-axis limits (using percentiles to avoid outliers, but ensure SLO is visible)
    if len(all_ttfts) > 0:
        ttft_p99 = np.percentile(all_ttfts, 99)
        ttft_xlim = (0, max(ttft_p99, ttft_slo) * 1.1)
    else:
        ttft_xlim = (0, ttft_slo * 1.2)
    
    if len(all_tbts) > 0:
        tbt_p99 = np.percentile(all_tbts, 99)
        tbt_xlim = (0, max(tbt_p99, tbt_slo) * 1.1)
    else:
        tbt_xlim = (0, tbt_slo * 1.2)
    
    if len(all_tpots) > 0:
        tpot_p99 = np.percentile(all_tpots, 99)
        tpot_xlim = (0, max(tpot_p99, tpot_slo) * 1.1)
    else:
        tpot_xlim = (0, tpot_slo * 1.2)
    
    # Create equi-sized bins for each metric (50 bins across the shared range)
    num_bins = 50
    ttft_bins = np.linspace(ttft_xlim[0], ttft_xlim[1], num_bins + 1)
    tbt_bins = np.linspace(tbt_xlim[0], tbt_xlim[1], num_bins + 1)
    tpot_bins = np.linspace(tpot_xlim[0], tpot_xlim[1], num_bins + 1)
    
    # Second pass: create plots with aligned axes
    for row_idx, trace_name in enumerate(trace_names):
        trace_data = traces_dict[trace_name]
        trace_alias = get_trace_alias(trace_name)
        
        # Prepare data
        ttfts = np.array(trace_data.ttfts)
        tpots = np.array(trace_data.tpots)
        
        # Flatten tbts (list of lists) to get all inter-token latencies
        tbts_flat = []
        for tbt_list in trace_data.tbts:
            tbts_flat.extend(tbt_list)
        tbts = np.array(tbts_flat)
        
        # Convert SLO values to float
        ttft_slo = float(trace_data.ttft_slo)
        tbt_slo = float(trace_data.tbt_slo)
        tpot_slo = tbt_slo
        
        # Calculate percentages below SLO
        ttft_below_slo = (ttfts <= ttft_slo).sum() / len(ttfts) * 100 if len(ttfts) > 0 else 0
        tbt_below_slo = (tbts <= tbt_slo).sum() / len(tbts) * 100 if len(tbts) > 0 else 0
        tpot_below_slo = (tpots <= tpot_slo).sum() / len(tpots) * 100 if len(tpots) > 0 else 0
        
        # TTFT histogram
        ax_ttft = axes[row_idx, 0]
        if len(ttfts) > 0:
            ttft_mean = np.mean(ttfts)
            ttft_p90 = np.percentile(ttfts, 90)
            
            ax_ttft.hist(ttfts, bins=ttft_bins, alpha=0.7, color='skyblue', 
                        edgecolor='black', linewidth=0.5)
            ax_ttft.axvline(ttft_slo, color='red', linestyle='--', linewidth=1.5, 
                           label=f'SLO')
            ax_ttft.axvline(ttft_mean, color='blue', linestyle='-.', linewidth=1.5,
                           label=f'Mean')
            ax_ttft.axvline(ttft_p90, color='orange', linestyle=':', linewidth=1.5,
                           label=f'P90')
            
            # Add percentage below SLO text
            ax_ttft.text(0.98, 0.65, f'{ttft_below_slo:.1f}% under SLO',
                        transform=ax_ttft.transAxes,
                        fontsize=9, fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                 edgecolor='black', alpha=0.8))
        ax_ttft.set_xlim(ttft_xlim)
        ax_ttft.set_xlabel('TTFT (s)', fontsize=10)
        ax_ttft.set_ylabel(f'{trace_alias}\n\nFreq', fontsize=10, fontweight='bold')
        if row_idx == 0:
            ax_ttft.set_title('TTFT Distribution', fontsize=11, fontweight='bold')
        ax_ttft.legend(fontsize=8, loc='upper right')
        ax_ttft.grid(True, alpha=0.3, axis='y')
        
        # TBT histogram
        ax_tbt = axes[row_idx, 1]
        if len(tbts) > 0:
            tbt_mean = np.mean(tbts)
            tbt_p90 = np.percentile(tbts, 90)
            
            ax_tbt.hist(tbts, bins=tbt_bins, alpha=0.7, color='lightgreen', 
                       edgecolor='black', linewidth=0.5)
            ax_tbt.axvline(tbt_slo, color='red', linestyle='--', linewidth=1.5,
                          label=f'SLO')
            ax_tbt.axvline(tbt_mean, color='blue', linestyle='-.', linewidth=1.5,
                          label=f'Mean')
            ax_tbt.axvline(tbt_p90, color='orange', linestyle=':', linewidth=1.5,
                          label=f'P90')
            
            # Add percentage below SLO text
            ax_tbt.text(0.98, 0.65, f'{tbt_below_slo:.1f}% under SLO',
                       transform=ax_tbt.transAxes,
                       fontsize=9, fontweight='bold',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                edgecolor='black', alpha=0.8))
        ax_tbt.set_xlim(tbt_xlim)
        ax_tbt.set_xlabel('TBT (s)', fontsize=10)
        ax_tbt.set_ylabel('Freq', fontsize=10)
        if row_idx == 0:
            ax_tbt.set_title('TBT Distribution', fontsize=11, fontweight='bold')
        ax_tbt.legend(fontsize=8, loc='upper right')
        ax_tbt.grid(True, alpha=0.3, axis='y')
        
        # TPOT histogram
        ax_tpot = axes[row_idx, 2]
        if len(tpots) > 0:
            tpot_mean = np.mean(tpots)
            tpot_p90 = np.percentile(tpots, 90)
            
            ax_tpot.hist(tpots, bins=tpot_bins, alpha=0.7, color='salmon', 
                        edgecolor='black', linewidth=0.5)
            ax_tpot.axvline(tpot_slo, color='red', linestyle='--', linewidth=1.5,
                           label=f'SLO')
            ax_tpot.axvline(tpot_mean, color='blue', linestyle='-.', linewidth=1.5,
                           label=f'Mean')
            ax_tpot.axvline(tpot_p90, color='orange', linestyle=':', linewidth=1.5,
                           label=f'P90')
            
            # Add percentage below SLO text
            ax_tpot.text(0.98, 0.65, f'{tpot_below_slo:.1f}% under SLO',
                        transform=ax_tpot.transAxes,
                        fontsize=9, fontweight='bold',
                        ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                 edgecolor='black', alpha=0.8))
        ax_tpot.set_xlim(tpot_xlim)
        ax_tpot.set_xlabel('TPOT (s)', fontsize=10)
        ax_tpot.set_ylabel('Freq', fontsize=10)
        if row_idx == 0:
            ax_tpot.set_title('TPOT Distribution', fontsize=11, fontweight='bold')
        ax_tpot.legend(fontsize=8, loc='upper right')
        ax_tpot.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distributions figure saved to: {save_path}")
    
    plt.show()
    return fig


def visualize_multiple_bars(traces_dict, save_path=None):
    """
    Create bar plots comparing metrics across traces.
    Organized with Mean metrics in first row, P90 metrics in second row, and power separately.
    
    Args:
        traces_dict: Dictionary mapping trace names to TraceData objects
        save_path: Path to save the figure (optional)
    """
    trace_names = sorted(traces_dict.keys())
    trace_aliases = [get_trace_alias(name) for name in trace_names]
    
    # Define colors for each trace
    trace_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {alias: trace_colors[i % len(trace_colors)] 
                 for i, alias in enumerate(trace_aliases)}
    
    # Compute metrics for each trace
    metrics = {
        'mean_ttft': [],
        'mean_tpot': [],
        'mean_tbt': [],
        'p90_ttft': [],
        'p90_tpot': [],
        'p90_tbt': [],
        'avg_power': [],
        'pct_ttft_under_slo': [],
        'pct_tbt_under_slo': [],
        'pct_tpot_under_slo': []
    }
    
    for trace_name in trace_names:
        trace_data = traces_dict[trace_name]
        
        # Prepare data
        ttfts = np.array(trace_data.ttfts)
        tpots = np.array(trace_data.tpots)
        
        # Flatten tbts
        tbts_flat = []
        for tbt_list in trace_data.tbts:
            tbts_flat.extend(tbt_list)
        tbts = np.array(tbts_flat)
        
        # Get SLO values
        ttft_slo = float(trace_data.ttft_slo)
        tbt_slo = float(trace_data.tbt_slo)
        tpot_slo = tbt_slo
        
        # Calculate power
        power_watts = trace_data.energy_consumption / trace_data.total_duration
        
        # Calculate percentages under SLO
        pct_ttft_under_slo = (ttfts <= ttft_slo).sum() / len(ttfts) * 100 if len(ttfts) > 0 else 0
        pct_tbt_under_slo = (tbts <= tbt_slo).sum() / len(tbts) * 100 if len(tbts) > 0 else 0
        pct_tpot_under_slo = (tpots <= tpot_slo).sum() / len(tpots) * 100 if len(tpots) > 0 else 0
        
        # Store metrics
        metrics['mean_ttft'].append(np.mean(ttfts) if len(ttfts) > 0 else 0)
        metrics['mean_tpot'].append(np.mean(tpots) if len(tpots) > 0 else 0)
        metrics['mean_tbt'].append(np.mean(tbts) if len(tbts) > 0 else 0)
        metrics['p90_ttft'].append(np.percentile(ttfts, 90) if len(ttfts) > 0 else 0)
        metrics['p90_tpot'].append(np.percentile(tpots, 90) if len(tpots) > 0 else 0)
        metrics['p90_tbt'].append(np.percentile(tbts, 90) if len(tbts) > 0 else 0)
        metrics['avg_power'].append(power_watts)
        metrics['pct_ttft_under_slo'].append(pct_ttft_under_slo)
        metrics['pct_tbt_under_slo'].append(pct_tbt_under_slo)
        metrics['pct_tpot_under_slo'].append(pct_tpot_under_slo)
    
    # Create figure with 3 rows: Mean metrics, P90 metrics, and percentage under SLO
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    
    x = np.arange(len(trace_aliases))
    bar_width = 0.6
    
    # Row 1: Mean metrics
    mean_configs = [
        ('mean_ttft', 'Mean TTFT (s)'),
        ('mean_tbt', 'Mean TBT (s)'),
        ('mean_tpot', 'Mean TPOT (s)'),
        ('avg_power', 'Avg Power (W)'),
    ]
    
    for col_idx, (metric_key, metric_label) in enumerate(mean_configs):
        ax = axes[0, col_idx]
        colors = [color_map[alias] for alias in trace_aliases]
        bars = ax.bar(x, metrics[metric_key], bar_width, color=colors, 
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(trace_aliases, rotation=0, ha='center', fontsize=10)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: P90 metrics
    p90_configs = [
        ('p90_ttft', 'P90 TTFT (s)'),
        ('p90_tbt', 'P90 TBT (s)'),
        ('p90_tpot', 'P90 TPOT (s)'),
    ]
    
    for col_idx, (metric_key, metric_label) in enumerate(p90_configs):
        ax = axes[1, col_idx]
        colors = [color_map[alias] for alias in trace_aliases]
        bars = ax.bar(x, metrics[metric_key], bar_width, color=colors, 
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(trace_aliases, rotation=0, ha='center', fontsize=10)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend in the empty subplot of row 2
    ax_legend = axes[1, 3]
    ax_legend.axis('off')
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[alias], 
                                     edgecolor='black', linewidth=1.2, alpha=0.8, label=alias)
                      for alias in trace_aliases]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=11, 
                    title='Traces', title_fontsize=12, frameon=True)
    
    # Row 3: Percentage under SLO metrics
    slo_configs = [
        ('pct_ttft_under_slo', '% TTFT under SLO'),
        ('pct_tbt_under_slo', '% TBT under SLO'),
        ('pct_tpot_under_slo', '% TPOT under SLO'),
    ]
    
    for col_idx, (metric_key, metric_label) in enumerate(slo_configs):
        ax = axes[2, col_idx]
        colors = [color_map[alias] for alias in trace_aliases]
        bars = ax.bar(x, metrics[metric_key], bar_width, color=colors, 
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(trace_aliases, rotation=0, ha='center', fontsize=10)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 105])  # Set y-axis to 0-105% for better visualization
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add a reference line at 100%
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='100%')
    
    # Hide the last subplot in row 3
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar comparison figure saved to: {save_path}")
    
    plt.show()
    return fig


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
        description="Visualize trace data from benchmark results. Can handle single trace or multiple traces."
    )
    parser.add_argument(
        "base_directory",
        type=str,
        help="Base directory containing either traced_dataset folder (single trace) or multiple trace subdirectories"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multiple", "auto"],
        default="auto",
        help="Visualization mode: 'single' for one trace, 'multiple' for comparing multiple traces, 'auto' to detect automatically"
    )
 
    args = parser.parse_args()
    
    # Determine mode automatically if set to auto
    mode = args.mode
    if mode == "auto":
        # Check if base_directory directly contains traced_dataset
        traced_dataset_path = os.path.join(args.base_directory, "traced_dataset")
        if os.path.exists(traced_dataset_path):
            mode = "single"
        else:
            # Check if any subdirectories contain traced_dataset
            has_trace_subdirs = False
            for item in os.listdir(args.base_directory):
                full_path = os.path.join(args.base_directory, item)
                if os.path.isdir(full_path):
                    sub_traced_path = os.path.join(full_path, "traced_dataset")
                    if os.path.exists(sub_traced_path):
                        has_trace_subdirs = True
                        break
            mode = "multiple" if has_trace_subdirs else "single"
    
    print(f"Running in {mode} mode")
    
    if mode == "single":
        # Original single trace visualization
        trace_data = TraceData(args.base_directory)
        
        # Create visualization directory
        viz_dir = os.path.join(args.base_directory, "visualization")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate filename based on the trace data
        dir_name = os.path.basename(os.path.normpath(args.base_directory))
        output_filename = f"trace_visualization_{dir_name}.png"
        save_path = os.path.join(viz_dir, output_filename)
        
        # Generate visualization
        visualize_trace(trace_data, save_path=save_path)
        
    else:  # multiple mode
        # Load multiple traces
        traces_dict = load_multiple_traces(args.base_directory)
        
        if not traces_dict:
            print("No valid trace directories found!")
            exit(1)
        
        print(f"\nFound {len(traces_dict)} traces:")
        for name in sorted(traces_dict.keys()):
            print(f"  - {name}")
        
        # Create visualization directory
        viz_dir = os.path.join(args.base_directory, "comparison_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate distributions plot
        distributions_path = os.path.join(viz_dir, "distributions.png")
        print(f"\nGenerating distributions comparison...")
        visualize_multiple_distributions(traces_dict, save_path=distributions_path)
        
        # Generate bar comparison plot
        bars_path = os.path.join(viz_dir, "bar.png")
        print(f"\nGenerating bar comparison...")
        visualize_multiple_bars(traces_dict, save_path=bars_path)
        
        print(f"\nAll visualizations saved to: {viz_dir}")
