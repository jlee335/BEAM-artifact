#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def calculate_metrics_from_json(json_file):
    """Extract and calculate metrics from JSON benchmark results."""
    with open(json_file) as f:
        data = json.load(f)

    # Get mean TTFT directly
    mean_ttft_ms = data.get('mean_ttft_ms', 0)
    p99_ttft_ms = data.get('p99_ttft_ms', 0)

    # Calculate TBT values by flattening all inter-token latencies
    itls = data.get('itls', [])
    all_tbts_ms = [tbt * 1000 for request_itls in itls for tbt in request_itls]
    
    mean_tbt_ms = np.mean(all_tbts_ms) if all_tbts_ms else 0
    p90_tbt_ms = np.percentile(all_tbts_ms, 90) if all_tbts_ms else 0
    p99_tbt_ms = np.percentile(all_tbts_ms, 99) if all_tbts_ms else 0

    # Calculate P90 TTFT from raw data
    ttfts = data.get('ttfts', [])
    ttfts_ms = [t * 1000 for t in ttfts]
    p90_ttft_ms = np.percentile(ttfts_ms, 90) if ttfts_ms else 0

    return {
        'mean_ttft_ms': mean_ttft_ms,
        'mean_tbt_ms': mean_tbt_ms,
        'p90_ttft_ms': p90_ttft_ms,
        'p90_tbt_ms': p90_tbt_ms,
        'p99_ttft_ms': p99_ttft_ms,
        'p99_tbt_ms': p99_tbt_ms
    }


def calculate_power_and_energy(energy_file, batch_log_file, json_file=None, cut_to_last_entry=False):
    """Calculate average power and total energy consumption during the test period.
    
    Args:
        energy_file: Path to energy CSV file
        batch_log_file: Path to batch log CSV file
        json_file: Path to JSON benchmark results file
        cut_to_last_entry: If True, cut energy measurement to last TTFT timestamp
    
    Returns:
        tuple: (average_power_w, total_energy_kj)
    """
    try:
        energy_df = pd.read_csv(energy_file)
        energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])

        batch_df = pd.read_csv(batch_log_file)
        if batch_df.empty:
            return 0, 0

        start_time = pd.to_datetime(batch_df['start_time'].iloc[0])
        end_time = pd.to_datetime(batch_df['current_time'].iloc[-1])

        # If cut_to_last_entry is enabled, find the last TTFT timestamp from JSON
        if cut_to_last_entry and json_file and os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    json_data = json.load(f)
                
                entry_times = json_data.get('entry_times', [])
                ttfts = json_data.get('ttfts', [])
                
                if entry_times and ttfts and len(entry_times) == len(ttfts):
                    # Calculate when each TTFT was actually measured
                    ttft_timestamps = [entry + ttft for entry, ttft in zip(entry_times, ttfts)]
                    last_ttft_offset = max(ttft_timestamps)
                    
                    # Add this offset to the start_time to get absolute timestamp
                    last_ttft_time = start_time + pd.Timedelta(seconds=last_ttft_offset)
                    
                    if last_ttft_time < end_time:
                        print(f"  Cut to last TTFT: {last_ttft_time} (original end: {end_time}, saved {(end_time - last_ttft_time).total_seconds():.2f}s)")
                        end_time = last_ttft_time
                    else:
                        print(f"  Last TTFT time ({last_ttft_time}) is after end_time ({end_time}), keeping original end_time")
            except Exception as e:
                print(f"  Warning: Could not extract last TTFT from JSON: {e}")

        mask = (energy_df['timestamp'] >= start_time) & (energy_df['timestamp'] <= end_time)
        filtered_energy = energy_df[mask].copy()

        if len(filtered_energy) < 2:
            return 0, 0

        # Calculate total energy consumption for all GPUs (convert mJ to J)
        gpu_columns = [col for col in energy_df.columns if 'gpu_' in col and 'joules' in col]
        start_energy = filtered_energy.iloc[0][gpu_columns].sum() / 1e3
        end_energy = filtered_energy.iloc[-1][gpu_columns].sum() / 1e3

        # Total energy in kJ
        total_energy_kj = (end_energy - start_energy) / 1e3

        duration_seconds = (end_time - start_time).total_seconds()
        print(f"Duration: {duration_seconds} seconds")
        if duration_seconds > 0:
            average_power_w = (end_energy - start_energy) / duration_seconds
        else:
            average_power_w = 0

        return average_power_w, total_energy_kj

    except Exception as e:
        print(f"Error calculating power/energy for {energy_file}: {e}")
        return 0, 0


def collect_data_from_pareto_directory(pareto_dir, ttft_max_threshold=2000, filter_slo_violations=True, cut_to_last_entry=False):
    """Collect all data points from a pareto test directory."""
    data_points = []
    filtered_count = 0

    # Find all configuration directories (e.g., s1_s2_tp2_pp2_chunk_2048)
    for config_dir in sorted(glob.glob(os.path.join(pareto_dir, "s1_s2_*"))):
        if not os.path.isdir(config_dir):
            continue

        config_name = os.path.basename(config_dir)
        print(f"Processing configuration: {config_name}")

        # Process each SLO directory
        for slo_dir in sorted(glob.glob(os.path.join(config_dir, "SLO_*"))):
            if not os.path.isdir(slo_dir):
                continue

            slo_name = os.path.basename(slo_dir)
            traced_dataset_dir = os.path.join(slo_dir, "traced_dataset")

            if not os.path.exists(traced_dataset_dir):
                print(f"  Missing traced_dataset in {slo_dir}")
                continue

            # Find required files
            json_files = glob.glob(os.path.join(traced_dataset_dir, "*.json"))
            # filter out json_files with _estimations in the name
            json_files = [file for file in json_files if '_estimations' not in file]
            energy_file = os.path.join(traced_dataset_dir, "gpu_energy_and_frequency_huggyllama_llama-13b.csv")
            batch_log_file = os.path.join(traced_dataset_dir, "batch_log_GPU_0.csv")

            if not json_files or not os.path.exists(energy_file) or not os.path.exists(batch_log_file):
                print(f"  Missing files in {slo_dir}")
                continue

            json_file = json_files[0]

            # Extract metrics
            metrics = calculate_metrics_from_json(json_file)
            power, energy = calculate_power_and_energy(energy_file, batch_log_file, json_file, cut_to_last_entry)

            # Extract SLO targets from directory name (e.g., SLO_A_0.2_5.0 -> TBT=0.2, TTFT=5.0)
            slo_parts = slo_name.split('_')
            slo_tbt_target = float(slo_parts[2]) if len(slo_parts) >= 3 else 0
            slo_ttft_target = float(slo_parts[3]) if len(slo_parts) >= 4 else 0
            
            # Filter out invalid Mean TTFT data points
            mean_ttft = metrics['mean_ttft_ms']
            mean_tbt = metrics['mean_tbt_ms']
            p90_tbt = metrics['p90_tbt_ms']
            
            if mean_ttft <= 0:
                print(f"  Filtered out {slo_name}: Invalid TTFT={mean_ttft:.1f}ms")
                filtered_count += 1
                continue
            
            if mean_ttft > ttft_max_threshold:
                print(f"  Filtered out {slo_name}: Extreme TTFT={mean_ttft:.1f}ms (> {ttft_max_threshold}ms)")
                filtered_count += 1
                continue

            # Filter out data points that fail to meet SLO
            if filter_slo_violations and p90_tbt > slo_tbt_target * 1000:
                print(f"  Filtered out {slo_name}: TBT={p90_tbt:.1f}ms exceeds SLO target={slo_tbt_target*1000:.1f}ms")
                filtered_count += 1
                continue

            data_point = {
                'config': config_name,
                'slo_name': slo_name,
                'slo_tbt_target': slo_tbt_target,
                'slo_ttft_target': slo_ttft_target,
                'power_w': power,
                'energy_kj': energy,
                **metrics
            }

            data_points.append(data_point)
            print(f"    {slo_name}: TTFT={mean_ttft:.1f}ms, TBT={mean_tbt:.1f}ms, Power={power:.1f}W, Energy={energy:.1f}kJ")

    if filtered_count > 0:
        print(f"\nFiltered out {filtered_count} data points")
    
    return data_points


def get_gradient_color(base_color, slo_value, slo_min, slo_max):
    """Generate gradient color based on SLO value."""
    if slo_max == slo_min:
        intensity = 0.7
    else:
        intensity = 0.3 + 0.7 * (slo_value - slo_min) / (slo_max - slo_min)
    
    rgb = mcolors.hex2color(base_color)
    return tuple(c * intensity for c in rgb)


def plot_metric_vs_power(ax, df, metric_key, metric_label, title, configs, config_colors, config_markers, slo_min, slo_max, use_power=True, legend_handles=None):
    """Helper function to plot a metric vs power/energy."""
    y_metric = 'power_w' if use_power else 'energy_kj'
    y_label = 'Average Power (W)' if use_power else 'Total Energy (kJ)'
    
    for config in configs:
        config_data = df[df['config'] == config].sort_values('slo_tbt_target')
        
        colors = [get_gradient_color(config_colors[config], slo, slo_min, slo_max) 
                 for slo in config_data['slo_tbt_target']]
        
        ax.scatter(config_data[metric_key], config_data[y_metric],
                   c=colors, marker=config_markers[config], s=150,
                   edgecolors='black', linewidths=1.5, alpha=0.9)
        
        ax.plot(config_data[metric_key], config_data[y_metric],
                color=config_colors[config], linestyle='--', alpha=0.6, linewidth=2)
        
        # Add SLO labels (show TBT/TTFT)
        for _, row in config_data.iterrows():
            ax.annotate(f'{row["slo_tbt_target"]:.2f}/{row["slo_ttft_target"]:.1f}',
                        (row[metric_key], row[y_metric]),
                        textcoords="offset points", xytext=(0, 12), ha='center',
                        fontsize=10, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.8, edgecolor='none'))
        
        # Create legend handle only once
        if legend_handles is not None and config not in [h.get_label() for h in legend_handles]:
            legend_handles.append(plt.Line2D([0], [0], marker=config_markers[config], 
                                           color='w', markerfacecolor=config_colors[config],
                                           markersize=10, markeredgecolor='black',
                                           markeredgewidth=1.5, label=config))
    
    ax.set_xlabel(metric_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.2, linewidth=0.8)


def create_pareto_plots(data_points, output_dir, use_power=True):
    """Create pareto plots separated by configuration type (tp1_pp4 vs tp2_pp2)."""
    if not data_points:
        print("No data points found!")
        return

    df = pd.DataFrame(data_points)

    # Set up plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.edgecolor': 'black',
        'legend.borderpad': 0.3
    })

    # Separate configurations by type
    df_tp1_pp4 = df[df['config'].str.contains('tp2_pp4')]
    df_tp2_pp2 = df[df['config'].str.contains('tp4_pp2')]

    # Create figure with 2x2 subplots (top row for tp1_pp4, bottom row for tp2_pp2)
    metric_type = 'Power Consumption' if use_power else 'Energy Consumption'
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Pareto Analysis: Performance vs {metric_type}',
                 fontsize=18, fontweight='bold', y=0.98)

    # Configure colors and markers
    all_configs = df['config'].unique()
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    config_colors = dict(zip(all_configs, base_colors[:len(all_configs)]))
    config_markers = dict(zip(all_configs, markers[:len(all_configs)]))
    
    slo_min = df['slo_tbt_target'].min()
    slo_max = df['slo_tbt_target'].max()
    
    legend_handles = []
    
    # Plot all metrics
    y_label_short = 'Power' if use_power else 'Energy'
    
    # Top row: tp1_pp4 configurations (1x4)
    if not df_tp1_pp4.empty:
        configs_tp1_pp4 = df_tp1_pp4['config'].unique()
        plot_metric_vs_power(axes[0, 0], df_tp1_pp4, 'p90_ttft_ms', 'P90 TTFT (ms)',
                            f'TP1_PP4: P90 TTFT vs {y_label_short}', configs_tp1_pp4, config_colors,
                            config_markers, slo_min, slo_max, use_power, legend_handles)
        
        plot_metric_vs_power(axes[0, 1], df_tp1_pp4, 'p90_tbt_ms', 'P90 TBT (ms)',
                            f'TP1_PP4: P90 TBT vs {y_label_short}', configs_tp1_pp4, config_colors,
                            config_markers, slo_min, slo_max, use_power, legend_handles)
    
    # Bottom row: tp2_pp2 configurations (2x2)
    if not df_tp2_pp2.empty:
        configs_tp2_pp2 = df_tp2_pp2['config'].unique()
        plot_metric_vs_power(axes[1, 0], df_tp2_pp2, 'p90_ttft_ms', 'P90 TTFT (ms)',
                            f'TP2_PP2: P90 TTFT vs {y_label_short}', configs_tp2_pp2, config_colors,
                            config_markers, slo_min, slo_max, use_power, legend_handles)
        
        plot_metric_vs_power(axes[1, 1], df_tp2_pp2, 'p90_tbt_ms', 'P90 TBT (ms)',
                            f'TP2_PP2: P90 TBT vs {y_label_short}', configs_tp2_pp2, config_colors,
                            config_markers, slo_min, slo_max, use_power, legend_handles)

    # Synchronize axis scales: same x scale per metric, same y scale across all plots
    y_col = 'power_w' if use_power else 'energy_kj'
    pad = 0.05  # 5% padding

    def padded(vmin, vmax):
        margin = (vmax - vmin) * pad if vmax != vmin else abs(vmax) * pad or 1
        return vmin - margin, vmax + margin

    ttft_xmin, ttft_xmax = padded(df['p90_ttft_ms'].min(), df['p90_ttft_ms'].max())
    tbt_xmin, tbt_xmax = padded(df['p90_tbt_ms'].min(), df['p90_tbt_ms'].max())
    y_min, y_max = padded(df[y_col].min(), df[y_col].max())

    for row in range(2):
        axes[row, 0].set_xlim(ttft_xmin, ttft_xmax)
        axes[row, 1].set_xlim(tbt_xmin, tbt_xmax)
        for col in range(2):
            axes[row, col].set_ylim(y_min, y_max)

    # Add shared legend
    labels = [handle.get_label() for handle in legend_handles]
    fig.legend(legend_handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=min(len(all_configs), 4), fontsize=12, frameon=True, 
               fancybox=False, shadow=False, edgecolor='black')
    
    # Add SLO gradient explanation
    fig.text(0.02, 0.02, f'Color intensity = TBT SLO (lighter = {slo_min:.2f}s, darker = {slo_max:.2f}s). Labels show TBT/TTFT SLO targets.', 
             fontsize=10, style='italic', ha='left', va='bottom')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    # Save plot
    output_file = os.path.join(output_dir, 'pareto_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Pareto plots saved to: {output_file}")

    # Save data to CSV
    csv_file = os.path.join(output_dir, 'pareto_data.csv')
    df.to_csv(csv_file, index=False)
    print(f"Data saved to: {csv_file}")

    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Configurations: {', '.join(all_configs)}")
    print(f"Power range: {df['power_w'].min():.1f}W - {df['power_w'].max():.1f}W")
    print(f"Energy range: {df['energy_kj'].min():.1f}kJ - {df['energy_kj'].max():.1f}kJ")
    print(f"Mean TTFT range: {df['mean_ttft_ms'].min():.1f}ms - {df['mean_ttft_ms'].max():.1f}ms")
    print(f"Mean TBT range: {df['mean_tbt_ms'].min():.1f}ms - {df['mean_tbt_ms'].max():.1f}ms")
    print(f"P99 TTFT range: {df['p99_ttft_ms'].min():.1f}ms - {df['p99_ttft_ms'].max():.1f}ms")
    print(f"P99 TBT range: {df['p99_tbt_ms'].min():.1f}ms - {df['p99_tbt_ms'].max():.1f}ms")


def main():
    parser = argparse.ArgumentParser(description='Visualize Pareto analysis from test results')
    parser.add_argument('pareto_dir', help='Path to pareto test directory')
    parser.add_argument('--output', '-o', default='.', help='Output directory for plots')
    parser.add_argument('--ttft-max-threshold', type=float, default=2000.0,
                        help='Maximum Mean TTFT threshold in ms (default: 2000)')
    parser.add_argument('--include-slo-violations', action='store_true',
                        help='Include data points that violate their SLO targets')
    parser.add_argument('--power', action='store_true',
                        help='Use average power (W) instead of total energy (kJ)')
    parser.add_argument('--cut-to-last-entry', action='store_true',
                        help='Cut energy measurement to last TTFT timestamp instead of end of test')

    args = parser.parse_args()

    if not os.path.exists(args.pareto_dir):
        print(f"Error: Directory {args.pareto_dir} does not exist!")
        return 1

    os.makedirs(args.output, exist_ok=True)

    print(f"Collecting data from: {args.pareto_dir}")
    print(f"Using Mean TTFT threshold: ≤ {args.ttft_max_threshold}ms")
    
    metric_type = "average power (W)" if args.power else "total energy (kJ)"
    print(f"Metric: {metric_type}")
    
    if args.cut_to_last_entry:
        print("Cutting energy measurement to last TTFT timestamp")
    
    filter_slo_violations = not args.include_slo_violations
    if filter_slo_violations:
        print("Filtering out SLO violations (TBT > SLO target)")
    else:
        print("Including all data points (even SLO violations)")
    
    data_points = collect_data_from_pareto_directory(
        args.pareto_dir, args.ttft_max_threshold, filter_slo_violations, args.cut_to_last_entry)

    if not data_points:
        print("No valid data points found!")
        return 1

    print(f"Found {len(data_points)} valid data points")
    create_pareto_plots(data_points, args.output, use_power=args.power)

    return 0


if __name__ == "__main__":
    exit(main())