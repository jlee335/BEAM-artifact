# SPDX-License-Identifier: Apache-2.0
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def create_visualization(tps_value, csv_file):
    """Create visualization for a specific TPS value."""

    # Read the data
    df = pd.read_csv(csv_file)

    # Filter data for the specified TPS
    df_tps = df[df['tps'] == tps_value].copy()

    if df_tps.empty:
        print(f"No data found for TPS = {tps_value}")
        return

    # Get unique values - handle different column names
    if 'in' in df_tps.columns and 'out' in df_tps.columns:
        in_values = sorted(df_tps['in'].unique())
        out_values = sorted(df_tps['out'].unique())
    else:
        # If no 'in'/'out' columns, create dummy values for visualization
        in_values = [1]  # Default single value
        out_values = [1]  # Default single value
        df_tps['in'] = 1
        df_tps['out'] = 1
    
    clock_values = sorted(df_tps['clock'].unique())

    # Create tp×pp labels
    df_tps['tp_pp_config'] = df_tps['tp'].astype(
        str) + 'x' + df_tps['pp'].astype(str)
    tp_pp_configs = sorted(df_tps['tp_pp_config'].unique())

    print(f"Creating visualization for TPS = {tps_value}")
    print(f"IN values: {in_values}")
    print(f"OUT values: {out_values}")
    print(f"TP×PP configs: {tp_pp_configs}")
    
    # Diagnostic information about available columns
    print(f"\nAvailable columns: {list(df_tps.columns)}")
    print(f"TTFT mean data available: {'ttft_mean' in df_tps.columns}")
    print(f"TTFT P99 data available: {'ttft_p99' in df_tps.columns}")
    print(f"TBT mean data available: {'tbt_mean' in df_tps.columns}")
    print(f"TBT P90 data available: {'tbt_p90' in df_tps.columns}")
    
    # Check if TBT data has actual values
    if 'tbt_mean' in df_tps.columns:
        tbt_mean_values = df_tps['tbt_mean'].dropna()
        print(f"TBT mean non-null values: {len(tbt_mean_values)} out of {len(df_tps)}")
        if len(tbt_mean_values) > 0:
            print(f"TBT mean sample values: {tbt_mean_values.head().tolist()}")
        else:
            print("WARNING: TBT mean column exists but contains no valid data!")
    
    if 'tbt_p90' in df_tps.columns:
        tbt_p90_values = df_tps['tbt_p90'].dropna()
        print(f"TBT P90 non-null values: {len(tbt_p90_values)} out of {len(df_tps)}")
        if len(tbt_p90_values) > 0:
            print(f"TBT P90 sample values: {tbt_p90_values.head().tolist()}")
        else:
            print("WARNING: TBT P90 column exists but contains no valid data!")

    # Set up the figure with subplots
    # Create 3 separate figures for the 3 metrics
    # Metrics info - handle different column names and show both mean and P90/P99
    available_metrics = []
    
    # Check for TTFT metrics
    if 'ttft_mean' in df_tps.columns:
        available_metrics.append(('ttft_mean', 'TTFT Mean (ms)', 'Time to First Token (Mean)'))
    if 'ttft_p99' in df_tps.columns:
        available_metrics.append(('ttft_p99', 'TTFT P99 (ms)', 'Time to First Token (P99)'))
    
    # Check for TBT metrics
    if 'tbt_mean' in df_tps.columns:
        available_metrics.append(('tbt_mean', 'TBT Mean (ms)', 'Time Between Tokens (Mean)'))
    if 'tbt_p90' in df_tps.columns:
        available_metrics.append(('tbt_p90', 'TBT P90 (ms)', 'Time Between Tokens (P90)'))
    
    # Always include GPU power
    if 'gpu_power' in df_tps.columns:
        available_metrics.append(('gpu_power', 'Power (W)', 'GPU Power Consumption'))
    
    # Fallback to old column names if new ones not available
    if not available_metrics:
        if 'ttft' in df_tps.columns and 'tbt' in df_tps.columns:
            available_metrics = [('ttft', 'TTFT (ms)', 'Time to First Token'),
                                 ('tbt', 'TBT (ms)', 'Time Between Tokens'),
                                 ('gpu_power', 'Power (W)', 'GPU Power Consumption')]
        else:
            available_metrics = [('gpu_power', 'Power (W)', 'GPU Power Consumption')]
    
    metrics = available_metrics

    # Colors for tp×pp configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'v']

    for metric_name, ylabel, title_suffix in metrics:
        fig, axes = plt.subplots(len(in_values),
                                 len(out_values),
                                 figsize=(5 * len(out_values),
                                          4 * len(in_values)))
        fig.suptitle(f'{title_suffix} vs Clock Frequency (TPS = {tps_value})',
                     fontsize=16,
                     fontweight='bold')

        # Handle case where axes might be 1D
        if len(in_values) == 1:
            axes = axes.reshape(1, -1)
        if len(out_values) == 1:
            axes = axes.reshape(-1, 1)

        # Calculate y-axis limits for this metric to align scales across subplots
        valid_values = df_tps[metric_name].dropna()
        if not valid_values.empty:
            y_min = valid_values.min()
            y_max = valid_values.max()
            # Add 5% padding
            y_range = y_max - y_min
            y_min = y_min - 0.05 * y_range
            y_max = y_max + 0.05 * y_range
            metric_ylim = (y_min, y_max)
        else:
            metric_ylim = (0, 1)  # Default range if no data

        for i, in_val in enumerate(in_values):
            for j, out_val in enumerate(out_values):
                ax = axes[i, j]

                # Filter data for this IN×OUT combination
                subset = df_tps[(df_tps['in'] == in_val)
                                & (df_tps['out'] == out_val)]

                if subset.empty:
                    ax.text(0.5,
                            0.5,
                            'No Data',
                            transform=ax.transAxes,
                            ha='center',
                            va='center',
                            fontsize=12)
                    ax.set_title(f'IN={in_val}, OUT={out_val}')
                    continue

                # Plot each tp×pp configuration
                for k, config in enumerate(tp_pp_configs):
                    config_data = subset[subset['tp_pp_config'] ==
                                         config].copy()

                    if not config_data.empty:
                        # Sort by clock frequency for proper line plotting
                        config_data = config_data.sort_values('clock')

                        ax.plot(
                            config_data['clock'],
                            config_data[metric_name],
                            marker=markers[k % len(markers)],
                            color=colors[k % len(colors)],
                            linewidth=2,
                            markersize=8,
                            label=
                            f'TP{config.split("x")[0]} PP{config.split("x")[1]}',
                            alpha=0.8)

                # Formatting
                ax.set_xlabel('Clock Frequency (MHz)')
                ax.set_ylabel(ylabel)
                ax.set_title(f'IN={in_val}, OUT={out_val}')
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Set x-axis ticks to show all clock values
                ax.set_xticks(clock_values)

                # Apply consistent y-axis limits across all subplots for this metric
                ax.set_ylim(metric_ylim)

        plt.tight_layout()
        plt.savefig(f'visualization_tps{tps_value}_{metric_name}.png',
                    dpi=300,
                    bbox_inches='tight')
        plt.show()


def create_combined_visualization(tps_value, csv_file):
    """Create a combined visualization with all 3 metrics in one figure."""

    # Read the data
    df = pd.read_csv(csv_file)

    # Filter data for the specified TPS
    df_tps = df[df['tps'] == tps_value].copy()

    if df_tps.empty:
        print(f"No data found for TPS = {tps_value}")
        return

    # Get unique values - handle different column names
    if 'in' in df_tps.columns and 'out' in df_tps.columns:
        in_values = sorted(df_tps['in'].unique())
        out_values = sorted(df_tps['out'].unique())
    else:
        # If no 'in'/'out' columns, create dummy values for visualization
        in_values = [1]  # Default single value
        out_values = [1]  # Default single value
        df_tps['in'] = 1
        df_tps['out'] = 1
    
    clock_values = sorted(df_tps['clock'].unique())

    # Create tp×pp labels
    df_tps['tp_pp_config'] = df_tps['tp'].astype(
        str) + 'x' + df_tps['pp'].astype(str)
    tp_pp_configs = sorted(df_tps['tp_pp_config'].unique())

    print(f"Creating combined visualization for TPS = {tps_value}")
    
    # Diagnostic information about available columns
    print(f"\nAvailable columns: {list(df_tps.columns)}")
    print(f"TTFT mean data available: {'ttft_mean' in df_tps.columns}")
    print(f"TTFT P99 data available: {'ttft_p99' in df_tps.columns}")
    print(f"TBT mean data available: {'tbt_mean' in df_tps.columns}")
    print(f"TBT P90 data available: {'tbt_p90' in df_tps.columns}")
    
    # Check if TBT data has actual values
    if 'tbt_mean' in df_tps.columns:
        tbt_mean_values = df_tps['tbt_mean'].dropna()
        print(f"TBT mean non-null values: {len(tbt_mean_values)} out of {len(df_tps)}")
        if len(tbt_mean_values) > 0:
            print(f"TBT mean sample values: {tbt_mean_values.head().tolist()}")
        else:
            print("WARNING: TBT mean column exists but contains no valid data!")
    
    if 'tbt_p90' in df_tps.columns:
        tbt_p90_values = df_tps['tbt_p90'].dropna()
        print(f"TBT P90 non-null values: {len(tbt_p90_values)} out of {len(df_tps)}")
        if len(tbt_p90_values) > 0:
            print(f"TBT P90 sample values: {tbt_p90_values.head().tolist()}")
        else:
            print("WARNING: TBT P90 column exists but contains no valid data!")

    # Metrics info - handle different column names and show both mean and P90/P99
    available_metrics = []
    
    # Check for TTFT metrics
    if 'ttft_mean' in df_tps.columns:
        available_metrics.append(('ttft_mean', 'TTFT Mean (ms)', 'Time to First Token (Mean)'))
    if 'ttft_p99' in df_tps.columns:
        available_metrics.append(('ttft_p99', 'TTFT P99 (ms)', 'Time to First Token (P99)'))
    
    # Check for TBT metrics
    if 'tbt_mean' in df_tps.columns:
        available_metrics.append(('tbt_mean', 'TBT Mean (ms)', 'Time Between Tokens (Mean)'))
    if 'tbt_p90' in df_tps.columns:
        available_metrics.append(('tbt_p90', 'TBT P90 (ms)', 'Time Between Tokens (P90)'))
    
    # Always include GPU power
    if 'gpu_power' in df_tps.columns:
        available_metrics.append(('gpu_power', 'Power (W)', 'GPU Power Consumption'))
    
    # Fallback to old column names if new ones not available
    if not available_metrics:
        if 'ttft' in df_tps.columns and 'tbt' in df_tps.columns:
            available_metrics = [('ttft', 'TTFT (ms)', 'Time to First Token'),
                                 ('tbt', 'TBT (ms)', 'Time Between Tokens'),
                                 ('gpu_power', 'Power (W)', 'GPU Power Consumption')]
        else:
            available_metrics = [('gpu_power', 'Power (W)', 'GPU Power Consumption')]
    
    metrics = available_metrics

    # Set up the figure: dynamic rows based on available metrics × (in_values × out_values) columns
    n_cols = len(in_values) * len(out_values)
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f'Performance vs Clock Frequency (TPS = {tps_value})',
                 fontsize=16,
                 fontweight='bold')
    
    # Handle case where axes might be 1D
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)

    # Colors for tp×pp configurations
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']

    # Create column labels for IN×OUT combinations
    col_labels = []
    for in_val in in_values:
        for out_val in out_values:
            col_labels.append(f'IN={in_val}, OUT={out_val}')

    # Calculate y-axis limits for each metric to align scales across rows
    metric_limits = {}
    for metric_name, _, _ in metrics:
        # Get all valid values for this metric across all IN×OUT combinations
        valid_values = df_tps[metric_name].dropna()
        if not valid_values.empty:
            y_min = valid_values.min()
            y_max = valid_values.max()
            # Add 5% padding
            y_range = y_max - y_min
            y_min = y_min - 0.05 * y_range
            y_max = y_max + 0.05 * y_range
            metric_limits[metric_name] = (y_min, y_max)
        else:
            metric_limits[metric_name] = (0, 1)  # Default range if no data

    for metric_idx, (metric_name, ylabel, title_suffix) in enumerate(metrics):
        for col_idx, (in_val, out_val) in enumerate([(i, o) for i in in_values
                                                     for o in out_values]):
            ax = axes[metric_idx, col_idx]

            # Filter data for this IN×OUT combination
            subset = df_tps[(df_tps['in'] == in_val)
                            & (df_tps['out'] == out_val)]

            if subset.empty:
                ax.text(0.5,
                        0.5,
                        'No Data',
                        transform=ax.transAxes,
                        ha='center',
                        va='center',
                        fontsize=10)
            else:
                # Plot each tp×pp configuration
                for k, config in enumerate(tp_pp_configs):
                    config_data = subset[subset['tp_pp_config'] ==
                                         config].copy()

                    if not config_data.empty:
                        # Sort by clock frequency for proper line plotting
                        config_data = config_data.sort_values('clock')

                        ax.plot(
                            config_data['clock'],
                            config_data[metric_name],
                            marker=markers[k % len(markers)],
                            color=colors[k % len(colors)],
                            linewidth=2,
                            markersize=6,
                            label=
                            f'TP{config.split("x")[0]} PP{config.split("x")[1]}',
                            alpha=0.8)

            # Formatting
            if metric_idx == 2:  # Bottom row
                ax.set_xlabel('Clock Frequency (MHz)')
            if col_idx == 0:  # Left column
                ax.set_ylabel(ylabel)

            # Set title only for top row
            if metric_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=10)

            ax.grid(True, alpha=0.3)

            # Add legend only to first subplot of each metric
            if col_idx == 0:
                ax.legend(fontsize=9)

            # Set x-axis ticks to show all clock values
            ax.set_xticks(clock_values)

            # Apply consistent y-axis limits for this metric row
            ax.set_ylim(metric_limits[metric_name])

    plt.tight_layout()
    plt.savefig(f'visualization_tps{tps_value}_combined.png',
                dpi=300,
                bbox_inches='tight')
    plt.show()


def create_all_tps_combined_visualization(csv_file):
    """Create a single visualization with all TPS values combined, distinguished by color gradient."""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Get unique values
    if 'in' in df.columns and 'out' in df.columns:
        in_values = sorted(df['in'].unique())
        out_values = sorted(df['out'].unique())
    else:
        in_values = [1]
        out_values = [1]
        df['in'] = 1
        df['out'] = 1
    
    tps_values = sorted(df['tps'].unique())
    clock_values = sorted(df['clock'].unique())
    
    # Create tp×pp labels
    df['tp_pp_config'] = df['tp'].astype(str) + 'x' + df['pp'].astype(str)
    tp_pp_configs = sorted(df['tp_pp_config'].unique())
    
    print(f"Creating combined visualization for all TPS values: {tps_values}")
    print(f"IN values: {in_values}")
    print(f"OUT values: {out_values}")
    print(f"TP×PP configs: {tp_pp_configs}")
    
    # Determine available metrics
    available_metrics = []
    
    if 'ttft_mean' in df.columns:
        available_metrics.append(('ttft_mean', 'TTFT Mean (ms)', 'Time to First Token (Mean)'))
    if 'ttft_p99' in df.columns:
        available_metrics.append(('ttft_p99', 'TTFT P99 (ms)', 'Time to First Token (P99)'))
    if 'tbt_mean' in df.columns:
        available_metrics.append(('tbt_mean', 'TBT Mean (ms)', 'Time Between Tokens (Mean)'))
    if 'tbt_p90' in df.columns:
        available_metrics.append(('tbt_p90', 'TBT P90 (ms)', 'Time Between Tokens (P90)'))
    if 'gpu_power' in df.columns:
        available_metrics.append(('gpu_power', 'Power (W)', 'GPU Power Consumption'))
    
    if not available_metrics:
        if 'ttft' in df.columns and 'tbt' in df.columns:
            available_metrics = [('ttft', 'TTFT (ms)', 'Time to First Token'),
                                 ('tbt', 'TBT (ms)', 'Time Between Tokens'),
                                 ('gpu_power', 'Power (W)', 'GPU Power Consumption')]
        else:
            available_metrics = [('gpu_power', 'Power (W)', 'GPU Power Consumption')]
    
    metrics = available_metrics
    
    # Set up the figure: metrics × (in_values × out_values) grid
    n_cols = len(in_values) * len(out_values)
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle('Performance vs Clock Frequency (All TPS Combined)',
                 fontsize=16,
                 fontweight='bold')
    
    # Handle case where axes might be 1D
    if n_cols == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    
    # Base colors for TP×PP configurations
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create color gradients for each TP×PP config across different TPS values
    # Each TP×PP config gets a base color, and TPS values get lighter/darker shades
    config_tps_colors = {}
    for config_idx, config in enumerate(tp_pp_configs):
        base_color = base_colors[config_idx % len(base_colors)]
        # Convert hex to RGB
        base_rgb = np.array([int(base_color[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
        
        # Create gradient from darker to lighter for this config
        for tps_idx, tps in enumerate(tps_values):
            # Interpolate between 0.5*base_color (darker) and base_color to 1.3*base_color (lighter, capped at 1)
            factor = 0.5 + 0.8 * (tps_idx / max(1, len(tps_values) - 1))
            gradient_rgb = np.minimum(base_rgb * factor, 1.0)
            config_tps_colors[(config, tps)] = tuple(gradient_rgb)
    
    # Markers for tp×pp configurations
    markers = ['o', 's', '^', 'v', 'D', 'P']
    
    # Create column labels for IN×OUT combinations
    col_labels = []
    for in_val in in_values:
        for out_val in out_values:
            col_labels.append(f'IN={in_val}, OUT={out_val}')
    
    # Calculate y-axis limits for each metric
    metric_limits = {}
    for metric_name, _, _ in metrics:
        valid_values = df[metric_name].dropna()
        if not valid_values.empty:
            y_min = valid_values.min()
            y_max = valid_values.max()
            y_range = y_max - y_min
            y_min = y_min - 0.05 * y_range
            y_max = y_max + 0.05 * y_range
            metric_limits[metric_name] = (y_min, y_max)
        else:
            metric_limits[metric_name] = (0, 1)
    
    for metric_idx, (metric_name, ylabel, title_suffix) in enumerate(metrics):
        for col_idx, (in_val, out_val) in enumerate([(i, o) for i in in_values
                                                     for o in out_values]):
            ax = axes[metric_idx, col_idx]
            
            # Filter data for this IN×OUT combination
            subset = df[(df['in'] == in_val) & (df['out'] == out_val)]
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data',
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=10)
            else:
                # Plot each TP×PP configuration
                for config_idx, config in enumerate(tp_pp_configs):
                    # Plot each TPS value for this config
                    for tps_idx, tps in enumerate(tps_values):
                        config_data = subset[(subset['tp_pp_config'] == config) & 
                                            (subset['tps'] == tps)].copy()
                        
                        if not config_data.empty:
                            # Sort by clock frequency
                            config_data = config_data.sort_values('clock')
                            
                            # Create label that includes both TP×PP config and TPS
                            label = f'TP{config.split("x")[0]}×PP{config.split("x")[1]} TPS{tps}'
                            
                            ax.plot(
                                config_data['clock'],
                                config_data[metric_name],
                                marker=markers[config_idx % len(markers)],
                                color=config_tps_colors[(config, tps)],
                                linewidth=2,
                                markersize=6,
                                label=label,
                                alpha=0.8)
            
            # Formatting
            if metric_idx == n_rows - 1:  # Bottom row
                ax.set_xlabel('Clock Frequency (MHz)', fontsize=10)
            if col_idx == 0:  # Left column
                ax.set_ylabel(ylabel, fontsize=10)
            
            # Set title only for top row
            if metric_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
            # Add legend - but make it smaller and more compact
            if col_idx == n_cols - 1:  # Right column
                ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Set x-axis ticks
            ax.set_xticks(clock_values)
            ax.tick_params(axis='both', labelsize=9)
            
            # Apply consistent y-axis limits
            ax.set_ylim(metric_limits[metric_name])
    
    plt.tight_layout()
    plt.savefig('visualization_all_tps_combined.png',
                dpi=300,
                bbox_inches='tight')
    print("\nSaved: visualization_all_tps_combined.png")
    plt.show()


def main(csv_file):
    """Main function to create visualizations."""

    # Read data to get available TPS values
    df = pd.read_csv(csv_file)
    tps_values = sorted(df['tps'].unique())

    print(f"Available TPS values: {tps_values}")
    print("\nCreating combined visualization with all TPS values...")
    
    # Create single combined visualization with all TPS values
    create_all_tps_combined_visualization(csv_file)


def visualize_specific_tps(tps_value, csv_file):
    """Create combined visualization for a specific TPS value."""

    df = pd.read_csv(csv_file)
    available_tps = sorted(df['tps'].unique())

    if tps_value not in available_tps:
        print(
            f"TPS value {tps_value} not found. Available values: {available_tps}"
        )
        return

    print(f"Creating combined visualization for TPS = {tps_value}")
    create_combined_visualization(tps_value, csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create visualizations from CSV data')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    parser.add_argument('--tps', type=int, help='Specific TPS value to visualize (optional)')
    
    args = parser.parse_args()
    
    if args.tps is not None:
        # Generate visualization for specific TPS value
        visualize_specific_tps(args.tps, args.csv_file)
    else:
        # Generate all visualizations
        main(args.csv_file)