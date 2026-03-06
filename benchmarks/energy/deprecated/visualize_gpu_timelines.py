#!/usr/bin/env python3
"""
GPU Timeline Visualization Tool

Creates timeline plots showing GPU execution blocks from batch logs
for both prefill and decode servers across multiple GPUs.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

def read_batch_logs(base_path):
    """Read all batch log CSV files from traced_dataset directory"""
    data = []
    
    # Check if traced_dataset exists
    traced_dataset_path = os.path.join(base_path, 'traced_dataset')
    if not os.path.exists(traced_dataset_path):
        print(f"Warning: traced_dataset directory not found at {traced_dataset_path}")
        print("Attempting to use base_path directly...")
        traced_dataset_path = base_path
    
    # Find all batch_log_GPU_*.csv files
    log_files = []
    for i in range(4):  # Check for GPU 0-3
        filename = f'batch_log_GPU_{i}.csv'
        file_path = os.path.join(traced_dataset_path, filename)
        if os.path.exists(file_path):
            log_files.append((filename, f'GPU {i}'))
    
    if not log_files:
        print(f"No batch log files found in {traced_dataset_path}")
        return data
    
    print(f"Found {len(log_files)} batch log files")
    print("Reading batch log files...")
    for filename, display_name in tqdm(log_files, desc="Loading CSV files"):
        file_path = os.path.join(traced_dataset_path, filename)
        
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                df['server_gpu'] = display_name
                df['server_type'] = 'traced'  # Generic server type
                data.append(df)
                tqdm.write(f"✓ Loaded {len(df)} records from {display_name}")
            else:
                tqdm.write(f"⚠ File is empty: {display_name}")
        except Exception as e:
            tqdm.write(f"✗ Error reading {display_name}: {e}")
    
    return data

def process_timeline_data(data_frames):
    """Process the data frames to create timeline segments"""
    timeline_data = []
    
    print("\nProcessing timeline data...")
    total_rows = sum(len(df) for df in data_frames)
    
    with tqdm(total=total_rows, desc="Processing timeline segments") as pbar:
        for df in data_frames:
            server_name = df['server_gpu'].iloc[0] if not df.empty else "Unknown"
            
            for _, row in df.iterrows():
                try:
                    # Parse datetime strings
                    start_time = pd.to_datetime(row['start_time'])
                    current_time = pd.to_datetime(row['current_time'])
                    
                    timeline_data.append({
                        'Server_GPU': row['server_gpu'],
                        'Server_Type': row['server_type'],
                        'Start': start_time,
                        'Finish': current_time,
                        'Duration_ms': row['elapsed_time'],
                        'Batch_Size': row['batch_size'],
                        'Step': row['step'],
                        'Phase': row['phase'],
                        'Num_Reqs': row['num_reqs'],
                        'Energy': row['curr_energy']
                    })
                    pbar.update(1)
                except Exception as e:
                    tqdm.write(f"✗ Error processing row in {server_name}: {e}")
                    pbar.update(1)
                    continue
    
    return pd.DataFrame(timeline_data)

def filter_timeline_by_time_limit(timeline_df, time_limit_seconds=None, time_start_seconds=None):
    """Filter timeline data to include events within a specified time window
    
    Args:
        timeline_df: DataFrame with timeline data
        time_limit_seconds: Duration in seconds (from start_time or time_start)
        time_start_seconds: Offset in seconds from earliest time to start displaying
    """
    if timeline_df.empty:
        return timeline_df
    
    # Find the earliest start time across all data
    earliest_time = timeline_df['Start'].min()
    
    # Determine start and end times for the filter
    if time_start_seconds is not None:
        start_time = earliest_time + timedelta(seconds=time_start_seconds)
    else:
        start_time = earliest_time
    
    if time_limit_seconds is not None:
        cutoff_time = start_time + timedelta(seconds=time_limit_seconds)
    else:
        cutoff_time = timeline_df['Finish'].max()
    
    # Filter to only include events that start before the cutoff and finish after start_time
    filtered_df = timeline_df[
        (timeline_df['Start'] < cutoff_time) & 
        (timeline_df['Finish'] > start_time)
    ].copy()
    
    # Trim events that start before start_time
    mask_start = filtered_df['Start'] < start_time
    filtered_df.loc[mask_start, 'Start'] = start_time
    
    # Trim events that extend beyond the cutoff
    mask_end = filtered_df['Finish'] > cutoff_time
    filtered_df.loc[mask_end, 'Finish'] = cutoff_time
    
    # Recalculate duration for trimmed events
    mask_trimmed = mask_start | mask_end
    filtered_df.loc[mask_trimmed, 'Duration_ms'] = (
        (filtered_df.loc[mask_trimmed, 'Finish'] - filtered_df.loc[mask_trimmed, 'Start']).dt.total_seconds() * 1000
    )
    
    # Print filter information
    print(f"\n✓ Filtered timeline data")
    print(f"  - Original segments: {len(timeline_df)}")
    print(f"  - Filtered segments: {len(filtered_df)}")
    if time_start_seconds is not None:
        print(f"  - Start offset: {time_start_seconds} seconds")
    if time_limit_seconds is not None:
        print(f"  - Duration: {time_limit_seconds} seconds")
    print(f"  - Time range: {start_time.strftime('%H:%M:%S.%f')[:-3]} to {cutoff_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    return filtered_df

def create_server_data(args):
    """Process a single server's data in parallel"""
    df_subset, server_gpu, y_center, box_height, color = args
    
    # Vectorized hover text creation
    hover_text = (
        "Server: " + df_subset['Server_GPU'].astype(str) + "<br>" +
        "Step: " + df_subset['Step'].astype(str) + ", Phase: " + df_subset['Phase'].astype(str) + "<br>" +
        "Duration: " + df_subset['Duration_ms'].round(2).astype(str) + " ms<br>" +
        "Batch Size: " + df_subset['Batch_Size'].astype(str) + "<br>" +
        "Num Reqs: " + df_subset['Num_Reqs'].astype(str) + "<br>" +
        "Start: " + df_subset['Start'].dt.strftime('%H:%M:%S.%f').str[:-3] + "<br>" +
        "End: " + df_subset['Finish'].dt.strftime('%H:%M:%S.%f').str[:-3] + "<br>" +
        "Energy: " + df_subset['Energy'].apply(lambda x: f"{x:,}")
    ).tolist()
    
    # Pre-compute all shape coordinates
    shapes_data = {
        'x0': df_subset['Start'].tolist(),
        'x1': df_subset['Finish'].tolist(),
        'y0': [y_center - box_height/2] * len(df_subset),
        'y1': [y_center + box_height/2] * len(df_subset),
        'fillcolor': color,
        'opacity': 0.7,
        'line_color': color
    }
    
    # Pre-compute scatter trace data  
    scatter_data = {
        'x': (df_subset['Start'] + (df_subset['Finish'] - df_subset['Start']) / 2).tolist(),
        'y': [y_center] * len(df_subset),
        'hovertext': hover_text,
        'name': server_gpu
    }
    
    # Legend trace data
    legend_data = {
        'x': [df_subset['Start'].min()],
        'y': [y_center], 
        'name': server_gpu,
        'color': color
    }
    
    return shapes_data, scatter_data, legend_data

def create_timeline_plot(timeline_df):
    """Create a Plotly timeline visualization with optimizations"""
    
    # Define colors and positions
    color_map = {
        'GPU 0': '#1f77b4',  # Blue
        'GPU 1': '#ff7f0e',  # Orange  
        'GPU 2': '#2ca02c',  # Green
        'GPU 3': '#d62728'   # Red
    }
    
    server_positions = {
        'GPU 0': 0,
        'GPU 1': 1,
        'GPU 2': 2,
        'GPU 3': 3
    }
    
    box_height = 1.0  # Boxes touch without overlapping (spacing = 1.0)
    fig = go.Figure()
    
    # Prepare arguments for parallel processing
    server_args = []
    unique_servers = timeline_df['Server_GPU'].unique()
    
    for server_gpu in unique_servers:
        df_subset = timeline_df[timeline_df['Server_GPU'] == server_gpu].copy()
        y_center = server_positions[server_gpu]
        color = color_map.get(server_gpu, '#333333')
        server_args.append((df_subset, server_gpu, y_center, box_height, color))
    
    # Process servers in parallel (if multiple servers, otherwise sequential)
    print("\nCreating visualization data...")
    if len(server_args) > 1:
        # Use parallel processing for multiple servers
        with ProcessPoolExecutor(max_workers=min(len(server_args), mp.cpu_count())) as executor:
            with tqdm(total=len(server_args), desc="Processing servers") as pbar:
                results = []
                for result in executor.map(create_server_data, server_args):
                    results.append(result)
                    pbar.update(1)
    else:
        # Sequential processing for single server
        results = []
        for args in tqdm(server_args, desc="Processing servers"):
            results.append(create_server_data(args))
    
    # Collect all shapes and traces
    print("Building plot components...")
    all_shapes = []
    all_scatter_data = []
    legend_traces = []
    
    total_segments = sum(len(shapes_data['x0']) for shapes_data, _, _ in results)
    
    with tqdm(total=len(results), desc="Building plot traces") as pbar:
        for shapes_data, scatter_data, legend_data in results:
            # Add legend trace
            legend_traces.append(go.Scatter(
                x=legend_data['x'],
                y=legend_data['y'],
                mode='markers',
                marker=dict(size=0.1, color=legend_data['color']),
                name=legend_data['name'],
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # Collect shapes data
            for i in range(len(shapes_data['x0'])):
                all_shapes.append({
                    'type': 'rect',
                    'x0': shapes_data['x0'][i],
                    'x1': shapes_data['x1'][i],
                    'y0': shapes_data['y0'][i],
                    'y1': shapes_data['y1'][i],
                    'fillcolor': shapes_data['fillcolor'],
                    'opacity': shapes_data['opacity'],
                    'line': dict(color=shapes_data['line_color'], width=1)
                })
            
            # Add scatter trace for hover (all points at once)
            all_scatter_data.append(go.Scatter(
                x=scatter_data['x'],
                y=scatter_data['y'],
                mode='markers',
                marker=dict(size=15, opacity=0),
                hoverinfo='text',
                hovertext=scatter_data['hovertext'],
                showlegend=False
            ))
            
            pbar.update(1)
    
    # Add all legend traces
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Add all scatter traces
    for trace in all_scatter_data:
        fig.add_trace(trace)
    
    # Add all shapes at once (much faster than one-by-one)
    fig.update_layout(shapes=all_shapes)
    
    # Update layout for compact visualization
    # Determine the number of GPUs dynamically
    unique_gpus = sorted(timeline_df['Server_GPU'].unique())
    num_gpus = len(unique_gpus)
    
    fig.update_layout(
        title={
            'text': 'GPU Timeline Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}  # Smaller title font
        },
        xaxis=dict(
            title='Time',
            type='date',
            tickformat='%H:%M:%S.%L',
            title_font=dict(size=12),  # Smaller axis title
            tickfont=dict(size=10)     # Smaller tick font
        ),
        yaxis=dict(
            title='GPU',
            tickvals=list(range(num_gpus)),
            ticktext=unique_gpus,
            range=[-0.55, num_gpus - 0.45],  # Minimal padding for compact view with touching boxes
            title_font=dict(size=12),   # Smaller axis title
            tickfont=dict(size=10)      # Smaller tick font
        ),
        height=400,  # More compact height
        margin=dict(l=80, r=20, t=60, b=50),  # Tighter margins
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right", 
            x=1,
            font=dict(size=10)  # Smaller legend font
        )
    )
    
    return fig

def main():
    """Main function to create the timeline visualization"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Create GPU timeline visualization from batch logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python visualize_gpu_timelines.py benchmark_results --time-limit 60\n'
               '  python visualize_gpu_timelines.py benchmark_results --time-start 30 --time-limit 60'
    )
    parser.add_argument('directory', 
                        help='Benchmark results directory containing batch logs')
    parser.add_argument('--time-limit', 
                        type=float, 
                        metavar='SECONDS',
                        help='Duration in seconds to visualize (from start or time-start)')
    parser.add_argument('--time-start',
                        type=float,
                        metavar='SECONDS',
                        help='Start offset in seconds from beginning of trace')
    
    args = parser.parse_args()
    base_path = args.directory
    
    # If relative path, make it relative to script directory
    if not os.path.isabs(base_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(script_dir, base_path)
    
    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' does not exist!")
        sys.exit(1)
    
    print(f"Looking for batch logs in: {base_path}")
    if args.time_start:
        print(f"Time start offset: {args.time_start} seconds")
    if args.time_limit:
        print(f"Time limit: {args.time_limit} seconds")
    
    # Read all batch log files
    data_frames = read_batch_logs(base_path)
    
    if not data_frames:
        print("\n❌ No data found! Please check the file paths.")
        return
    
    print(f"\n✓ Loaded {len(data_frames)} data files successfully")
    
    # Process the data into timeline format
    timeline_df = process_timeline_data(data_frames)
    
    if timeline_df.empty:
        print("\n❌ No timeline data could be processed!")
        return
        
    print(f"\n✓ Created timeline with {len(timeline_df)} segments")
    
    # Apply time filter if specified
    if args.time_limit or args.time_start:
        timeline_df = filter_timeline_by_time_limit(
            timeline_df, 
            time_limit_seconds=args.time_limit,
            time_start_seconds=args.time_start
        )
        if timeline_df.empty:
            print("\n❌ No data remains after applying time filter!")
            return
    
    # Create the visualization
    print("\nGenerating timeline plot...")
    fig = create_timeline_plot(timeline_df)
    
    # Save the plot with dynamic naming
    dir_name = os.path.basename(base_path.rstrip(os.sep))
    time_suffix = ''
    if args.time_start:
        time_suffix += f'_{int(args.time_start)}s'
    if args.time_limit:
        if args.time_start:
            time_suffix += f'-{int(args.time_start + args.time_limit)}s'
        else:
            time_suffix += f'_{int(args.time_limit)}s'
    output_file = os.path.join(os.path.dirname(base_path), f'gpu_timeline_{dir_name}{time_suffix}.html')
    
    print("Saving visualization...")
    with tqdm(total=1, desc="Writing HTML file") as pbar:
        fig.write_html(output_file)
        pbar.update(1)
    
    print(f"\n✓ Timeline plot saved to: {output_file}")
    
    # show the plot
    fig.show()
    
    
    # Print summary statistics
    print("\nTimeline Summary:")
    print("=" * 50)
    servers = timeline_df['Server_GPU'].unique()
    for server_gpu in tqdm(servers, desc="Computing summary stats"):
        subset = timeline_df[timeline_df['Server_GPU'] == server_gpu]
        total_duration = subset['Duration_ms'].sum()
        avg_batch_size = subset['Batch_Size'].mean()
        tqdm.write(f"{server_gpu}:")
        tqdm.write(f"  - Total segments: {len(subset)}")
        tqdm.write(f"  - Total duration: {total_duration:.2f} ms")
        tqdm.write(f"  - Average batch size: {avg_batch_size:.1f}")
    
    print(f"\n🎉 Visualization complete! Open {output_file} in your browser.")

if __name__ == "__main__":
    main()
