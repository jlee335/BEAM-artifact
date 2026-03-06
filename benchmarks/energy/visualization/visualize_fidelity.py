#!/usr/bin/env python3
"""
Visualize fidelity test results by comparing actual vs estimated TTFT and TBT/ITL.
"""

import json
import sys
import argparse
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def find_json_files(test_dir):
    """Find the traced dataset and estimations JSON files in the test directory."""
    test_path = Path(test_dir)
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Look for JSON files in subdirectories
    traced_files = []
    estimation_files = []
    
    for json_file in test_path.rglob("*.json"):
        if "_estimations.json" in json_file.name:
            estimation_files.append(json_file)
        elif json_file.name.endswith(".json"):
            traced_files.append(json_file)
    
    # Match traced files with their estimation counterparts
    pairs = []
    for traced_file in traced_files:
        estimation_file = traced_file.parent / f"{traced_file.stem}_estimations.json"
        if estimation_file in estimation_files:
            pairs.append((traced_file, estimation_file))
    
    if not pairs:
        raise FileNotFoundError(f"No matching traced/estimation file pairs found in {test_dir}")
    
    return pairs


def load_json_file(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_ttft_data(traced_data, estimations_data):
    """Extract TTFT data for comparison."""
    ttfts_actual = []
    ttfts_estimated = []
    
    # Get actual TTFTs
    if isinstance(traced_data, dict) and 'ttfts' in traced_data:
        ttfts_actual = traced_data['ttfts']
    
    # Get estimated TTFTs
    if isinstance(estimations_data, dict) and 'estimations' in estimations_data:
        for est in estimations_data['estimations']:
            ttfts_estimated.append(est['estimated_ttft'])
    
    # Filter out None values and convert to ms
    valid_pairs = []
    for i, (actual, estimated) in enumerate(zip(ttfts_actual, ttfts_estimated)):
        if actual is not None and estimated is not None:
            valid_pairs.append((i, actual * 1000, estimated * 1000))
    
    return valid_pairs


def extract_tbt_itl_data(traced_data, estimations_data):
    """Extract TBT/ITL data for comparison."""
    all_actual_itls = []
    all_estimated_tbts = []
    per_request_data = []
    
    # Get actual ITLs and estimated TBTs per request
    itls_actual = []
    tbts_estimated = []
    
    if isinstance(traced_data, dict) and 'itls' in traced_data:
        itls_actual = traced_data['itls']
    
    if isinstance(estimations_data, dict) and 'estimations' in estimations_data:
        for est in estimations_data['estimations']:
            tbts = est.get('estimated_tbts', [])
            tbts_estimated.append(tbts)
    
    # Collect per-request data and aggregate all tokens
    longest_request_idx = -1
    longest_request_len = 0
    
    for i, (itl_list, tbt_list) in enumerate(zip(itls_actual, tbts_estimated)):
        actual_itls_ms = [itl * 1000 for itl in itl_list if itl is not None]
        estimated_tbts_ms = [tbt * 1000 for tbt in tbt_list if tbt is not None]
        
        # shift estimated_tbts by 1
        estimated_tbts_ms = estimated_tbts_ms[1:]
        actual_itls_ms = actual_itls_ms[:-1]
        
        # Track the longest request
        n_tokens = min(len(actual_itls_ms), len(estimated_tbts_ms))
        if n_tokens > longest_request_len:
            longest_request_len = n_tokens
            longest_request_idx = i
        
        per_request_data.append({
            'request_idx': i,
            'actual_itls': actual_itls_ms,
            'estimated_tbts': estimated_tbts_ms,
            'n_tokens': n_tokens
        })
        
        # Aggregate all tokens
        all_actual_itls.extend(actual_itls_ms)
        all_estimated_tbts.extend(estimated_tbts_ms)
    
    # Match the number of tokens for aggregate data
    n_tokens = min(len(all_actual_itls), len(all_estimated_tbts))
    all_actual_itls = all_actual_itls[:n_tokens]
    all_estimated_tbts = all_estimated_tbts[:n_tokens]
    
    # Get the longest request data
    longest_request = None
    if longest_request_idx >= 0:
        longest_request = per_request_data[longest_request_idx]
        # Match token counts for the longest request
        n = longest_request['n_tokens']
        longest_request['actual_itls'] = longest_request['actual_itls'][:n]
        longest_request['estimated_tbts'] = longest_request['estimated_tbts'][:n]
    
    return all_actual_itls, all_estimated_tbts, longest_request, per_request_data


def plot_ttft_comparison(valid_pairs, output_dir):
    """Create TTFT comparison plots."""
    if not valid_pairs:
        print("No valid TTFT data to plot")
        return
    
    indices, actual, estimated = zip(*valid_pairs)
    indices = list(indices)
    actual = list(actual)
    estimated = list(estimated)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Request-by-request comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(indices, actual, 'o-', label='Actual TTFT', alpha=0.7, markersize=4)
    ax1.plot(indices, estimated, 's-', label='Estimated TTFT', alpha=0.7, markersize=4)
    ax1.set_xlabel('Request Index')
    ax1.set_ylabel('TTFT (ms)')
    ax1.set_title('TTFT: Actual vs Estimated (Request-by-Request)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 2. Scatter plot: Actual vs Estimated
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(actual, estimated, alpha=0.6, s=30)
    
    # Perfect prediction line
    max_val = max(max(actual), max(estimated))
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction', linewidth=2)
    
    ax2.set_xlabel('Actual TTFT (ms)')
    ax2.set_ylabel('Estimated TTFT (ms)')
    ax2.set_title('TTFT: Actual vs Estimated (Scatter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.axis('equal')
    
    # 3. Error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    errors = [est - act for act, est in zip(actual, estimated)]
    ax3.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax3.axvline(x=statistics.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {statistics.mean(errors):.2f} ms')
    ax3.set_xlabel('Error (Estimated - Actual) (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('TTFT Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relative error distribution
    ax4 = fig.add_subplot(gs[2, 0])
    rel_errors = [(est - act) / act * 100 for act, est in zip(actual, estimated) if act != 0]
    ax4.hist(rel_errors, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax4.axvline(x=statistics.mean(rel_errors), color='g', linestyle='-', linewidth=2, 
                label=f'Mean: {statistics.mean(rel_errors):.2f}%')
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('TTFT Relative Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Calculate MAPE
    mape = statistics.mean([abs((est - act) / act * 100) for act, est in zip(actual, estimated) if act != 0])
    
    # 5. Statistics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"""TTFT Statistics Summary
    
Valid Comparisons: {len(valid_pairs)}

Actual TTFT:
  Mean: {statistics.mean(actual):.2f} ms
  Median: {statistics.median(actual):.2f} ms
  Std Dev: {statistics.stdev(actual):.2f} ms

Estimated TTFT:
  Mean: {statistics.mean(estimated):.2f} ms
  Median: {statistics.median(estimated):.2f} ms
  Std Dev: {statistics.stdev(estimated):.2f} ms

Error Metrics:
  Mean Difference: {statistics.mean(errors):.2f} ms
  Median Difference: {statistics.median(errors):.2f} ms
  Mean Relative Error: {statistics.mean(rel_errors):.2f}%
  Median Relative Error: {statistics.median(rel_errors):.2f}%
  MAPE: {mape:.2f}%
  Max Absolute Error: {max(abs(e) for e in errors):.2f} ms
"""
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
             verticalalignment='center')
    
    plt.suptitle('TTFT Comparison: Actual vs Estimated', fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = output_dir / 'ttft_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TTFT comparison plot saved to: {output_path}")
    plt.close()


def plot_tbt_itl_comparison(all_actual_itls, all_estimated_tbts, longest_request, per_request_data, output_dir, use_all_tokens=False):
    """Create TBT/ITL comparison plots."""
    if not all_actual_itls or not all_estimated_tbts:
        print("No valid TBT/ITL data to plot")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Token-by-token comparison
    ax1 = fig.add_subplot(gs[0, :])
    if use_all_tokens:
        # Use all concatenated tokens
        actual_itls = all_actual_itls
        estimated_tbts = all_estimated_tbts
        n_tokens = len(actual_itls)
        
        indices = list(range(n_tokens))
        ax1.plot(indices, actual_itls, 'o-', label='Actual ITL', alpha=0.7, markersize=2, linewidth=1)
        ax1.plot(indices, estimated_tbts, 's-', label='Estimated TBT', alpha=0.7, markersize=2, linewidth=1)
        ax1.set_xlabel('Token Index (All Requests Concatenated)')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title(f'ITL/TBT Timeline: All Requests Concatenated ({n_tokens} tokens)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    elif longest_request:
        # Use longest request only
        request_idx = longest_request['request_idx']
        actual_itls = longest_request['actual_itls']
        estimated_tbts = longest_request['estimated_tbts']
        n_tokens = len(actual_itls)
        
        indices = list(range(n_tokens))
        ax1.plot(indices, actual_itls, 'o-', label='Actual ITL', alpha=0.7, markersize=3, linewidth=1.5)
        ax1.plot(indices, estimated_tbts, 's-', label='Estimated TBT', alpha=0.7, markersize=3, linewidth=1.5)
        ax1.set_xlabel('Token Index')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title(f'ITL/TBT Timeline: Request {request_idx} (Longest Request, {n_tokens} tokens)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, 'No request data available', ha='center', va='center')
        ax1.set_title('ITL/TBT Timeline: No Data')
    
    # 2. Scatter plot: Actual vs Estimated
    ax2 = fig.add_subplot(gs[1, 0])
    # Sample data if too many points
    sample_actual = all_actual_itls
    sample_estimated = all_estimated_tbts
    
    ax2.scatter(sample_actual, sample_estimated, alpha=0.02, s=3)
    
    # Perfect prediction line
    max_val = max(max(all_actual_itls), max(all_estimated_tbts))
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction', linewidth=2)
    
    ax2.set_xlabel('Actual ITL (ms)')
    ax2.set_ylabel('Estimated TBT (ms)')
    ax2.set_title('ITL/TBT: Actual vs Estimated (Scatter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.axis('equal')
    
    # 3. Error distribution - iterate through requests
    ax3 = fig.add_subplot(gs[1, 1])
    errors = []
    for request in per_request_data:
        actual_itls = request['actual_itls']
        estimated_tbts = request['estimated_tbts']
        n_tokens = min(len(actual_itls), len(estimated_tbts))
        for i in range(n_tokens):
            errors.append(estimated_tbts[i] - actual_itls[i])
    
    ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax3.axvline(x=statistics.mean(errors), color='g', linestyle='-', linewidth=2, 
                label=f'Mean: {statistics.mean(errors):.2f} ms')
    ax3.set_xlabel('Error (Estimated - Actual) (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('ITL/TBT Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relative error distribution - iterate through requests
    ax4 = fig.add_subplot(gs[2, 0])
    rel_errors = []
    for request in per_request_data:
        actual_itls = request['actual_itls']
        estimated_tbts = request['estimated_tbts']
        n_tokens = min(len(actual_itls), len(estimated_tbts))
        for i in range(n_tokens):
            if actual_itls[i] != 0:
                rel_errors.append((estimated_tbts[i] - actual_itls[i]) / actual_itls[i] * 100)
    
    # Clip extreme outliers for better visualization
    rel_errors_clipped = [max(-100, min(100, e)) for e in rel_errors]
    ax4.hist(rel_errors_clipped, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax4.axvline(x=statistics.mean(rel_errors), color='g', linestyle='-', linewidth=2, 
                label=f'Mean: {statistics.mean(rel_errors):.2f}%')
    ax4.set_xlabel('Relative Error (%) [clipped to ±100%]')
    ax4.set_ylabel('Frequency')
    ax4.set_title('ITL/TBT Relative Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Calculate MAPE - iterate through requests
    mape = statistics.mean([abs((request['estimated_tbts'][i] - request['actual_itls'][i]) / request['actual_itls'][i] * 100) 
                           for request in per_request_data
                           for i in range(min(len(request['actual_itls']), len(request['estimated_tbts'])))
                           if request['actual_itls'][i] != 0])
    
    # 5. Statistics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"""ITL/TBT Statistics Summary
    
Total Tokens Compared: {len(all_actual_itls)}

Actual ITL:
  Mean: {statistics.mean(all_actual_itls):.2f} ms
  Median: {statistics.median(all_actual_itls):.2f} ms
  Std Dev: {statistics.stdev(all_actual_itls):.2f} ms
  Min: {min(all_actual_itls):.2f} ms
  Max: {max(all_actual_itls):.2f} ms

Estimated TBT:
  Mean: {statistics.mean(all_estimated_tbts):.2f} ms
  Median: {statistics.median(all_estimated_tbts):.2f} ms
  Std Dev: {statistics.stdev(all_estimated_tbts):.2f} ms
  Min: {min(all_estimated_tbts):.2f} ms
  Max: {max(all_estimated_tbts):.2f} ms

Error Metrics:
  Mean Difference: {statistics.mean(errors):.2f} ms
  Median Difference: {statistics.median(errors):.2f} ms
  Mean Relative Error: {statistics.mean(rel_errors):.2f}%
  Median Relative Error: {statistics.median(rel_errors):.2f}%
  MAPE: {mape:.2f}%
"""
    ax5.text(0.1, 0.5, stats_text, fontsize=9, family='monospace', 
             verticalalignment='center')
    
    plt.suptitle('ITL/TBT Comparison: Actual vs Estimated', fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = output_dir / 'tbt_itl_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TBT/ITL comparison plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize fidelity test results comparing actual vs estimated metrics'
    )
    parser.add_argument(
        'test_dir',
        type=str,
        help='Path to the fidelity test directory (e.g., fidelity_test_20251024_131537/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as test_dir)'
    )
    parser.add_argument(
        '--itl-all',
        action='store_true',
        help='Show all concatenated token indices instead of just the longest request in ITL/TBT timeline'
    )
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir) if args.output_dir else test_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching for JSON files in: {test_dir}")
    
    # Find traced and estimation JSON file pairs
    file_pairs = find_json_files(test_dir)
    print(f"Found {len(file_pairs)} file pair(s)")
    
    # Process each pair
    for traced_file, estimation_file in file_pairs:
        print(f"\nProcessing:")
        print(f"  Traced: {traced_file.name}")
        print(f"  Estimations: {estimation_file.name}")
        
        # Load data
        traced_data = load_json_file(traced_file)
        estimations_data = load_json_file(estimation_file)
        
        # Extract data
        ttft_pairs = extract_ttft_data(traced_data, estimations_data)
        actual_itls, estimated_tbts, longest_request, per_request_data = extract_tbt_itl_data(traced_data, estimations_data)
        
        # Create plots
        print(f"\nGenerating plots...")
        plot_ttft_comparison(ttft_pairs, output_dir)
        plot_tbt_itl_comparison(actual_itls, estimated_tbts, longest_request, per_request_data, output_dir, use_all_tokens=args.itl_all)
        
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

