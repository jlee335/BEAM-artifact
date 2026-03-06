# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TIME_LIMIT = 60

def parse_directory_name(dir_name):
    """Parse directory name to extract parameters."""
    pattern = r'tp(\d+)_pp(\d+)_f(\d+)_tps(\d+)_in(\d+)_out(\d+)'
    match = re.match(pattern, dir_name)
    if match:
        return {
            'tp': int(match.group(1)),
            'pp': int(match.group(2)),
            'clock': int(match.group(3)),
            'tps': int(match.group(4)),
            'in': int(match.group(5)),
            'out': int(match.group(6))
        }
    return None


def calculate_gpu_power_30s(energy_csv_path):
    """Calculate average GPU power from first 30 seconds of energy data."""
    try:
        df = pd.read_csv(energy_csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get the first timestamp
        start_time = df['timestamp'].iloc[0]

        # Filter data for first 30 seconds
        end_time = start_time + pd.Timedelta(seconds=30)
        df_30s = df[df['timestamp'] <= end_time]

        if len(df_30s) < 2:
            return np.nan

        # Calculate power for each GPU (Watts = Joules / time_diff)
        gpu_powers = []
        for gpu_id in range(8):  # Assuming 8 GPUs based on file structure
            joule_col = f'gpu_{gpu_id}_joules'
            if joule_col in df_30s.columns:
                # Get energy difference over 30 seconds
                energy_diff = df_30s[joule_col].iloc[-1] - df_30s[
                    joule_col].iloc[0]
                time_diff = (df_30s['timestamp'].iloc[-1] -
                             df_30s['timestamp'].iloc[0]).total_seconds()

                if time_diff > 0:
                    power_mw = energy_diff / time_diff  # Power in milliwatts (mJ/s = mW)
                    power_w = power_mw / 1000.0  # Convert mW to W
                    gpu_powers.append(power_w)

        return np.sum(gpu_powers) if gpu_powers else np.nan

    except Exception as e:
        print(f"Error processing {energy_csv_path}: {e}")
        return np.nan


def filter_json_by_time_limit(json_path, time_limit=TIME_LIMIT):
    """Load and filter JSON data to exclude requests/ITLs after time_limit."""
    with open(json_path) as f:
        data = json.load(f)
    
    entry_times = data.get('entry_times', [])
    ttfts = data.get('ttfts', [])
    itls = data.get('itls', [])
    
    filtered_ttfts = []
    filtered_itls = []
    
    for i, entry_time in enumerate(entry_times):
        # Skip requests that start after time_limit
        if entry_time >= time_limit:
            continue
        
        ttft = ttfts[i]
        request_itls = itls[i]
        
        # Filter ITLs that occur after time_limit
        filtered_request_itls = []
        cumulative_time = entry_time + ttft
        
        for itl in request_itls:
            if cumulative_time >= time_limit:
                break
            filtered_request_itls.append(itl)
            cumulative_time += itl
        
        # Only include request if it has at least the TTFT within time_limit
        if entry_time + ttft < time_limit:
            filtered_ttfts.append(ttft)
            filtered_itls.append(filtered_request_itls)
    
    return filtered_ttfts, filtered_itls


def extract_ttft_tbt_from_json(json_path):
    """Extract TTFT and TBT metrics from vLLM JSON file."""
    try:
        filtered_ttfts, filtered_itls = filter_json_by_time_limit(json_path)
        
        if not filtered_ttfts:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Calculate metrics from filtered data
        ttft_mean = np.mean(filtered_ttfts) * 1000  # Convert to ms
        ttft_p99 = np.percentile(filtered_ttfts, 99) * 1000
        ttft_std = np.std(filtered_ttfts) * 1000
        
        # Flatten all ITLs
        all_itls = [itl for request_itls in filtered_itls for itl in request_itls]
        # sort
        all_itls = sorted(all_itls)
        
        if all_itls:
            tbt_mean = np.mean(all_itls) * 1000  # Convert to ms
            tbt_p90 = np.percentile(all_itls, 90) * 1000
            tbt_std = np.std(all_itls) * 1000
        else:
            tbt_mean = tbt_p90 = tbt_std = np.nan

        return ttft_mean, tbt_mean, ttft_p99, tbt_p90, ttft_std, tbt_std

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Collect metrics from vLLM profiling results')
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Target folder path to process (e.g., shorttime_20s, fulltime)')
    parser.add_argument(
        '--gpu-name',
        type=str,
        required=True,
        help='GPU name (e.g., "NVIDIA A100-SXM4-80GB") for output filename')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., "Qwen/Qwen2.5-32B") for output filename')

    args = parser.parse_args()

    # Use the specified folder path
    base_dir = Path(args.folder)
    if not base_dir.exists():
        print(f"Error: Folder '{args.folder}' does not exist!")
        return

    print(f"Processing folder: {base_dir.absolute()}")
    results = []

    # Find all directories matching the pattern
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir():
            params = parse_directory_name(dir_path.name)
            if params:
                print(f"Processing {dir_path.name}...")

                # Find the traced_dataset subdirectory
                traced_dir = dir_path / 'traced_dataset'
                if traced_dir.exists():
                    # Find energy CSV file
                    energy_csv = traced_dir / 'gpu_energy_and_frequency_huggyllama_llama-13b.csv'
                    gpu_power = np.nan
                    if energy_csv.exists():
                        gpu_power = calculate_gpu_power_30s(energy_csv)

                    # Find vLLM JSON file
                    ttft_mean, tbt_mean, ttft_p99, tbt_p90, ttft_std, tbt_std = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    json_files = list(traced_dir.glob('vllm-*.json'))
                    # remove file with _estimations in name 
                    json_files = [file for file in json_files if '_estimations' not in file.name]
                    if json_files:
                        ttft_mean, tbt_mean, ttft_p99, tbt_p90, ttft_std, tbt_std = extract_ttft_tbt_from_json(
                            json_files[0])

                    results.append({
                        'clock': params['clock'],
                        'tp': params['tp'],
                        'pp': params['pp'],
                        'tps': params['tps'],
                        'gpu_power': gpu_power,
                        'ttft_mean': ttft_mean,
                        'tbt_mean': tbt_mean,
                        'ttft_p99': ttft_p99,
                        'tbt_p90': tbt_p90,
                        'ttft_std': ttft_std,
                        'tbt_std': tbt_std
                    })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        # sort w.r.t lexicographic order of clock, tp, pp, tps
        df = df.sort_values(by=['clock', 'tp', 'pp', 'tps'])
        model_name_clean = args.model.replace("/", "_")
        output_file = f'dynamo_dvfs_profile_{args.gpu_name}_{model_name_clean}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(results)} records to {output_file}")
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("No data found!")


if __name__ == "__main__":
    main()
