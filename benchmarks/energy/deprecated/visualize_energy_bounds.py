# SPDX-License-Identifier: Apache-2.0
# dvfs_profile_b'<GPU_NAME>'_<MODEL_NAME>.csv

import os
import re

# matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Global variable for max batch size threshold in graphs
MAX_BATCH_SIZE = 4096  # Set this as needed

# Parse all files in directory with .csv and pattern match
# clock,batch_size,time_taken,energy_consumption,valid,total_ctx_len


def set_fig_size(plt):
    # Larger plot size
    plt.figure(figsize=(16, 8))


def save_fig(plt, model_name, gpu_name, mode, suffix):
    # y start from 0
    plt.ylim(0, None)
    plt.xlim(0, 2000)
    plt.savefig(f'[{mode}]{model_name}_{gpu_name}_{suffix}.png',
                bbox_inches='tight')
    plt.close()


for file in os.listdir('.'):
    if file.endswith('.csv'):
        # Parse the file name to get the GPU name and model name with regex
        # dvfs_profile_NVIDIA RTX A6000_ByteResearch_Llama-3-8B-Instruct_tp1_pp4_one
        # GPU names start with NVIDIA, end with _, rest is model name, then run name
        match = re.search(
            r'dvfs_profile_([^_]+(?: [^_]+)*)_([^_]+(?:_[^_]+)*)_(.*)\.csv',
            file)
        if match:
            # Use regex groups to extract GPU and model names appropriately
            gpu_name = match.group(1).strip()
            model_name = match.group(2).strip()
            # Only allow A6000 and A100 in GPU name, else continue
            if "A6000" not in gpu_name:
                continue

            print(f"GPU: {gpu_name}, Model: {model_name}")
            df_orig = pd.read_csv(file)

            unique_batch_sizes = df_orig['batch_size'].unique()
            unique_batch_sizes.sort()

            # mJ to J
            df_orig[
                'energy_consumption'] = df_orig['energy_consumption'] / 1000

            df_orig['energy-per-token'] = df_orig[
                'energy_consumption'] / df_orig['batch_size']
            df_orig['power'] = df_orig['energy_consumption'] / df_orig[
                'time_taken']
            # Add time-per-token column
            df_orig['time-per-token'] = df_orig['time_taken'] / df_orig[
                'batch_size']

            # get unique batch sizes from df_orig, filter by MAX_BATCH_SIZE
            unique_batch_sizes = df_orig[
                df_orig['batch_size'] <=
                MAX_BATCH_SIZE]['batch_size'].unique()
            unique_batch_sizes.sort()

            if "A100" not in gpu_name:
                standard_clock = 1770
            else:
                standard_clock = 1050

            df = df_orig

            import numpy as np

            # Parameters
            request_length = 4096
            decode_size = 256  # configurable decode size
            chunk_sizes = [4096, 2048, 1024, 512, 256, 128, 64]

            # Use only standard clock rows
            df_standard_clock = df[df['clock'] == standard_clock]

            # Get mapping: batch_size -> mean time_taken, mean energy_consumption
            batch_time_map = df_standard_clock.groupby(
                'batch_size')['time_taken'].mean().to_dict()
            batch_energy_map = df_standard_clock.groupby(
                'batch_size')['energy_consumption'].mean().to_dict()

            # For interpolation
            batch_sizes_sorted = np.array(sorted(batch_time_map.keys()))
            time_taken_sorted = np.array(
                [batch_time_map[bs] for bs in batch_sizes_sorted])
            energy_sorted = np.array(
                [batch_energy_map[bs] for bs in batch_sizes_sorted])

            def find_smallest_batch_size_ge(bs, batch_sizes_sorted):
                """
                Find the smallest batch size >= bs in batch_sizes_sorted.
                If not found, raise ValueError.
                """
                for b in batch_sizes_sorted:
                    if b >= bs:
                        return b
                raise ValueError(
                    f"No batch size >= {bs} found in available batch sizes: {batch_sizes_sorted}"
                )

            def get_time_taken(bs, clock=standard_clock):
                # Use only rows with the specified clock
                df_clock = df[df['clock'] == clock]
                batch_time_map_clock = df_clock.groupby(
                    'batch_size')['time_taken'].mean().to_dict()
                batch_sizes_sorted_clock = sorted(batch_time_map_clock.keys())
                if not batch_sizes_sorted_clock:
                    raise ValueError(f"No batch sizes found for clock {clock}")
                try:
                    b = find_smallest_batch_size_ge(bs,
                                                    batch_sizes_sorted_clock)
                    return batch_time_map_clock[b]
                except ValueError:
                    raise ValueError(
                        f"Requested batch size {bs} not found and no larger batch size available for clock {clock}"
                    )

            def get_energy(bs, clock=standard_clock):
                # Use only rows with the specified clock
                df_clock = df[df['clock'] == clock]
                batch_energy_map_clock = df_clock.groupby(
                    'batch_size')['energy_consumption'].mean().to_dict()
                batch_sizes_sorted_clock = sorted(
                    batch_energy_map_clock.keys())
                if not batch_sizes_sorted_clock:
                    raise ValueError(f"No batch sizes found for clock {clock}")
                try:
                    b = find_smallest_batch_size_ge(bs,
                                                    batch_sizes_sorted_clock)
                    return batch_energy_map_clock[b]
                except ValueError:
                    raise ValueError(
                        f"Requested batch size {bs} not found and no larger batch size available for clock {clock}"
                    )

            def get_idle_power(gpu_name, clock):
                """
                Estimate idle power for the given GPU/model/clock.
                Tries to use _idle.csv if available, else falls back to lowest batch size's power.
                """
                idle_csv = f"dvfs_profile_{gpu_name}_idle.csv"
                if os.path.exists(idle_csv):
                    df_idle = pd.read_csv(idle_csv)
                    # Convert mJ to J for energy_consumption
                    if 'energy_consumption' in df_idle.columns:
                        df_idle['energy_consumption'] = df_idle[
                            'energy_consumption'] / 1000
                    # Compute power manually as energy_consumption / time_taken
                    if 'energy_consumption' in df_idle.columns and 'time_taken' in df_idle.columns:
                        df_idle['power'] = df_idle[
                            'energy_consumption'] / df_idle['time_taken']
                    else:
                        raise ValueError(
                            "Idle CSV missing required columns for power calculation"
                        )
                    # Use the mean power at the standard clock, or fallback to any available
                    if 'clock' in df_idle.columns and (df_idle['clock']
                                                       == clock).any():
                        return df_idle[df_idle['clock'] ==
                                       clock]['power'].mean()
                    else:
                        return df_idle['power'].mean()
                else:
                    raise ValueError(f"Idle CSV not found for {gpu_name}")

            idle_power = get_idle_power(gpu_name, standard_clock)

            # For bounded mode, we need to estimate decode energy and idle energy
            # We'll use the bounded csv if available, else fallback to unbounded

            prefill_energies = []
            bounded_energies = []
            prefill_plus_decode_energies = []
            idle_energies = []
            decode_only_energies = []

            def get_dvfs_clock(bound_size, decode_size):
                """
                For a given bound_size (chunk size), find the clock frequency that minimizes
                total energy for a decode window (decode + idle), i.e., 
                energy = decode_energy + idle_energy, where
                decode_energy = get_energy(decode_size, clock)
                idle_energy = idle_power(clock) * idle_time
                idle_time = chunk_time - decode_time (if > 0)
                Returns: (best_clock, min_total_energy)
                """
                # Get all available clocks from the dataframe
                available_clocks = sorted(df['clock'].unique())
                min_total_energy = None
                best_clock = None

                for clock in available_clocks:
                    try:
                        # Get decode and chunk times at this clock
                        decode_time = get_time_taken(decode_size, clock)
                        chunk_time = get_time_taken(bound_size, clock)
                        idle_time = max(chunk_time - decode_time, 0)
                        # Get decode energy at this clock
                        decode_energy = get_energy(decode_size, clock)
                        # Get idle power at this clock
                        try:
                            idle_pwr = get_idle_power(gpu_name, clock)
                        except Exception:
                            idle_pwr = idle_power  # fallback to standard clock idle power
                        idle_energy = idle_pwr * idle_time
                        total_energy = decode_energy + idle_energy
                        if (min_total_energy is None) or (total_energy
                                                          < min_total_energy):
                            min_total_energy = total_energy
                            best_clock = clock
                    except Exception:
                        continue  # skip clocks where data is missing

                if best_clock is None:
                    raise ValueError(
                        "No valid clock found for DVFS optimization")
                return best_clock, min_total_energy

            for chunk_size in chunk_sizes:
                # --- Prefill energy ---
                num_chunks = int(np.ceil(request_length / chunk_size))
                prefill_energy = 0.0
                for i in range(num_chunks):
                    prefill_energy += get_energy(chunk_size)
                prefill_energies.append(prefill_energy)

                # --- Bounded decode energy ---
                # The bound time is the prefill time for this chunk size
                chunk_time = get_time_taken(chunk_size)
                decode_time = get_time_taken(decode_size)

                # print decode_time and chunk_time
                print(f"Decode time: {decode_time}, Chunk time: {chunk_time}")
                print(f"Decode size: {decode_size}, Chunk size: {chunk_size}")

                # Number of decode requests in this time window = num_chunks
                dvfs_clock, _ = get_dvfs_clock(chunk_size, decode_size)
                decode_energy = get_energy(decode_size, dvfs_clock)

                idle_time = max(chunk_time - decode_time, 0)
                idle_energy = idle_power * idle_time

                total_bounded_energy = decode_energy + idle_energy

                total_decode_energy = total_bounded_energy * num_chunks * 3

                bounded_energies.append(total_decode_energy)

                # --- Prefill + Decode energy ---
                prefill_plus_decode_energies.append(prefill_energy +
                                                    total_decode_energy)

                # --- Idle energy only (for all decode windows) ---
                total_idle_energy = idle_energy * num_chunks * 3
                idle_energies.append(total_idle_energy)

                # --- Decode-only energy (for all decode windows) ---
                total_decode_only_energy = decode_energy * num_chunks * 3
                decode_only_energies.append(total_decode_only_energy)

            # Plot
            plt.figure(figsize=(14, 8))
            plt.plot(chunk_sizes,
                     prefill_energies,
                     marker='o',
                     label='Prefill Energy (J)',
                     linewidth=3)
            plt.plot(
                chunk_sizes,
                bounded_energies,
                marker='s',
                label=f'Bounded Decode Energy (decode size={decode_size})',
                linewidth=3)
            plt.plot(chunk_sizes,
                     prefill_plus_decode_energies,
                     marker='^',
                     label='Prefill + Decode Energy (J)',
                     linewidth=3)
            plt.plot(chunk_sizes,
                     idle_energies,
                     marker='x',
                     label='Idle Energy Only (J)',
                     linestyle='--',
                     linewidth=1.5)
            plt.plot(chunk_sizes,
                     decode_only_energies,
                     marker='d',
                     label='Decode-Only Energy (J)',
                     linestyle='--',
                     linewidth=1.5)
            plt.xlabel('Chunk Size')
            plt.ylabel('Total Energy (J)')
            plt.title(
                f'Energy vs Chunk Size for {model_name} on {gpu_name} (Standard Clock)'
            )
            plt.ylim(bottom=0)

            # ylim top 1000

            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'{model_name}_{gpu_name}_energy_vs_chunksize.png',
                        bbox_inches='tight')
            plt.close()
