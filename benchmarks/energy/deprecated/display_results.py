# SPDX-License-Identifier: Apache-2.0
# dvfs_profile_b'<GPU_NAME>'_<MODEL_NAME>.csv

import os
import re

# matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Global variable for max batch size threshold in graphs
MAX_BATCH_SIZE = 4000  # Set this as needed

# Parse all files in directory with .csv and pattern match
# clock,batch_size,time_taken,energy_consumption,valid,total_ctx_len

# Ensure temp_images directory exists
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


def set_fig_size(plt):
    # Larger plot size
    plt.figure(figsize=(16, 8))


def save_fig(plt, model_name, gpu_name, mode, suffix):
    # y start from 0
    plt.ylim(0, None)
    plt.xlim(0, 2000)
    plt.savefig(os.path.join(TEMP_IMAGE_DIR,
                             f'[{mode}]{model_name}_{gpu_name}_{suffix}.png'),
                bbox_inches='tight')
    plt.close()


for file in os.listdir('.'):
    if file.endswith('.csv'):
        # Parse the file name to get the GPU name and model name with regex
        match = re.search(r'dvfs_profile_(.*)_(.*)_(.*).csv', file)
        if match:
            gpu_name = match.group(1)
            model_name = match.group(2)
            mode = match.group(3)
            print(f"GPU: {gpu_name}, Model: {model_name}, Mode: {mode}")
            df_orig = pd.read_csv(file)

            unique_batch_sizes = df_orig['batch_size'].unique()
            unique_batch_sizes.sort()

            # mJ to J
            # str to float
            # Show line of which "energy_consumption" is not float
            for i, row in df_orig.iterrows():
                if not isinstance(row['energy_consumption'], float):
                    print(f"Line {i}: {row['energy_consumption']}")

            df_orig['energy_consumption'] = df_orig[
                'energy_consumption'].astype(float)
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

            unique_bound_sizes = [None]
            if mode == "bounded":
                unique_bound_size = df_orig['bound_size'].unique()
                unique_bound_size.sort()
                unique_bound_sizes = unique_bound_size.tolist()

            for bound_size in unique_bound_sizes:
                # Remove "nobound" suffix for unbounded
                bound_suffix = ""
                if bound_size is not None:
                    df = df_orig[df_orig['bound_size'] == bound_size]
                    # Filter by MAX_BATCH_SIZE
                    df = df[df['batch_size'] <= MAX_BATCH_SIZE]
                    bound_suffix = f"bound_{bound_size}"
                else:
                    df = df_orig[df_orig['batch_size'] <= MAX_BATCH_SIZE]
                # 1. energy vs clock
                set_fig_size(plt)
                # plt.yscale('log')

                # if bound, get data from _one.csv version
                if mode == "bounded":
                    df_one = pd.read_csv(
                        f'dvfs_profile_{gpu_name}_{model_name}_one.csv')
                    # Filter by MAX_BATCH_SIZE
                    df_one = df_one[df_one['batch_size'] <= MAX_BATCH_SIZE]
                    # For each batch_size, plot energy_consumption vs clock in dashed line

                for batch_size in unique_batch_sizes:
                    batch_data = df[df['batch_size'] == batch_size]
                    # Remove rows where clock or energy_consumption is 0
                    batch_data = batch_data[(batch_data['clock'] != 0) & (
                        batch_data['energy_consumption'] != 0)]
                    # if empty, skip
                    if batch_data.empty:
                        continue
                    label = f'Batch Size: {batch_size}'
                    if bound_suffix:
                        label += f' {bound_suffix}'
                    plt.plot(batch_data['clock'],
                             batch_data['energy_consumption'],
                             label=label)
                    if mode == "bounded":
                        batch_data_one = df_one[df_one['batch_size'] ==
                                                batch_size]
                        # energy to J
                        batch_data_one = batch_data_one.copy()
                        batch_data_one['energy_consumption'] = batch_data_one[
                            'energy_consumption'] / 1000
                        # Remove rows where clock or energy_consumption is 0
                        batch_data_one = batch_data_one[
                            (batch_data_one['clock'] != 0)
                            & (batch_data_one['energy_consumption'] != 0)]
                        if not batch_data_one.empty:
                            plt.plot(
                                batch_data_one['clock'],
                                batch_data_one['energy_consumption'],
                                label=f'Batch Size: {batch_size} unbounded',
                                linestyle='--')

                max_clock = df['clock'].max()

                # Add a dot, and show for each batch_size the clock-frequency pair with the minimum energy and the maximum energy
                for batch_size in unique_batch_sizes:
                    batch_data = df[df['batch_size'] == batch_size]
                    # Remove rows where clock or energy_consumption is 0
                    batch_data = batch_data[(batch_data['clock'] != 0) & (
                        batch_data['energy_consumption'] != 0)]
                    # if empty, skip
                    if batch_data.empty:
                        continue
                    min_energy_freq = batch_data.loc[
                        batch_data['energy_consumption'].idxmin()]['clock']
                    max_energy_freq = batch_data.loc[
                        batch_data['energy_consumption'].idxmax()]['clock']
                    plt.scatter(min_energy_freq,
                                batch_data['energy_consumption'].min(),
                                color='red')
                    plt.scatter(max_energy_freq,
                                batch_data['energy_consumption'].max(),
                                color='blue')
                    # Add text to the plot. Show ratio of min to max energy
                    max_energy = batch_data['energy_consumption'].max()
                    max_clock_energy = batch_data[
                        batch_data['clock'] ==
                        max_clock]['energy_consumption'].max()
                    min_energy = batch_data['energy_consumption'].min()
                    plt.text(
                        min_energy_freq,
                        min_energy,
                        f'{max_energy / min_energy:.2f}, {max_clock_energy / min_energy:.2f}',
                        color='red')
                    # Show ratio of min to max clock energy

                plt.xlabel('Clock')
                plt.ylabel('Energy')
                plt.title(f'Energy vs Clock for {model_name} on {gpu_name}' +
                          (f' {bound_suffix}' if bound_suffix else ''))
                plt.legend(fontsize='small', loc='best', frameon=True)
                save_fig(
                    plt, model_name, gpu_name, mode,
                    f'energy_vs_clock{("_" + bound_suffix) if bound_suffix else ""}'
                )

                # 2. power vs clock
                set_fig_size(plt)
                # y-axis is log scale
                # plt.yscale('log')

                for batch_size in unique_batch_sizes:
                    batch_data = df[df['batch_size'] == batch_size]
                    # Remove rows where clock or power is 0
                    batch_data = batch_data[(batch_data['clock'] != 0)
                                            & (batch_data['power'] != 0)]
                    if batch_data.empty:
                        continue
                    label = f'Batch Size: {batch_size}'
                    if bound_suffix:
                        label += f' {bound_suffix}'
                    plt.plot(batch_data['clock'],
                             batch_data['power'],
                             label=label)
                plt.xlabel('Clock')
                plt.ylabel('Power')
                plt.title(f'Power vs Clock for {model_name} on {gpu_name}' +
                          (f' {bound_suffix}' if bound_suffix else ''))
                plt.legend(fontsize='small', loc='best', frameon=True)
                # save
                save_fig(
                    plt, model_name, gpu_name, mode,
                    f'power_vs_clock{("_" + bound_suffix) if bound_suffix else ""}'
                )

                # 3. energy-per-token vs clock
                set_fig_size(plt)
                # plt.yscale('log')

                for batch_size in unique_batch_sizes:
                    batch_data = df[df['batch_size'] == batch_size]
                    # Remove rows where clock or energy-per-token is 0
                    batch_data = batch_data[(batch_data['clock'] != 0) & (
                        batch_data['energy-per-token'] != 0)]
                    if batch_data.empty:
                        continue
                    label = f'Batch Size: {batch_size}'
                    if bound_suffix:
                        label += f' {bound_suffix}'
                    plt.plot(batch_data['clock'],
                             batch_data['energy-per-token'],
                             label=label)
                plt.xlabel('Clock')
                plt.ylabel('Energy-per-token')
                plt.title(
                    f'Energy-per-token vs Clock for {model_name} on {gpu_name}'
                    + (f' {bound_suffix}' if bound_suffix else ''))
                plt.legend(fontsize='small', loc='best', frameon=True)
                save_fig(plt, model_name, gpu_name, mode,
                         'energy-per-token_vs_clock')

                # 4. time_taken vs clock
                if mode != "bounded":
                    set_fig_size(plt)
                    for batch_size in unique_batch_sizes:
                        batch_data = df[df['batch_size'] == batch_size]
                        # Remove rows where clock or time_taken is 0
                        batch_data = batch_data[(batch_data['clock'] != 0) & (
                            batch_data['time_taken'] != 0)]
                        if batch_data.empty:
                            continue
                        label = f'Batch Size: {batch_size}'
                        if bound_suffix:
                            label += f' {bound_suffix}'
                        plt.plot(batch_data['clock'],
                                 batch_data['time_taken'],
                                 label=label)
                    plt.xlabel('Clock')
                    plt.ylabel('Time Taken')
                    plt.title(
                        f'Time Taken vs Clock for {model_name} on {gpu_name}' +
                        (f' {bound_suffix}' if bound_suffix else ''))
                    plt.legend(fontsize='small', loc='best', frameon=True)
                    save_fig(
                        plt, model_name, gpu_name, mode,
                        f'time_taken_vs_clock{("_" + bound_suffix) if bound_suffix else ""}'
                    )

            # --- NEW PLOT: Energy-per-token vs Batch Size, lines for each clock ---
            # Now, also plot time-per-token on a secondary y-axis in the same plot
            set_fig_size(plt)
            unique_clocks = df_orig['clock'].unique()
            unique_clocks = [clk for clk in unique_clocks if clk != 0]
            unique_clocks.sort()
            # For each clock, plot batch_size vs energy-per-token and time-per-token

            # If nvidia A100, do not override
            if "A100" not in gpu_name:
                unique_clocks = [1770]
            else:
                unique_clocks = [1050]

            fig, ax1 = plt.subplots(figsize=(16, 8))
            ax2 = ax1.twinx()

            for clock in unique_clocks:
                clock_data = df_orig[(df_orig['clock'] == clock) &
                                     (df_orig['batch_size'] <= MAX_BATCH_SIZE)]
                # Remove rows where energy-per-token or time-per-token is 0 or nan
                clock_data = clock_data[
                    (clock_data['energy-per-token'] != 0)
                    & (~clock_data['energy-per-token'].isna()) &
                    (clock_data['time-per-token'] != 0) &
                    (~clock_data['time-per-token'].isna())]
                if clock_data.empty:
                    continue
                # Sort by batch_size for line plot
                clock_data = clock_data.sort_values('batch_size')
                label = f'Clock: {clock}'
                # Plot energy-per-token
                ax1.plot(clock_data['batch_size'],
                         clock_data['energy-per-token'],
                         label=label,
                         marker='o',
                         markersize=4,
                         linewidth=2)
                # Plot time-per-token on secondary axis
                ax2.plot(clock_data['batch_size'],
                         clock_data['time-per-token'],
                         label=label + " (time/token)",
                         marker='s',
                         markersize=4,
                         linewidth=2,
                         linestyle='dashed',
                         alpha=0.7)
                # Find minimum energy-per-token for this clock
                min_energy = clock_data['energy-per-token'].min()
                # Add small dots and text for each point, with red multiplier if not min
                for x, y, t in zip(clock_data['batch_size'],
                                   clock_data['energy-per-token'],
                                   clock_data['time-per-token']):
                    ax1.scatter(x, y, color='red', s=12, zorder=5)
                    # Calculate how many times more energy than min
                    if min_energy > 0:
                        multiplier = y / min_energy
                    else:
                        multiplier = float('nan')
                    ax1.text(x,
                             y,
                             f"({multiplier:.2f}x)",
                             fontsize=7,
                             ha='left',
                             va='bottom',
                             rotation=0,
                             color='black',
                             alpha=0.7,
                             zorder=10)
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Energy-per-token (J/token)')
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=1e-4)
            ax1.set_xlim(0, MAX_BATCH_SIZE)
            ax1.set_title(
                f'Energy-per-token and Time-per-token vs Batch Size for {model_name} on {gpu_name}'
            )
            # Legend for both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2,
                       labels1 + labels2,
                       fontsize='small',
                       loc='best',
                       frameon=True)
            ax2.set_ylabel('Time-per-token (s/token)')
            ax2.set_yscale('log')
            ax2.set_ylim(bottom=1e-6)
            fig.tight_layout()
            plt.savefig(
                f'[{mode}]{model_name}_{gpu_name}_energy-per-token_and_time-per-token_vs_batchsize.png',
                bbox_inches='tight')
            plt.close()

            # TTFT vs TBT plot for a request of length 4096, simulating different chunk sizes
            import numpy as np

            # Parameters
            request_length = 4096
            num_pp = 4
            scheduling_delay_ms = 2  # per chunk, in ms
            scheduling_delay_s = scheduling_delay_ms / 1000.0

            if "A100" not in gpu_name:
                standard_clock = 1770
            else:
                standard_clock = 1050

            # chunk_sizes = [4096, 2048, 1024, 512, 256, 128, 64]
            chunk_sizes = [
                128, 256, 512, 1024, 1280, 1536, 1792, 2048, 3072, 4096
            ]

            # For each chunk size, compute:
            # - num_chunks = (request_length / chunk_size) * num_pp
            # - TTFT = (num_chunks - 1) * scheduling_delay + sum(time_taken for each chunk)
            # - TBT = time_taken for a single chunk (from profile)

            # We'll use the profiled time_taken for each batch_size (chunk_size)
            # If chunk_size not in df, interpolate/extrapolate

            # Get mapping: batch_size -> mean time_taken
            batch_time_map = df.groupby(
                'batch_size')['time_taken'].mean().to_dict()

            # Filter data to only use rows with the standard clock
            df_standard_clock = df[df['clock'] == standard_clock]
            # Recompute batch_time_map for standard clock only
            batch_time_map_std = df_standard_clock.groupby(
                'batch_size')['time_taken'].mean().to_dict()

            # For interpolation
            batch_sizes_sorted = np.array(sorted(batch_time_map_std.keys()))
            time_taken_sorted = np.array(
                [batch_time_map_std[bs] for bs in batch_sizes_sorted])

            def get_time_taken(bs, clock=None):
                # Only use data from standard clock
                if bs in batch_time_map_std:
                    return batch_time_map_std[bs]
                else:
                    # Interpolate/extrapolate
                    return float(
                        np.interp(bs,
                                  batch_sizes_sorted,
                                  time_taken_sorted,
                                  left=time_taken_sorted[0],
                                  right=time_taken_sorted[-1]))

            ttft_list = []
            tbt_list = []
            for chunk_size in chunk_sizes:
                num_chunks = int(np.ceil(request_length / chunk_size)) * num_pp
                tbt = get_time_taken(chunk_size)
                # TTFT: first chunk starts immediately, each subsequent chunk incurs scheduling delay
                # TTFT = (num_chunks - 1) * scheduling_delay + num_chunks * tbt
                ttft = (num_chunks - 1) * scheduling_delay_s + num_chunks * tbt
                ttft_list.append(ttft)
                tbt_list.append(tbt)

            # Get standard time_taken for 4096 tokens (full batch, no chunking)
            standard_ttft = get_time_taken(4096) * 4
            # Get standard time_taken for 16 tokens (smallest batch, for TBT reference)
            standard_tbt = get_time_taken(
                16) if 16 in batch_time_map else get_time_taken(
                    batch_sizes_sorted[0])

            # Plot
            fig, (ax_top, ax_bot) = plt.subplots(2,
                                                 1,
                                                 figsize=(10, 8),
                                                 sharex=True)

            # Top: chunk size vs TTFT
            ax_top.plot(chunk_sizes,
                        ttft_list,
                        marker='o',
                        label='Simulated TTFT')
            ax_top.axhline(standard_ttft,
                           color='r',
                           linestyle='--',
                           label='Standard TTFT (4096 tokens)')
            ax_top.set_ylabel('TTFT (s)')
            ax_top.set_title(
                f'TTFT vs Chunk Size (Request Length={request_length}, {num_pp} PP, {scheduling_delay_ms}ms sched delay)'
            )
            ax_top.set_ylim(bottom=0)
            ax_top.legend()
            ax_top.grid(True, which='both', linestyle='--', alpha=0.5)

            # Bottom: chunk size vs TBT
            ax_bot.plot(chunk_sizes,
                        tbt_list,
                        marker='o',
                        color='g',
                        label='TBT (per chunk)')
            ax_bot.axhline(standard_tbt,
                           color='orange',
                           linestyle='--',
                           label='Standard TBT (16 tokens)')
            ax_bot.set_xlabel('Chunk Size')
            ax_bot.set_ylabel('TBT (s)')
            ax_bot.set_ylim(bottom=0)
            ax_bot.legend()
            ax_bot.grid(True, which='both', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(
                f'[{mode}]{model_name}_{gpu_name}_ttft_vs_tbt_vs_chunksize.png',
                bbox_inches='tight')
            plt.close()

            # Plot time_taken vs clock w.r.t different batch sizes
            set_fig_size(plt)
            for batch_size in unique_batch_sizes:
                batch_data = df[df['batch_size'] == batch_size]
                # Remove rows where clock or time_taken is 0
                batch_data = batch_data[(batch_data['clock'] != 0)
                                        & (batch_data['time_taken'] != 0)]
                if batch_data.empty:
                    continue
                plt.plot(batch_data['clock'],
                         batch_data['time_taken'],
                         label=f'Batch Size: {batch_size}')

            plt.xlabel('Clock')
            plt.ylabel('Time Taken')
            plt.title(f'Time Taken vs Clock for {model_name} on {gpu_name}' +
                      (f' {bound_suffix}' if bound_suffix else ''))
            plt.legend(fontsize='small', loc='best', frameon=True)
            save_fig(
                plt, model_name, gpu_name, mode,
                f'time_taken_vs_clock{("_" + bound_suffix) if bound_suffix else ""}'
            )
