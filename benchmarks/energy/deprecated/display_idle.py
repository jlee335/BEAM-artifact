# SPDX-License-Identifier: Apache-2.0
# Display idle power vs clock for all *_idle.csv files in the current directory

import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# Only process files with _idle.csv suffix
for file in os.listdir('.'):
    if file.endswith('_idle.csv'):
        # Parse the file name to get the GPU name
        # Example: dvfs_profile_NVIDIA RTX A6000_idle.csv
        match = re.match(r'dvfs_profile_([^_]+(?: [^_]+)*)_idle\.csv', file)
        if not match:
            continue
        gpu_name = match.group(1).strip()
        # Only allow A6000 and A100 in GPU name, else continue
        print(f"Processing idle power for GPU: {gpu_name} from file: {file}")
        df = pd.read_csv(file)

        # Convert mJ to J for energy_consumption if present
        if 'energy_consumption' in df.columns:
            df['energy_consumption'] = df['energy_consumption'] / 1000

        # Compute power as energy_consumption / time_taken
        if 'energy_consumption' in df.columns and 'time_taken' in df.columns:
            df['power'] = df['energy_consumption'] / df['time_taken']
        else:
            print(
                f"File {file} missing required columns for power calculation, skipping."
            )
            continue

        # Group by clock and take mean power for each clock
        power_by_clock = df.groupby('clock')['power'].mean().reset_index()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(power_by_clock['clock'],
                 power_by_clock['power'],
                 marker='o',
                 linewidth=2,
                 label='Idle Power (W)')
        plt.xlabel('Clock Frequency (MHz)')
        plt.ylabel('Idle Power (Watts)')
        plt.title(f'Idle Power vs Clock for {gpu_name}')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{gpu_name}_idle_power_vs_clock.png', bbox_inches='tight')
        plt.close()
