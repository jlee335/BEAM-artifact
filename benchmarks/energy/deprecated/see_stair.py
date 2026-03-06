# SPDX-License-Identifier: Apache-2.0
# Choose 1770
# Plot batch_size (x) vs time_taken (y) with much larger dots and bigger figure

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    clocks = [930, 1530, 2070]
    max_chunk_size = 2048
    df = pd.read_csv(
        "dvfs_profile_NVIDIA RTX A6000_RedHatAI_Meta-Llama-3.1-70B-Instruct-quantized.w8a8_tp1_pp4_one.csv"
    )
    for clk in clocks:
        plt.plot(df[(df['clock'] == clk)
                    & (df['batch_size'] <= max_chunk_size)]['batch_size'],
                 df[(df['clock'] == clk)
                    & (df['batch_size'] <= max_chunk_size)]['time_taken'],
                 marker='o',
                 linestyle='-',
                 markersize=8,
                 label=f"{clk} MHz")
    plt.figure(figsize=(12, 8))
    for clk in clocks:
        plt.plot(df[(df['clock'] == clk)
                    & (df['batch_size'] <= max_chunk_size)]['batch_size'],
                 df[(df['clock'] == clk)
                    & (df['batch_size'] <= max_chunk_size)]['time_taken'],
                 marker='o',
                 linestyle='-',
                 markersize=8,
                 label=f"{clk} MHz")
    plt.xlabel("Batch Size", fontsize=18)
    plt.ylabel("Time Taken (s)", fontsize=18)
    plt.title("Batch Size vs Time Taken at Different Clocks", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, None)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("stair.png")
