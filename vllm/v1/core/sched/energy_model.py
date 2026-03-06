# SPDX-License-Identifier: Apache-2.0
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class EnergySimulatorResult:
    # Grand total energy
    grand_total_energy: float
    # Estimated decode energy
    est_decode_energy: float
    # Prefill decode energy (P+D interfere)
    prefill_decode_energy_full_pipeline_repeat: float
    # Decode energy info
    decode_energy_info: Dict[int, Tuple[float, float, float, float]]
    # time_taken (prefill + decode)
    time_taken: float


class EnergySimulator:
    """Energy configuration simulator based on profiling data."""

    def __init__(
        self,
        profiling_csv_path: str,
        num_pp: int,
        num_tp: int,
        gpu_name: str | None = None,
        model_name: str | None = None,
    ):
        """Initialize with profiling data."""
        # TODO: profile
        
        # If A6000, minimum clock is 800, else, 500
        self.min_clock = 800 if "A6000" in gpu_name else 500
        
        self.scheduling_overhead = 0.012 # Conservative estimate.
        self.scheduling_overhead_decode_only = 0.003 * num_pp

        self.profiling_csv_path = profiling_csv_path
        self.num_pp = num_pp
        self.num_tp = num_tp
        self.num_gpus = num_pp * num_tp
        # Extract names from filename if not provided
        if gpu_name is None or model_name is None:
            gpu_name, model_name = self._extract_names_from_filename(profiling_csv_path)

        self.gpu_name = gpu_name
        self.model_name = model_name
        self.standard_model_prefill_size = 256
        self.dvfs_delay = 0.01

        self._load_profiling_data()

        self.profiled_batch_sizes = self.df["batch_size"].unique()
        # sort
        self.profiled_batch_sizes.sort()
        print(f"Profiled batch sizes: {self.profiled_batch_sizes}")
        # self.standard_clock = 1350 if "A100" in self.gpu_name else 1830
        self.standard_clock = 1410 if "100" in self.gpu_name else 1800

        self.available_chunk_sizes = [
            128,
            256,
            320,
            384,
            448,
            512,
        ]
            # 640,
            # 768,
            # 896,
            # 1024,
            # 1280,
            # 1536,
            # 1792,
            # 2048,
            # 3072,
            # 4096,

        # Remove batch sizes that are not in the df
        self.available_chunk_sizes = [
            b for b in self.available_chunk_sizes if b in self.df["batch_size"].unique()
        ]
        self.available_clocks = [c for c in self.df["clock"].unique() if c >= self.min_clock]
         
        # Set default chunk size and clock
        if "A100" in self.gpu_name:
            self.default_chunk_size = 128
            self.default_clock_high = 1410
            self.default_clock = 1050
        else:
            self.default_chunk_size = 128
            self.default_clock_high = 1800
            self.default_clock = 1260

    def _load_profiling_data(self):
        """Load and preprocess profiling CSV."""
        # print the profiling_csv_path
        print(f"Profiling CSV path: {self.profiling_csv_path}")
        if not os.path.exists(self.profiling_csv_path):
            # dvfs_profile_NVIDIA RTX A6000_Qwen_Qwen2.5-32B_tp1_pp4_one.csv
            raise FileNotFoundError(f"CSV not found: {self.profiling_csv_path}")

        self.df = pd.read_csv(self.profiling_csv_path)
        
        # Use total_ctx_len over 100
        # self.df = self.df[self.df["total_ctx_len"] > 100]
        
        self.df["energy_consumption"] = self.df["energy_consumption"] / 1000  # mJ to J
        self.df["energy-per-token"] = (
            self.df["energy_consumption"] / self.df["batch_size"]
        )
        self.df["power"] = self.df["energy_consumption"] / self.df["time_taken"]
        self.df["time-per-token"] = self.df["time_taken"] / self.df["batch_size"]

        self._create_lookup_dicts()

    def _create_lookup_dicts(self):
        """Create lookup dictionaries for fast access."""
        self.batch_time_map = {}
        self.batch_energy_map = {}
        
        for clock in self.df["clock"].unique():
            if clock < self.min_clock:
                continue
            df_clock = self.df[self.df["clock"] == clock]
            
            # For energy: use max values
            self.batch_energy_map[clock] = (
                df_clock.groupby("batch_size")["energy_consumption"].max().to_dict()
            )
            
            # For time: store min/max context length info for interpolation
            self.batch_time_map[clock] = {}
            for batch_size in df_clock["batch_size"].unique():
                df_batch = df_clock[df_clock["batch_size"] == batch_size]
                
                # Get min and max context length entries
                min_ctx_row = df_batch.loc[df_batch["total_ctx_len"].idxmin()]
                max_ctx_row = df_batch.loc[df_batch["total_ctx_len"].idxmax()]
                
                self.batch_time_map[clock][batch_size] = {
                    'min_ctx_len': min_ctx_row["total_ctx_len"],
                    'max_ctx_len': max_ctx_row["total_ctx_len"],
                    'min_time': min_ctx_row["time_taken"],
                    'max_time': max_ctx_row["time_taken"]
                }
        # # print all chunk x clock combinations 
        # for clock in self.batch_time_map:
        #     for chunk in self.batch_time_map[clock]:
        #         logger.info(f"Chunk: {chunk}, Clock: {clock}, Time: {self.batch_time_map[clock][chunk]['min_time']}, Energy: {self.batch_energy_map[clock][chunk]}")

    def _interpolate_time_for_ctx_len(self, time_info: Dict, total_ctx_len: int | None) -> float:
        """Interpolate time based on context length.
        
        Args:
            time_info: Dictionary with 'min_ctx_len', 'max_ctx_len', 'min_time', 'max_time'
            total_ctx_len: Optional total context length. If None, uses max context length.
        
        Returns:
            Interpolated time value
        """
        if total_ctx_len is None:
            # Use max context length value
            return time_info['max_time']
        
        # Extract values
        min_ctx = time_info['min_ctx_len']
        max_ctx = time_info['max_ctx_len']
        min_time = time_info['min_time']
        max_time = time_info['max_time']
        
        # Clip total_ctx_len to min/max range
        clipped_ctx_len = max(min_ctx, min(max_ctx, total_ctx_len))
        
        # Linear interpolation
        if max_ctx == min_ctx:
            return max_time
        
        alpha = (clipped_ctx_len - min_ctx) / (max_ctx - min_ctx)
        return min_time + alpha * (max_time - min_time)

    def get_energy(self, batch_size: int, clock: int) -> float:
        """Get energy consumption for batch_size at clock frequency."""
        if clock not in self.batch_energy_map:
            raise ValueError(f"No data for clock {clock}")

        if batch_size <= 512:
            actual_batch_size = self.find_smallest_batch_size_ge(batch_size, clock)
            return self.batch_energy_map[clock][actual_batch_size]
        else:
            # Linear interpolation for larger batch sizes
            lower, upper = self._find_surrounding_batch_sizes(batch_size, clock)
            if lower == upper:
                return self.batch_energy_map[clock][lower]
            energy_lower = self.batch_energy_map[clock][lower]
            energy_upper = self.batch_energy_map[clock][upper]
            alpha = (batch_size - lower) / (upper - lower)
            return energy_lower + alpha * (energy_upper - energy_lower)

    def get_time_taken(self, batch_size: int, clock: int, total_ctx_len: int | None = None) -> float:
        """Get time taken for batch_size at clock frequency.
        
        Args:
            batch_size: The batch size
            clock: The clock frequency
            total_ctx_len: Optional total context length. If None, uses max context length.
                          If provided, interpolates between min and max context lengths.
        """
        if clock not in self.batch_time_map:
            raise ValueError(f"No data for clock {clock}")

        if batch_size <= 512:
            actual_batch_size = self.find_smallest_batch_size_ge(batch_size, clock)
            time_info = self.batch_time_map[clock][actual_batch_size]
            return float(self._interpolate_time_for_ctx_len(time_info, total_ctx_len))
        else:
            # Linear interpolation for larger batch sizes
            lower, upper = self._find_surrounding_batch_sizes(batch_size, clock)
            if lower == upper:
                time_info = self.batch_time_map[clock][lower]
                return float(self._interpolate_time_for_ctx_len(time_info, total_ctx_len))
            
            # Interpolate across both batch size and context length
            time_lower_info = self.batch_time_map[clock][lower]
            time_upper_info = self.batch_time_map[clock][upper]
            
            # Get interpolated times for both lower and upper batch sizes
            time_lower = float(self._interpolate_time_for_ctx_len(time_lower_info, total_ctx_len))
            time_upper = float(self._interpolate_time_for_ctx_len(time_upper_info, total_ctx_len))
            
            # Interpolate between batch sizes
            alpha = (batch_size - lower) / (upper - lower)
            return time_lower + alpha * (time_upper - time_lower)

    def find_smallest_batch_size_ge(self, target_batch_size: int, clock: int) -> int:
        """Find smallest available batch size >= target for given clock."""
        if clock not in self.batch_time_map:
            raise ValueError(f"No data for clock {clock}")

        available_batch_sizes = sorted(self.profiled_batch_sizes)
        for batch_size in available_batch_sizes:
            if batch_size >= target_batch_size:
                return batch_size
        raise ValueError(f"No batch size >= {target_batch_size} found")

    def _find_surrounding_batch_sizes(self, target_batch_size: int, clock: int) -> Tuple[int, int]:
        """Find lower and upper batch sizes for interpolation."""
        available_batch_sizes = sorted(self.profiled_batch_sizes)
        lower = available_batch_sizes[0]
        upper = available_batch_sizes[-1]
        
        for i, batch_size in enumerate(available_batch_sizes):
            if batch_size >= target_batch_size:
                upper = batch_size
                lower = available_batch_sizes[i-1] if i > 0 else batch_size
                break
        
        return lower, upper


if __name__ == "__main__":

    OFFLINE_PROFILE_DIR = "/workspace/disagg/energy-inf-v1-disagg/benchmarks/energy"

    def _strip_benchmarks_energy(path):
        cwd = os.getcwd()
        if OFFLINE_PROFILE_DIR in cwd:
            print(f"Stripping benchmarks energy from path: {path}")
            return path.replace(OFFLINE_PROFILE_DIR, "")
        return path

    # GPU_NAME = "NVIDIA RTX A6000"
    # MODEL_NAME = "Qwen/Qwen2.5-32B"
    GPU_NAME = "NVIDIA A100-SXM4-80GB"
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
    MODEL_NAME = MODEL_NAME.replace("/", "_")

    NUM_PP = 4
    NUM_TP = 2

    PROFILE_FILE = _strip_benchmarks_energy(
        f"{OFFLINE_PROFILE_DIR}/offline_profile_results/dvfs_profile_{GPU_NAME}_{MODEL_NAME}_tp{NUM_TP}_pp{NUM_PP}_one.csv"
    )

    print(f"Profile file: {PROFILE_FILE}")

    energy_simulator = EnergySimulator(
        profiling_csv_path=PROFILE_FILE,
        num_pp=NUM_PP,
        num_tp=NUM_TP,
        gpu_name=GPU_NAME,
        model_name=MODEL_NAME,
    )
    
    ARBITRARY_PREFILL_SIZE = 4000
    
    chunk_sizes = energy_simulator.available_chunk_sizes
    clocks = energy_simulator.available_clocks
    
    data = []
    
    tbt_slo = 0.2
    ttft_slo = 1
    
    for chunk_size in chunk_sizes:
        for clock in clocks:
            per_chunk_energy = energy_simulator.get_energy(chunk_size, clock)
            per_chunk_latency = energy_simulator.get_time_taken(chunk_size, clock)
            num_chunks_floor = math.floor(ARBITRARY_PREFILL_SIZE / chunk_size) * NUM_PP
            num_ttft_chunks_floor = math.floor(ARBITRARY_PREFILL_SIZE / chunk_size) + (NUM_PP - 1)
            
            
            if chunk_size >= ARBITRARY_PREFILL_SIZE:
                # Single chunk can handle entire request
                per_chunk_energy = energy_simulator.get_energy(ARBITRARY_PREFILL_SIZE, clock)
                per_chunk_latency = energy_simulator.get_time_taken(ARBITRARY_PREFILL_SIZE, clock)
                
                tbt = (per_chunk_latency + energy_simulator.scheduling_overhead) * NUM_PP
                ttft = (per_chunk_latency + energy_simulator.scheduling_overhead) * NUM_PP
                energy = per_chunk_energy * NUM_PP
                
                ttft_adhering = ttft <= ttft_slo
                tbt_adhering = tbt <= tbt_slo
                
                tbt_adhering = True
                
                data.append((chunk_size, clock, tbt, ttft, energy, ttft_adhering, tbt_adhering))
            else:
                # Multiple chunks needed
                num_leftover_tokens = ARBITRARY_PREFILL_SIZE % chunk_size
                leftover_chunk_energy = energy_simulator.get_energy(num_leftover_tokens, clock)
                leftover_chunk_latency = energy_simulator.get_time_taken(num_leftover_tokens, clock)
                
                tbt = (per_chunk_latency + energy_simulator.scheduling_overhead) * NUM_PP
                ttft = num_ttft_chunks_floor * (per_chunk_latency + energy_simulator.scheduling_overhead) + leftover_chunk_latency
                energy = per_chunk_energy * num_chunks_floor + leftover_chunk_energy * NUM_PP
                
                print(f"Chunk_size: {chunk_size} + {num_leftover_tokens}, num_ttft_chunks_floor: {num_ttft_chunks_floor}, num_chunks_floor: {num_chunks_floor}")
                
                ttft_adhering = ttft <= ttft_slo
                tbt_adhering = tbt <= tbt_slo
                
                tbt_adhering = True
                
                data.append((chunk_size, clock, tbt, ttft, energy, ttft_adhering, tbt_adhering))

    df = pd.DataFrame(data, columns=["chunk_size", "clock", "tbt", "ttft", "energy", "ttft_adhering", "tbt_adhering"])
    
    # Create heatmap
    # Pivot the dataframe to create a 2D matrix for the heatmap
    pivot_energy = df.pivot(index="clock", columns="chunk_size", values="energy")
    pivot_ttft = df.pivot(index="clock", columns="chunk_size", values="ttft_adhering")
    pivot_tbt = df.pivot(index="clock", columns="chunk_size", values="tbt_adhering")
    
    # Find minimum energy configuration that satisfies both SLOs
    adhering_df = df[(df["ttft_adhering"] == True) & (df["tbt_adhering"] == True)]
    if not adhering_df.empty:
        min_energy_row = adhering_df.loc[adhering_df["energy"].idxmin()]
        min_chunk = min_energy_row["chunk_size"]
        min_clock = min_energy_row["clock"]
    else:
        min_chunk = None
        min_clock = None
    
    # Create masked energy data: set violations to NaN
    pivot_energy_masked = pivot_energy.copy()
    for i in range(len(pivot_energy.index)):
        for j in range(len(pivot_energy.columns)):
            if not pivot_ttft.iloc[i, j] or not pivot_tbt.iloc[i, j]:
                pivot_energy_masked.iloc[i, j] = np.nan
    
    # Create the plot (more compact for academic paper)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create custom colormap with grey for NaN values
    from matplotlib.colors import ListedColormap
    viridis = plt.cm.viridis_r
    newcolors = viridis(np.linspace(0, 1, 256))
    grey = np.array([0.4, 0.4, 0.4, 1])
    cmap_with_grey = ListedColormap(newcolors)
    cmap_with_grey.set_bad(color=grey)
    
    # Create heatmap (use reversed colormap so brighter = less energy)
    # vmin and vmax only span valid (non-NaN) points
    valid_energies = pivot_energy_masked.values[~np.isnan(pivot_energy_masked.values)]
    if len(valid_energies) > 0:
        vmin, vmax = valid_energies.min(), valid_energies.max()
    else:
        vmin, vmax = None, None
    
    im = ax.imshow(pivot_energy_masked.values, cmap=cmap_with_grey, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_energy.columns)))
    ax.set_yticks(np.arange(len(pivot_energy.index)))
    ax.set_xticklabels(pivot_energy.columns, fontsize=9)
    ax.set_yticklabels(pivot_energy.index, fontsize=9)
    
    # Reverse y-axis so largest clock is at top
    ax.invert_yaxis()
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Find minimum energy for each chunk size (column) - only among valid (non-violated) configs
    min_energy_per_chunk = {}
    for j, chunk_size in enumerate(pivot_energy.columns):
        # Only consider non-violated configurations
        valid_mask = pivot_ttft[chunk_size] & pivot_tbt[chunk_size]
        if valid_mask.any():
            valid_energies = pivot_energy[chunk_size][valid_mask]
            min_energy_idx = valid_energies.idxmin()
            min_energy_per_chunk[j] = list(pivot_energy.index).index(min_energy_idx)
        else:
            min_energy_per_chunk[j] = None
    
    # Add markers
    for i, clock in enumerate(pivot_energy.index):
        for j, chunk_size in enumerate(pivot_energy.columns):
            # Get the adherence values
            ttft_adhering = pivot_ttft.iloc[i, j]
            tbt_adhering = pivot_tbt.iloc[i, j]
            energy_value = pivot_energy.iloc[i, j]
            
            # Add markers based on violations (black color)
            if not ttft_adhering:
                ax.scatter(j, i, marker='x', s=150, c='black', linewidths=2.5, zorder=4, alpha=0.7)
            if not tbt_adhering:
                ax.scatter(j, i, marker='s', s=150, facecolors='none', 
                          edgecolors='black', linewidths=2, zorder=4, alpha=0.7)
            
            # Only add energy dots and stars for valid configurations
            if ttft_adhering and tbt_adhering:
                # Add dot for minimum energy in this column
                # if min_energy_per_chunk[j] == i:
                #     ax.scatter(j, i, marker='o', s=80, c='white', edgecolors='black', linewidths=1.5, zorder=3)
                
                # Add star for minimum energy with full adherence
                if min_chunk is not None and chunk_size == min_chunk and clock == min_clock:
                    ax.scatter(j, i, marker='*', s=300, c='gold', edgecolors='black', linewidths=1.5, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Energy (J)', rotation=270, labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Labels and title (more compact for academic paper)
    ax.set_xlabel('Chunk Size', fontsize=11)
    ax.set_ylabel('Clock Frequency (MHz)', fontsize=11)
    ax.set_title(f'Energy Consumption: {MODEL_NAME.split("_")[-1]} (TP={NUM_TP}, PP={NUM_PP})', fontsize=11, pad=10)
    
    # Add legend (more compact)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markeredgecolor='black', markersize=10, label='Global Min'),
        # Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
        #        markeredgecolor='black', markersize=7, label='Column Min'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', 
               markeredgecolor='black', markersize=8, label='TTFT Violation'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', 
               markeredgecolor='black', markersize=8, label='TBT Violation')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.22, 0.5), 
             fontsize=9, frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save the figure
    output_path = 'energy_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")
    
    # Create a new figure with 4 subplots using raw profiling data
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Use the raw profiling data from energy_simulator.df
    profile_df = energy_simulator.df
    
    # Get unique batch sizes and clocks for plotting
    unique_batches = sorted(profile_df['batch_size'].unique())
    unique_clocks = sorted(profile_df['clock'].unique())
    
    # Use a colormap for different lines
    batch_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_batches)))
    clock_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clocks)))

    # Filter the profile_df to only rows with max context length
    max_ctx_len = profile_df['total_ctx_len'].max()
    max_ctx_df = profile_df[profile_df['total_ctx_len'] == max_ctx_len]

    # Update unique_batches and unique_clocks for this context
    unique_batches = sorted(max_ctx_df['batch_size'].unique())
    unique_clocks = sorted(max_ctx_df['clock'].unique())

    # 1) Energy (y) vs Frequency (x) w.r.t different batch sizes (max context length only)
    for idx, batch_size in enumerate(unique_batches):
        batch_df = max_ctx_df[max_ctx_df['batch_size'] == batch_size].sort_values('clock')
        ax1.plot(batch_df['clock'], batch_df['energy_consumption'], marker='o',
                 label=f'Batch={batch_size}', color=batch_colors[idx], linewidth=2)
    ax1.set_xlabel('Clock Frequency (MHz)', fontsize=11)
    ax1.set_ylabel('Energy (J)', fontsize=11)
    ax1.set_title('Energy vs Frequency (max ctx)', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2) Energy (y) vs Batch Size (x) w.r.t different frequencies (max context length only)
    for idx, clock in enumerate(unique_clocks):
        clock_df = max_ctx_df[max_ctx_df['clock'] == clock].sort_values('batch_size')
        ax2.plot(clock_df['batch_size'], clock_df['energy_consumption'], marker='s',
                 label=f'Clock={clock}MHz', color=clock_colors[idx], linewidth=2)
    ax2.set_xlabel('Batch Size', fontsize=11)
    ax2.set_ylabel('Energy (J)', fontsize=11)
    ax2.set_title('Energy vs Batch Size (max ctx)', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) Latency (y) vs Frequency (x) w.r.t different batch sizes (max context length only)
    for idx, batch_size in enumerate(unique_batches):
        batch_df = max_ctx_df[max_ctx_df['batch_size'] == batch_size].sort_values('clock')
        ax3.plot(batch_df['clock'], batch_df['time_taken'], marker='o',
                 label=f'Batch={batch_size}', color=batch_colors[idx], linewidth=2)
    ax3.set_xlabel('Clock Frequency (MHz)', fontsize=11)
    ax3.set_ylabel('Latency (s)', fontsize=11)
    ax3.set_title('Latency vs Frequency (max ctx)', fontsize=12, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4) Latency (y) vs Batch Size (x) w.r.t different frequencies (max context length only)
    for idx, clock in enumerate(unique_clocks):
        clock_df = max_ctx_df[max_ctx_df['clock'] == clock].sort_values('batch_size')
        ax4.plot(clock_df['batch_size'], clock_df['time_taken'], marker='s',
                 label=f'Clock={clock}MHz', color=clock_colors[idx], linewidth=2)
    ax4.set_xlabel('Batch Size', fontsize=11)
    ax4.set_ylabel('Latency (s)', fontsize=11)
    ax4.set_title('Latency vs Batch Size (max ctx)', fontsize=12, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    fig2.suptitle(f'Energy and Latency Analysis - {MODEL_NAME} (TP={NUM_TP}, PP={NUM_PP})', 
                  fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save the multi-plot figure
    output_path2 = 'energy_latency_analysis.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to: {output_path2}")
    

    NUM_RUNNING_REQS = 64
    data_s2 = []
    for num_microbatches in range(1, NUM_PP + 1):
        for clock in clocks:
            # Get per-microbatch energy and latency
            num_reqs_per_microbatch = math.ceil(NUM_RUNNING_REQS / num_microbatches)
            # TBT is bound to individual microbatch
            tbt = (energy_simulator.get_time_taken(num_reqs_per_microbatch, clock) + energy_simulator.scheduling_overhead) * NUM_PP
            # Energy is bound to all microbatches
            energy = energy_simulator.get_energy(num_reqs_per_microbatch, clock) * num_microbatches
            
            data_s2.append((num_microbatches, clock, tbt, energy))    
    df_s2 = pd.DataFrame(data_s2, columns=["num_microbatches", "clock", "tbt", "energy"])

    # Create plots for Energy and Latency vs Clock for different microbatches
    fig3, (ax_energy, ax_latency) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique microbatch counts and prepare colors
    unique_microbatches = sorted(df_s2['num_microbatches'].unique())
    microbatch_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_microbatches)))
    
    # Plot A: Energy (y) vs Clock (x) w.r.t each # microbatch
    for idx, num_mb in enumerate(unique_microbatches):
        mb_df = df_s2[df_s2['num_microbatches'] == num_mb].sort_values('clock')
        ax_energy.plot(mb_df['clock'], mb_df['energy'], marker='o', 
                      label=f'{num_mb} microbatch{"es" if num_mb > 1 else ""}', 
                      color=microbatch_colors[idx], linewidth=2, markersize=8)
    ax_energy.set_xlabel('Clock Frequency (MHz)', fontsize=12)
    ax_energy.set_ylabel('Energy (J)', fontsize=12)
    ax_energy.set_title('Energy vs Clock Frequency', fontsize=13, fontweight='bold')
    ax_energy.legend(fontsize=10, loc='best')
    ax_energy.grid(True, alpha=0.3)
    
    # Plot B: Latency (y) vs Clock (x) w.r.t each # microbatch
    for idx, num_mb in enumerate(unique_microbatches):
        mb_df = df_s2[df_s2['num_microbatches'] == num_mb].sort_values('clock')
        ax_latency.plot(mb_df['clock'], mb_df['tbt'], marker='s', 
                       label=f'{num_mb} microbatch{"es" if num_mb > 1 else ""}', 
                       color=microbatch_colors[idx], linewidth=2, markersize=8)
    ax_latency.set_xlabel('Clock Frequency (MHz)', fontsize=12)
    ax_latency.set_ylabel('Latency (s)', fontsize=12)
    ax_latency.set_title('Latency vs Clock Frequency', fontsize=13, fontweight='bold')
    ax_latency.legend(fontsize=10, loc='best')
    ax_latency.grid(True, alpha=0.3)
    
    # Add overall title
    fig3.suptitle(f'Energy and Latency vs Clock - {MODEL_NAME} (TP={NUM_TP}, PP={NUM_PP}, Running Reqs={NUM_RUNNING_REQS})', 
                  fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the figure
    output_path3 = 'microbatch_energy_latency_vs_clock.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Microbatch analysis plots saved to: {output_path3}")

    