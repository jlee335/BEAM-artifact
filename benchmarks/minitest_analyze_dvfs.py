#!/usr/bin/env python3
"""
Analyze compute characteristics of PyTorch linear kernel under different GPU clocks.
Based on LLAMA3-70B dimensions with configurable batch size.
"""

import os
import time
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import pynvml

# LLAMA3-70B typical dimensions (per-layer)
# Hidden size: 8192, Intermediate size: 28672
# HIDDEN_DIM = 8192
# INTERMEDIATE_DIM = 28672

# Llama-3 8b dimensions
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 14336

# GPU Specifications for A6000
A6000_SPECS = {
    "peak_fp16_tflops": 154.83,  # TFlops for FP16
    "memory_bandwidth_gb_s": 768,  # GB/s
    "name": "NVIDIA RTX A6000"
}


class DVFSProfiler:
    """Profile PyTorch operations under different GPU clock frequencies."""
    
    def __init__(self, output_dir: str = "dvfs_profile_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NVML
        pynvml.nvmlInit()
        self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get GPU info
        self.gpu_name = pynvml.nvmlDeviceGetName(self.device_handle)
        print(f"GPU: {self.gpu_name}")
        
        # Get available clocks
        self.available_clocks = self._get_available_clocks()
        print(f"Available GPU clocks (MHz): {self.available_clocks}")
        
        # Store original clock
        self.original_clock = pynvml.nvmlDeviceGetApplicationsClock(
            self.device_handle, pynvml.NVML_CLOCK_GRAPHICS
        )
        
    def _get_available_clocks(self) -> List[int]:
        """Get available GPU clock frequencies from NVML."""
        try:
            # Get supported graphics clocks for the default memory clock
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.device_handle)
            mem_clock = mem_clocks[0]  # Use highest memory clock
            
            graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                self.device_handle, mem_clock
            )
            
            # Select 5 evenly spaced clocks
            if len(graphics_clocks) <= 5:
                return sorted(graphics_clocks)
            else:
                indices = np.linspace(0, len(graphics_clocks) - 1, 5, dtype=int)
                return sorted([graphics_clocks[i] for i in indices])
        except Exception as e:
            print(f"Warning: Could not get supported clocks: {e}")
            print("Using default clock range estimation")
            # Fallback: estimate clock range
            try:
                max_clock = pynvml.nvmlDeviceGetMaxClockInfo(
                    self.device_handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                min_clock = int(max_clock * 0.3)  # Estimate minimum
                return list(range(min_clock, max_clock + 1, (max_clock - min_clock) // 9))
            except:
                # Further fallback for A6000
                return [210, 450, 690, 930, 1170, 1410, 1650, 1890, 2100, 2130]
    
    def set_gpu_clock(self, clock_mhz: int):
        """Set GPU clock frequency."""
        try:
            # Set both graphics and memory to application clocks
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.device_handle)
            mem_clock = mem_clocks[0]
            
            pynvml.nvmlDeviceSetApplicationsClocks(
                self.device_handle, mem_clock, clock_mhz
            )
            time.sleep(0.5)  # Wait for clock to stabilize
            print(f"Set GPU clock to {clock_mhz} MHz")
        except pynvml.NVMLError as e:
            print(f"Warning: Could not set clock to {clock_mhz} MHz: {e}")
            print("You may need to run with sudo or enable persistence mode")
    
    def reset_gpu_clock(self):
        """Reset GPU clock to default."""
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(self.device_handle)
            print("Reset GPU clock to default")
        except Exception as e:
            print(f"Warning: Could not reset clock: {e}")
    
    def measure_energy_and_latency(
        self, 
        operation_fn, 
        num_iterations: int,
        warmup_iterations: int = 10
    ) -> Tuple[float, float]:
        """
        Measure average latency and total energy for an operation.
        
        Returns:
            (avg_latency_ms, total_energy_joules)
        """
        device = torch.device("cuda:0")
        
        # Warmup
        for _ in range(warmup_iterations):
            operation_fn()
            torch.cuda.synchronize()
        
        # Get initial energy
        try:
            energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.device_handle)
        except:
            # Fallback: use power measurement
            energy_start = None
        
        # Measure latency
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            operation_fn()
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Get final energy
        try:
            energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.device_handle)
            total_energy_mj = energy_end - energy_start
            total_energy_j = total_energy_mj / 1000.0  # Convert mJ to J
        except:
            # Fallback: estimate from power
            elapsed_time = end_time - start_time
            power_w = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0
            total_energy_j = power_w * elapsed_time
        
        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        
        return avg_latency_ms, total_energy_j
    
    def profile_linear_kernel(
        self, 
        batch_sizes: List[int],
        num_clocks: int = 10
    ):
        """Profile linear kernel across batch sizes and clock frequencies."""
        
        csv_path = self.output_dir / "profile_results.csv"
        
        # Check if CSV exists
        if csv_path.exists():
            print(f"Profile results already exist at {csv_path}")
            print("Skipping profiling stage. Delete the file to re-run.")
            return
        
        device = torch.device("cuda:0")
        
        # Create linear layer (similar to LLAMA3-70B)
        linear = torch.nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False).to(device).half()
        
        results = []
        
        # Select clocks to use
        clocks_to_test = self.available_clocks[:num_clocks]
        
        for batch_size in batch_sizes:
            print(f"\n=== Batch Size: {batch_size} ===")
            
            # Create input tensor
            x = torch.randn(batch_size, HIDDEN_DIM, device=device, dtype=torch.float16)
            
            # Number of iterations (1M / B)
            num_iterations = max(10, 1000000 // batch_size)
            
            def operation():
                return linear(x)
            
            for clock in clocks_to_test:
                print(f"  Clock: {clock} MHz", end=" ")
                
                # Set clock
                self.set_gpu_clock(clock)
                
                # Measure
                latency_ms, energy_j = self.measure_energy_and_latency(
                    operation, num_iterations
                )
                
                # Calculate per-iteration metrics
                energy_per_iter = energy_j / num_iterations
                
                print(f"-> Latency: {latency_ms:.4f} ms, Energy: {energy_per_iter*1000:.4f} mJ")
                
                results.append({
                    'batch_size': batch_size,
                    'clock_mhz': clock,
                    'latency_ms': latency_ms,
                    'energy_mj': energy_per_iter * 1000,  # mJ per iteration
                    'num_iterations': num_iterations
                })
        
        # Reset clock
        self.reset_gpu_clock()
        
        # Save to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'batch_size', 'clock_mhz', 'latency_ms', 'energy_mj', 'num_iterations'
            ])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {csv_path}")
    
    def cleanup(self):
        """Clean up NVML."""
        self.reset_gpu_clock()
        pynvml.nvmlShutdown()


class PerformanceAnalyzer:
    """Analyze performance characteristics including MFU and MBU."""
    
    def __init__(self, csv_path: str, gpu_specs: Dict):
        self.csv_path = Path(csv_path)
        self.gpu_specs = gpu_specs
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load profiling results from CSV."""
        data = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'batch_size': int(row['batch_size']),
                    'clock_mhz': int(row['clock_mhz']),
                    'latency_ms': float(row['latency_ms']),
                    'energy_mj': float(row['energy_mj']),
                })
        return data
    
    def calculate_metrics(self) -> List[Dict]:
        """Calculate MFU, MBU, and other metrics."""
        
        results = []
        
        for entry in self.data:
            B = entry['batch_size']
            latency_s = entry['latency_ms'] / 1000.0
            
            # Calculate FLOPs for one linear operation: 2 * B * M * N
            # (2 accounts for multiply-add)
            flops = 2 * B * HIDDEN_DIM * INTERMEDIATE_DIM
            
            # Calculate achieved TFLOPS
            achieved_tflops = (flops / latency_s) / 1e12
            
            # Calculate MFU (Model FLOPS Utilization)
            peak_tflops = self.gpu_specs['peak_fp16_tflops']
            mfu = (achieved_tflops / peak_tflops) * 100  # Percentage
            
            # Calculate memory traffic (bytes)
            # Input: B * HIDDEN_DIM (FP16 = 2 bytes)
            # Weight: HIDDEN_DIM * INTERMEDIATE_DIM
            # Output: B * INTERMEDIATE_DIM
            bytes_input = B * HIDDEN_DIM * 2
            bytes_weight = HIDDEN_DIM * INTERMEDIATE_DIM * 2
            bytes_output = B * INTERMEDIATE_DIM * 2
            total_bytes = bytes_input + bytes_weight + bytes_output
            
            # Calculate achieved bandwidth (GB/s)
            achieved_bandwidth_gb_s = (total_bytes / latency_s) / 1e9
            
            # Calculate MBU (Model Bandwidth Utilization)
            peak_bandwidth = self.gpu_specs['memory_bandwidth_gb_s']
            mbu = (achieved_bandwidth_gb_s / peak_bandwidth) * 100
            
            # Calculate arithmetic intensity (FLOPS/byte)
            arithmetic_intensity = flops / total_bytes
            
            results.append({
                **entry,
                'achieved_tflops': achieved_tflops,
                'mfu_percent': mfu,
                'achieved_bandwidth_gb_s': achieved_bandwidth_gb_s,
                'mbu_percent': mbu,
                'arithmetic_intensity': arithmetic_intensity,
                'flops': flops,
                'bytes': total_bytes
            })
        
        return results


class Visualizer:
    """Create visualizations for the profiling results."""
    
    def __init__(self, analyzed_data: List[Dict], output_dir: str, gpu_specs: Dict):
        self.data = analyzed_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_specs = gpu_specs
    
    def plot_roofline(self):
        """Create roofline plots - one per clock frequency."""
        # Get GPU specs
        peak_tflops = self.gpu_specs['peak_fp16_tflops']
        peak_bandwidth_gb_s = self.gpu_specs['memory_bandwidth_gb_s']
        reference_clock = 1800  # Reference clock in MHz where peak is achieved
        
        # Get clock frequencies
        clocks = sorted(set(d['clock_mhz'] for d in self.data))
        num_clocks = len(clocks)
        
        # Create subplots - 2 columns for compact layout
        ncols = 2
        nrows = (num_clocks + 1) // 2  # Ceiling division
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows), sharex=True, sharey=True)
        
        # Flatten axes array for easier iteration
        if num_clocks == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_clocks > 2 else axes.reshape(-1)
        
        # Find global ranges for consistent axis limits
        min_ai = min(d['arithmetic_intensity'] for d in self.data)
        max_ai = max(d['arithmetic_intensity'] for d in self.data)
        ai_start = 10
        ai_end = max_ai * 2
        
        # Colors for batch sizes
        batch_sizes = sorted(set(d['batch_size'] for d in self.data))
        batch_colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
        batch_color_map = {bs: color for bs, color in zip(batch_sizes, batch_colors)}
        
        for idx, clock in enumerate(clocks):
            ax = axes[idx]
            # Scale peak performance: stays at peak for >= reference_clock, else scales linearly
            if clock >= reference_clock:
                scaled_peak_tflops = peak_tflops
            else:
                scaled_peak_tflops = peak_tflops * (clock / reference_clock)
            
            ridge_point_scaled = scaled_peak_tflops * 1000 / peak_bandwidth_gb_s
            
            # Memory-bound line (only label on first subplot for legend)
            ai_memory = np.logspace(np.log10(ai_start), np.log10(ridge_point_scaled), 100)
            perf_memory = (peak_bandwidth_gb_s / 1000) * ai_memory
            ax.loglog(ai_memory, perf_memory, '--', linewidth=3, color='#FF6B6B', zorder=1)
            
            # Draw ALL compute-bound lines for all clocks in light grey
            for other_clock in clocks:
                if other_clock >= reference_clock:
                    other_scaled_peak = peak_tflops
                else:
                    other_scaled_peak = peak_tflops * (other_clock / reference_clock)
                if other_clock == clock:
                    continue
                
                other_ridge_point = other_scaled_peak * 1000 / peak_bandwidth_gb_s
                ai_compute_other = np.logspace(np.log10(other_ridge_point), np.log10(ai_end), 100)
                perf_compute_other = np.ones_like(ai_compute_other) * other_scaled_peak
                ax.loglog(ai_compute_other, perf_compute_other, '-', linewidth=3, 
                         color='grey', alpha=0.5, zorder=0)
            # Compute-bound line for this specific clock
            ai_compute = np.logspace(np.log10(ridge_point_scaled), np.log10(ai_end), 100)
            perf_compute = np.ones_like(ai_compute) * scaled_peak_tflops
            ax.loglog(ai_compute, perf_compute, '--', linewidth=2.5, color='#4ECDC4',
                     label=f'Compute Bound ({scaled_peak_tflops:.1f} TFLOPS)', zorder=1)
            # Plot data points for this clock, colored by batch size
            clock_data = sorted(
                [d for d in self.data if d['clock_mhz'] == clock],
                key=lambda x: x['batch_size']
            )
            
            for batch_size in batch_sizes:
                batch_data = sorted(
                    [d for d in clock_data if d['batch_size'] == batch_size],
                    key=lambda x: x['arithmetic_intensity']
                )
                if not batch_data:
                    continue
                    
                ai = [d['arithmetic_intensity'] for d in batch_data]
                tflops = [d['achieved_tflops'] for d in batch_data]
                color = batch_color_map[batch_size]
                
                # Plot points (only label on first subplot for legend)
                label = f'B={batch_size}' if idx == 0 else None
                ax.scatter(ai, tflops, c=[color], s=120, alpha=0.8,
                          label=label, edgecolors='black', linewidths=0.7, zorder=3)
            
            # Formatting
            ax.set_ylabel('Performance (TFLOPS)', fontsize=14, fontweight='bold')
            ax.set_title(f'{clock} MHz', fontsize=16, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, which='both')
            ax.tick_params(labelsize=12)
            
            # Show x-label on bottom row
            row = idx // ncols
            if row == nrows - 1:
                ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=14, fontweight='bold')
        
        # Create single legend in remaining space (if odd number of clocks) or in last subplot area
        if num_clocks % 2 == 1:
            # Use the unused subplot space for legend
            legend_ax = axes[num_clocks]
            legend_ax.axis('off')
            
            # Get handles and labels from first subplot
            handles, labels = axes[0].get_legend_handles_labels()
            
            # Add roofline bound lines to legend
            from matplotlib.lines import Line2D
            bound_handles = [
                Line2D([0], [0], color='#FF6B6B', linewidth=3, linestyle='--', label='Memory Bound'),
                Line2D([0], [0], color='#4ECDC4', linewidth=3, linestyle='--', label='Compute Bound')
            ]
            handles = bound_handles + handles
            labels = ['Memory Bound', 'Compute Bound'] + labels
            
            legend_ax.legend(handles, labels, loc='center', fontsize=13, 
                           frameon=True, framealpha=0.9, ncol=2)
        else:
            # Place legend outside the plot area
            handles, labels = axes[0].get_legend_handles_labels()
            from matplotlib.lines import Line2D
            bound_handles = [
                Line2D([0], [0], color='#FF6B6B', linewidth=3, linestyle='--', label='Memory Bound'),
                Line2D([0], [0], color='#4ECDC4', linewidth=3, linestyle='--', label='Compute Bound')
            ]
            handles = bound_handles + handles
            labels = ['Memory Bound', 'Compute Bound'] + labels
            
            fig.legend(handles, labels, loc='lower center', fontsize=13, 
                      frameon=True, framealpha=0.9, ncol=4, bbox_to_anchor=(0.5, -0.02))
        
        # Hide any remaining unused subplots
        for idx in range(num_clocks, len(axes)):
            if idx != num_clocks or num_clocks % 2 == 0:
                axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(f'Roofline Model: {self.gpu_specs["name"]} - Linear Layer {HIDDEN_DIM}×{INTERMEDIATE_DIM}', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        output_path = self.output_dir / 'roofline_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved roofline plot to {output_path}")
        plt.close()
    
    def plot_clock_vs_metrics(self):
        """Plot Clock vs Energy and Clock vs Latency."""
        
        batch_sizes = sorted(set(d['batch_size'] for d in self.data))
        colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Clock vs Energy
        for batch_size, color in zip(batch_sizes, colors):
            batch_data = sorted(
                [d for d in self.data if d['batch_size'] == batch_size],
                key=lambda x: x['clock_mhz']
            )
            clocks = [d['clock_mhz'] for d in batch_data]
            energy = [d['energy_mj'] for d in batch_data]
            ax1.plot(clocks, energy, 'o-', color=color, label=f'B={batch_size}', 
                    linewidth=2, markersize=6)
        
        ax1.set_xlabel('GPU Clock (MHz)', fontsize=12)
        ax1.set_ylabel('Energy per Operation (mJ)', fontsize=12)
        ax1.set_title('GPU Clock vs Energy Consumption', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, ncol=2)
        
        # Plot 2: Clock vs Latency
        for batch_size, color in zip(batch_sizes, colors):
            batch_data = sorted(
                [d for d in self.data if d['batch_size'] == batch_size],
                key=lambda x: x['clock_mhz']
            )
            clocks = [d['clock_mhz'] for d in batch_data]
            latency = [d['latency_ms'] for d in batch_data]
            ax2.plot(clocks, latency, 's-', color=color, label=f'B={batch_size}', 
                    linewidth=2, markersize=6)
        
        ax2.set_xlabel('GPU Clock (MHz)', fontsize=12)
        ax2.set_ylabel('Latency per Operation (ms)', fontsize=12)
        ax2.set_title('GPU Clock vs Latency', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, ncol=2)
        
        plt.tight_layout()
        output_path = self.output_dir / 'clock_vs_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved clock vs metrics plot to {output_path}")
        plt.close()
    

def main():
    parser = argparse.ArgumentParser(
        description='Analyze PyTorch linear kernel under different GPU clocks'
    )
    parser.add_argument(
        '--batch-sizes', 
        type=int, 
        nargs='+', 
        default=[64, 128, 256, 384, 512, 768, 1024, 1536, 2048],
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--num-clocks',
        type=int,
        default=5,
        help='Number of different clock frequencies to test'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dvfs_profile_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--skip-profile',
        action='store_true',
        help='Skip profiling stage (use existing CSV)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyTorch Linear Kernel DVFS Analysis")
    print("=" * 60)
    print(f"Model dimensions: {HIDDEN_DIM} x {INTERMEDIATE_DIM} (LLAMA3-70B style)")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of clocks: {args.num_clocks}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Stage 1: Profiling
    if not args.skip_profile:
        print("\n### Stage 1: Profiling ###")
        profiler = DVFSProfiler(output_dir=args.output_dir)
        try:
            profiler.profile_linear_kernel(
                batch_sizes=args.batch_sizes,
                num_clocks=args.num_clocks
            )
        finally:
            profiler.cleanup()
    else:
        print("\n### Stage 1: Profiling (SKIPPED) ###")
    
    # Stage 2: Analysis
    print("\n### Stage 2: Analysis ###")
    csv_path = Path(args.output_dir) / "profile_results.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run profiling first.")
        return
    
    analyzer = PerformanceAnalyzer(csv_path, A6000_SPECS)
    analyzed_data = analyzer.calculate_metrics()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Peak FP16 Performance: {A6000_SPECS['peak_fp16_tflops']} TFLOPS")
    print(f"  Peak Memory Bandwidth: {A6000_SPECS['memory_bandwidth_gb_s']} GB/s")
    print(f"\n  MFU Range: {min(d['mfu_percent'] for d in analyzed_data):.2f}% - "
          f"{max(d['mfu_percent'] for d in analyzed_data):.2f}%")
    print(f"  MBU Range: {min(d['mbu_percent'] for d in analyzed_data):.2f}% - "
          f"{max(d['mbu_percent'] for d in analyzed_data):.2f}%")
    
    # Save detailed analysis
    analysis_csv = Path(args.output_dir) / "analysis_results.csv"
    with open(analysis_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=analyzed_data[0].keys())
        writer.writeheader()
        writer.writerows(analyzed_data)
    print(f"\nDetailed analysis saved to {analysis_csv}")
    
    # Stage 3: Visualization
    print("\n### Stage 3: Visualization ###")
    visualizer = Visualizer(analyzed_data, args.output_dir, A6000_SPECS)
    visualizer.plot_roofline()
    visualizer.plot_clock_vs_metrics()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"All results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
