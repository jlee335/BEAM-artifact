# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from scipy.interpolate import interp1d

DIRECTORY_PREFIX = '/workspace/disagg/energy-inf-v1-disagg/'


@dataclass
class ClockChoice:
    tp_degree: int
    pp_degree: int
    freq_mhz: int
    target_tps: int
    energy_per_req_j: float
    ttft_ms: float
    tbt_ms: float
    feasible: bool


class ClockSelector:
    """
    Selects the optimal GPU clock (frequency) configuration for minimum energy consumption
    while meeting SLO constraints.
    
    Expects CSV format: clock,tp,pp,tps,gpu_power,ttft_mean,tbt_mean,ttft_p99,tbt_p90
    
    Uses SLO constraints for TTFT (Time to First Token) and TBT (Time Between Tokens)
    to find the minimum energy configuration that meets performance requirements.
    Supports interpolation between measured data points.
    
    By default uses ttft_mean and tbt_mean. Set use_ttft_p99=True to use ttft_p99,
    and use_tbt_p90=True to use tbt_p90.
    """

    def __init__(self,
                 profiling_csv_path: str,
                 ttft_slo_ms: float,
                 tbt_slo_ms: float,
                 use_ttft_p99: bool = False,
                 use_tbt_p90: bool = False):
        self.profiling_csv_path = profiling_csv_path
        self.ttft_slo_ms = float(ttft_slo_ms)
        self.tbt_slo_ms = float(tbt_slo_ms)
        self.use_ttft_p99 = use_ttft_p99
        self.use_tbt_p90 = use_tbt_p90

        # Load and parse the CSV file
        raw_df = pd.read_csv(profiling_csv_path)

        # Validate required columns exist
        required_cols = [
            'clock', 'tp', 'pp', 'tps', 'gpu_power', 'ttft_mean', 'tbt_mean',
            'ttft_p99', 'tbt_p90'
        ]

        missing_cols = [
            col for col in required_cols if col not in raw_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"CSV file missing required columns: {missing_cols}")

        # Select columns based on configuration
        ttft_column = "ttft_p99" if self.use_ttft_p99 else "ttft_mean"
        tbt_column = "tbt_p90" if self.use_tbt_p90 else "tbt_mean"

        self.df = pd.DataFrame({
            "freq_mhz":
            raw_df["clock"].astype(int),
            "tp_degree":
            raw_df["tp"].astype(int),
            "pp_degree":
            raw_df["pp"].astype(int),
            "target_tps":
            raw_df["tps"].astype(int),
            "energy_per_req_j":
            raw_df["gpu_power"].astype(float),
            "ttft_ms":
            raw_df[ttft_column].astype(float),
            "tbt_ms":
            raw_df[tbt_column].astype(float)
        })

        # Validate the parsed data
        req_cols = [
            "freq_mhz", "tp_degree", "pp_degree", "target_tps",
            "energy_per_req_j", "ttft_ms", "tbt_ms"
        ]
        missing = [c for c in req_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Profiling DataFrame missing required columns: {missing}")

    def _interpolate_metrics(self, target_tp: int, target_pp: int,
                             target_tps: int) -> Optional[pd.DataFrame]:
        """
        Interpolate performance metrics for configurations not directly measured using scipy.interp1d.
        
        Args:
            target_tp: Target tensor parallelism degree
            target_pp: Target pipeline parallelism degree  
            target_tps: Target tokens per second
            
        Returns:
            DataFrame with interpolated metrics for each frequency
        """
        # Find exact TP/PP match for interpolation
        exact_tp_pp = self.df[(self.df["tp_degree"] == target_tp)
                              & (self.df["pp_degree"] == target_pp)]

        if exact_tp_pp.empty:
            return None

        available_tps = sorted(exact_tp_pp["target_tps"].unique())

        # Need at least 2 points for interpolation
        if len(available_tps) < 2:
            return None

        interpolated_rows = []
        frequencies = exact_tp_pp["freq_mhz"].unique()

        for freq in frequencies:
            # Get data for this frequency
            freq_data = exact_tp_pp[exact_tp_pp["freq_mhz"] ==
                                    freq].sort_values("target_tps")

            if len(freq_data) < 2:
                continue

            tps_points = freq_data["target_tps"].values

            # Check if target_tps is within or close to the available range
            min_tps, max_tps = min(tps_points), max(tps_points)

            # Skip if target is too far outside the range (avoid unreliable extrapolation)
            if target_tps < min_tps * 0.5 or target_tps > max_tps * 2.0:
                continue

            try:
                # Create interpolation functions for each metric
                energy_interp = interp1d(tps_points,
                                         freq_data["energy_per_req_j"].values,
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value='extrapolate')
                ttft_interp = interp1d(tps_points,
                                       freq_data["ttft_ms"].values,
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value='extrapolate')
                tbt_interp = interp1d(tps_points,
                                      freq_data["tbt_ms"].values,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value='extrapolate')

                # Interpolate metrics at target TPS
                interpolated_row = {
                    "freq_mhz": int(freq),
                    "tp_degree": target_tp,
                    "pp_degree": target_pp,
                    "target_tps": target_tps,
                    "energy_per_req_j": float(energy_interp(target_tps)),
                    "ttft_ms": float(ttft_interp(target_tps)),
                    "tbt_ms": float(tbt_interp(target_tps))
                }
                interpolated_rows.append(interpolated_row)

            except (ValueError, RuntimeError):
                # Skip this frequency if interpolation fails
                continue

        if interpolated_rows:
            return pd.DataFrame(interpolated_rows)

        return None

    def choose(self, tp_degree: int, pp_degree: int,
               target_tps: int) -> Optional[ClockChoice]:
        """
        Return the optimal clock frequency for minimum energy consumption 
        while meeting SLO constraints.
        
        Args:
            tp_degree: Tensor parallelism degree
            pp_degree: Pipeline parallelism degree  
            target_tps: Target tokens per second
            
        Returns:
            ClockChoice object with optimal frequency and performance metrics
        """
        # First, try to find exact match in measured data
        exact_match = self.df[(self.df["tp_degree"] == tp_degree)
                              & (self.df["pp_degree"] == pp_degree) &
                              (self.df["target_tps"] == target_tps)].copy()

        if exact_match.empty:
            # Try interpolation if exact match not found
            interpolated_data = self._interpolate_metrics(
                tp_degree, pp_degree, target_tps)
            if interpolated_data is not None:
                candidates = interpolated_data
            else:
                # Fallback: find closest configuration
                candidates = self._find_closest_config(tp_degree, pp_degree,
                                                       target_tps)
                if candidates is None or candidates.empty:
                    return None
        else:
            candidates = exact_match

        # Split feasible vs infeasible based on SLO constraints
        feasible = candidates[
            (candidates["ttft_ms"] <= self.ttft_slo_ms)
            & (candidates["tbt_ms"] <= self.tbt_slo_ms)].copy()

        if not feasible.empty:
            # Choose lowest energy configuration among feasible ones
            feasible = feasible.sort_values(
                by=["energy_per_req_j", "ttft_ms", "freq_mhz"],
                ascending=[True, True, True])
            row = feasible.iloc[0]
            return ClockChoice(tp_degree=int(row["tp_degree"]),
                               pp_degree=int(row["pp_degree"]),
                               freq_mhz=int(row["freq_mhz"]),
                               target_tps=int(row["target_tps"]),
                               energy_per_req_j=float(row["energy_per_req_j"]),
                               ttft_ms=float(row["ttft_ms"]),
                               tbt_ms=float(row["tbt_ms"]),
                               feasible=True)

        # No feasible solution: set clock to maximum frequency
        # Find the maximum frequency available for this configuration
        max_freq_row = candidates.loc[candidates["freq_mhz"].idxmax()]

        return ClockChoice(tp_degree=int(max_freq_row["tp_degree"]),
                           pp_degree=int(max_freq_row["pp_degree"]),
                           freq_mhz=int(max_freq_row["freq_mhz"]),
                           target_tps=int(max_freq_row["target_tps"]),
                           energy_per_req_j=float(
                               max_freq_row["energy_per_req_j"]),
                           ttft_ms=float(max_freq_row["ttft_ms"]),
                           tbt_ms=float(max_freq_row["tbt_ms"]),
                           feasible=False)

    def _find_closest_config(self, target_tp: int, target_pp: int,
                             target_tps: int) -> Optional[pd.DataFrame]:
        """
        Find the closest configuration when exact match is not available.
        
        Args:
            target_tp: Target tensor parallelism degree
            target_pp: Target pipeline parallelism degree
            target_tps: Target tokens per second
            
        Returns:
            DataFrame with closest configuration data
        """
        # Priority: exact tp/pp match, then closest tps
        exact_tp_pp = self.df[(self.df["tp_degree"] == target_tp)
                              & (self.df["pp_degree"] == target_pp)]

        assert not exact_tp_pp.empty
        # Find closest TPS for exact TP/PP match
        available_tps = exact_tp_pp["target_tps"].unique()
        closest_tps = min(available_tps, key=lambda x: abs(x - target_tps))
        return exact_tp_pp[exact_tp_pp["target_tps"] == closest_tps].copy()


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import numpy as np

    # Configuration
    # csv_path = '/workspace/vllm-energy-v1/vllm/benchmarks/energy/dynamollm_profiles/dynamo_dvfs_profile_NVIDIA RTX A6000_meta-llama_Llama-3.3-70B-Instruct.csv'
    # csv_path = '/workspace/vllm-energy-v1/vllm/benchmarks/energy/dynamollm_profiles/dynamo_dvfs_profile_NVIDIA A100-SXM4-80GB_meta-llama_Llama-3.3-70B-Instruct.csv'
    csv_path = f'{DIRECTORY_PREFIX}benchmarks/energy/dynamollm_profiles/dynamo_dvfs_profile_NVIDIA RTX A6000_Qwen_Qwen2.5-32B.csv'

    # Fixed TP×PP configuration (using TP=2, PP=2 as it has good data coverage)
    fixed_tp = 1
    fixed_pp = 4

    # A6000
    TTFT_SLO = 1000
    TBT_SLO = 300

    # A100
    # TTFT_SLO = 1000
    # TBT_SLO = 200

    # SLO constraints
    # Test different modes: mean, P99/P90, and combined
    # selector_mean = ClockSelector(csv_path,
    #                               ttft_slo_ms=TTFT_SLO,
    #                               tbt_slo_ms=TBT_SLO,
    #                               use_ttft_p99=False,
    #                               use_tbt_p90=False)
    selector_p99_p90 = ClockSelector(csv_path,
                                     ttft_slo_ms=TTFT_SLO,
                                     tbt_slo_ms=TBT_SLO,
                                     use_ttft_p99=False,
                                     use_tbt_p90=True)
    # selector_mixed = ClockSelector(csv_path,
    #                                ttft_slo_ms=TTFT_SLO,
    #                                tbt_slo_ms=TBT_SLO,
    #                                use_ttft_p99=True,
    #                                use_tbt_p90=False)

    # TPS values to test (including interpolation points)
    # Measured TPS values in the dataset (these are actual data points)
    # tps_values = list(range(300, 1800, 100))  # 300 to 1700 in steps of 100

    # A100
    # tps_values = list(range(500, 10000, 300))
    
    # A6000
    tps_values = list(range(300, 1800, 50))

    optimal_freqs = []
    energy_values = []
    ttft_values = []
    tbt_values = []
    feasible_flags = []
    prediction_times = []  # Track prediction times in milliseconds
    is_measured = []  # Track whether each point is measured or interpolated

    # Use the mean selector for the main analysis
    selector = selector_p99_p90

    print(
        f"Analyzing optimal clock frequencies for TP={fixed_tp}, PP={fixed_pp}"
    )
    print(
        f"Using Mean mode - TTFT SLO: {TTFT_SLO}ms, TBT SLO: {TBT_SLO}ms"
    )
    print(
        "TPS\tOptimal Freq (MHz)\tEnergy (J)\tTTFT (ms)\tTBT (ms)\tFeasible\tPrediction Time (ms)\tData Type"
    )
    print("-" * 120)

    for tps in tps_values:
        # Measure prediction time
        start_time = time.time()
        choice = selector.choose(tp_degree=fixed_tp,
                                 pp_degree=fixed_pp,
                                 target_tps=tps)
        end_time = time.time()
        prediction_time_ms = (end_time -
                              start_time) * 1000  # Convert to milliseconds

        if choice:
            # Check if this TPS value exists as measured data in the CSV
            exact_match = selector.df[(selector.df["tp_degree"] == fixed_tp)
                                      & (selector.df["pp_degree"] == fixed_pp)
                                      & (selector.df["target_tps"] == tps)]
            is_measured_point = not exact_match.empty

            optimal_freqs.append(choice.freq_mhz)
            energy_values.append(choice.energy_per_req_j)
            ttft_values.append(choice.ttft_ms)
            tbt_values.append(choice.tbt_ms)
            feasible_flags.append(choice.feasible)
            prediction_times.append(prediction_time_ms)
            is_measured.append(is_measured_point)

            measured_status = "Measured" if is_measured_point else "Interpolated"
            print(
                f"{tps}\t{choice.freq_mhz}\t\t\t{choice.energy_per_req_j:.1f}\t\t{choice.ttft_ms:.1f}\t\t{choice.tbt_ms:.1f}\t\t{choice.feasible}\t\t{prediction_time_ms:.2f}\t{measured_status}"
            )
        else:
            print(
                f"{tps}\tNo solution found\t\t\t\t\t\t\t\t\t\t{prediction_time_ms:.2f}"
            )

    # Create plots
    if optimal_freqs:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Separate data points by measured vs interpolated and feasible vs infeasible
        tps_plot = tps_values[:len(optimal_freqs)]

        # Create masks for different categories
        measured_feasible = [(is_measured[i] and feasible_flags[i])
                             for i in range(len(optimal_freqs))]
        measured_infeasible = [(is_measured[i] and not feasible_flags[i])
                               for i in range(len(optimal_freqs))]
        interpolated_feasible = [(not is_measured[i] and feasible_flags[i])
                                 for i in range(len(optimal_freqs))]
        interpolated_infeasible = [(not is_measured[i]
                                    and not feasible_flags[i])
                                   for i in range(len(optimal_freqs))]

        # Plot 1: TPS vs Optimal Clock Frequency
        if any(measured_feasible):
            ax1.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_feasible[i]
            ], [
                optimal_freqs[i]
                for i in range(len(optimal_freqs)) if measured_feasible[i]
            ],
                        c='green',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Feasible')
        if any(measured_infeasible):
            ax1.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_infeasible[i]
            ], [
                optimal_freqs[i]
                for i in range(len(optimal_freqs)) if measured_infeasible[i]
            ],
                        c='red',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Infeasible')
        if any(interpolated_feasible):
            ax1.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_feasible[i]
            ], [
                optimal_freqs[i]
                for i in range(len(optimal_freqs)) if interpolated_feasible[i]
            ],
                        c='green',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Feasible')
        if any(interpolated_infeasible):
            ax1.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_infeasible[i]
            ], [
                optimal_freqs[i] for i in range(len(optimal_freqs))
                if interpolated_infeasible[i]
            ],
                        c='red',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Infeasible')

        ax1.plot(tps_plot, optimal_freqs, 'b-', alpha=0.3)
        ax1.set_xlabel('Target TPS (Tokens Per Second)')
        ax1.set_ylabel('Optimal Clock Frequency (MHz)')
        ax1.set_title(
            f'TPS vs Optimal Clock Frequency\n(TP={fixed_tp}, PP={fixed_pp})')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Plot 2: TPS vs Energy Consumption
        if any(measured_feasible):
            ax2.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_feasible[i]
            ], [
                energy_values[i]
                for i in range(len(energy_values)) if measured_feasible[i]
            ],
                        c='green',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Feasible')
        if any(measured_infeasible):
            ax2.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_infeasible[i]
            ], [
                energy_values[i]
                for i in range(len(energy_values)) if measured_infeasible[i]
            ],
                        c='red',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Infeasible')
        if any(interpolated_feasible):
            ax2.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_feasible[i]
            ], [
                energy_values[i]
                for i in range(len(energy_values)) if interpolated_feasible[i]
            ],
                        c='green',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Feasible')
        if any(interpolated_infeasible):
            ax2.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_infeasible[i]
            ], [
                energy_values[i] for i in range(len(energy_values))
                if interpolated_infeasible[i]
            ],
                        c='red',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Infeasible')

        ax2.plot(tps_plot, energy_values, 'r-', alpha=0.3)
        ax2.set_xlabel('Target TPS (Tokens Per Second)')
        ax2.set_ylabel('Energy per Request (J)')
        ax2.set_title(
            f'TPS vs Energy Consumption\n(TP={fixed_tp}, PP={fixed_pp})')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # Plot 3: TPS vs TTFT
        if any(measured_feasible):
            ax3.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_feasible[i]
            ], [
                ttft_values[i]
                for i in range(len(ttft_values)) if measured_feasible[i]
            ],
                        c='green',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Feasible')
        if any(measured_infeasible):
            ax3.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_infeasible[i]
            ], [
                ttft_values[i]
                for i in range(len(ttft_values)) if measured_infeasible[i]
            ],
                        c='red',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Infeasible')
        if any(interpolated_feasible):
            ax3.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_feasible[i]
            ], [
                ttft_values[i]
                for i in range(len(ttft_values)) if interpolated_feasible[i]
            ],
                        c='green',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Feasible')
        if any(interpolated_infeasible):
            ax3.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_infeasible[i]
            ], [
                ttft_values[i]
                for i in range(len(ttft_values)) if interpolated_infeasible[i]
            ],
                        c='red',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Infeasible')

        ax3.plot(tps_plot, ttft_values, 'g-', alpha=0.3)
        ax3.axhline(y=TTFT_SLO,
                    color='red',
                    linestyle='--',
                    alpha=0.5,
                    label=f'TTFT SLO ({TTFT_SLO} ms)')
        ax3.set_xlabel('Target TPS (Tokens Per Second)')
        ax3.set_ylabel('TTFT (ms)')
        ax3.set_title(
            f'TPS vs Time to First Token\n(TP={fixed_tp}, PP={fixed_pp})')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        # Plot 4: TPS vs TBT
        if any(measured_feasible):
            ax4.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_feasible[i]
            ], [
                tbt_values[i]
                for i in range(len(tbt_values)) if measured_feasible[i]
            ],
                        c='green',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Feasible')
        if any(measured_infeasible):
            ax4.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if measured_infeasible[i]
            ], [
                tbt_values[i]
                for i in range(len(tbt_values)) if measured_infeasible[i]
            ],
                        c='red',
                        s=80,
                        alpha=0.8,
                        marker='o',
                        label='Measured - Infeasible')
        if any(interpolated_feasible):
            ax4.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_feasible[i]
            ], [
                tbt_values[i]
                for i in range(len(tbt_values)) if interpolated_feasible[i]
            ],
                        c='green',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Feasible')
        if any(interpolated_infeasible):
            ax4.scatter([
                tps_plot[i]
                for i in range(len(tps_plot)) if interpolated_infeasible[i]
            ], [
                tbt_values[i]
                for i in range(len(tbt_values)) if interpolated_infeasible[i]
            ],
                        c='red',
                        s=50,
                        alpha=0.6,
                        marker='^',
                        label='Interpolated - Infeasible')

        ax4.plot(tps_plot, tbt_values, 'm-', alpha=0.3)
        ax4.axhline(y=TBT_SLO,
                    color='red',
                    linestyle='--',
                    alpha=0.5,
                    label=f'TBT SLO ({TBT_SLO} ms)')
        ax4.set_xlabel('Target TPS (Tokens Per Second)')
        ax4.set_ylabel('TBT (ms)')
        ax4.set_title(
            f'TPS vs Time Between Tokens\n(TP={fixed_tp}, PP={fixed_pp})')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')

        # Adjust layout and save
        plt.tight_layout()

        # Save plot
        output_path = f'{DIRECTORY_PREFIX}vllm/v1/core/sched/tps_vs_optimal_clock_tp{fixed_tp}_pp{fixed_pp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

        # Also save data to CSV for reference
        data_path = f'{DIRECTORY_PREFIX}vllm/v1/core/sched/tps_vs_optimal_clock_tp{fixed_tp}_pp{fixed_pp}_data.csv'
        import pandas as pd
        results_df = pd.DataFrame({
            'tps': tps_values[:len(optimal_freqs)],
            'optimal_freq_mhz': optimal_freqs,
            'energy_per_req_j': energy_values,
            'ttft_ms': ttft_values,
            'tbt_ms': tbt_values,
            'feasible': feasible_flags,
            'prediction_time_ms': prediction_times,
            'is_measured': is_measured
        })
        results_df.to_csv(data_path, index=False)
        print(f"Data saved to: {data_path}")

        # Print timing statistics
        print("\n" + "=" * 60)
        print("PREDICTION TIMING STATISTICS")
        print("=" * 60)
        if prediction_times:
            avg_time = np.mean(prediction_times)
            min_time = np.min(prediction_times)
            max_time = np.max(prediction_times)
            total_time = np.sum(prediction_times)

            print(f"Total predictions: {len(prediction_times)}")
            print(f"Average prediction time: {avg_time:.3f} ms")
            print(f"Minimum prediction time: {min_time:.3f} ms")
            print(f"Maximum prediction time: {max_time:.3f} ms")
            print(f"Total time for all predictions: {total_time:.3f} ms")
            print(f"Predictions per second: {1000/avg_time:.1f} pred/sec")

        plt.show()
    else:
        print("No valid solutions found for plotting")
