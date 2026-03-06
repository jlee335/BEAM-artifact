# SPDX-License-Identifier: Apache-2.0
"""
MILP solver for finding optimal GPU clock frequency that minimizes energy
while adhering to TTFT and TPOT SLO constraints.

Uses PuLP for Mixed Integer Linear Programming optimization.
"""

import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from scipy import interpolate


def load_metrics_data(csv_path: str) -> pd.DataFrame:
    """Load and return the performance metrics CSV data."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)


def filter_data_by_config(df: pd.DataFrame, in_category: str,
                          out_category: str, tps: int, tp: int,
                          pp: int) -> pd.DataFrame:
    """Filter dataframe by the specified configuration parameters."""
    filtered_df = df[(df['in_category'] == in_category)
                     & (df['out_category'] == out_category) &
                     (df['tps'] == tps) & (df['tp'] == tp) &
                     (df['pp'] == pp)].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No data found for configuration: "
            f"in_category={in_category}, out_category={out_category}, "
            f"tps={tps}, tp={tp}, pp={pp}")

    return filtered_df


def solve_optimal_clock(df: pd.DataFrame, ttft_slo: float,
                        tpot_slo: float) -> Dict[str, Any]:
    """
    Solve MILP optimization problem to find optimal configuration.
    
    Objective: Minimize gpu_power (energy consumption)
    Constraints: ttft <= ttft_slo, tpot <= tpot_slo
    
    Each row in df represents a specific measurement point that can be selected.
    
    Returns:
        Dictionary containing optimal solution details
    """

    # Create the optimization problem
    prob = pulp.LpProblem("Optimal_Configuration_Selection", pulp.LpMinimize)

    # Create binary decision variables for each measurement point (row)
    # x[i] = 1 if measurement point i is selected, 0 otherwise
    x = {}
    for i in df.index:
        x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')

    # Objective: minimize GPU power consumption
    prob += pulp.lpSum([df.loc[i, 'gpu_power'] * x[i] for i in df.index])

    # Constraint 1: Exactly one measurement point must be selected
    prob += pulp.lpSum([x[i] for i in df.index]) == 1

    # Constraint 2: TTFT SLO constraint
    prob += pulp.lpSum([df.loc[i, 'ttft'] * x[i]
                        for i in df.index]) <= ttft_slo

    # Constraint 3: TPOT SLO constraint
    prob += pulp.lpSum([df.loc[i, 'tpot'] * x[i]
                        for i in df.index]) <= tpot_slo

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

    # Check solution status
    status = pulp.LpStatus[prob.status]

    if status != 'Optimal':
        return {
            'status': status,
            'optimal_clock': None,
            'optimal_power': None,
            'ttft': None,
            'tpot': None,
            'feasible': False,
            'message': f"No feasible solution found. Status: {status}",
            'available_configs': len(df)
        }

    # Extract optimal solution
    optimal_idx = None
    for i in df.index:
        if x[i].varValue == 1:
            optimal_idx = i
            break

    optimal_row = df.loc[optimal_idx]
    optimal_clock = optimal_row['clock']
    optimal_power = optimal_row['gpu_power']
    optimal_ttft = optimal_row['ttft']
    optimal_tpot = optimal_row['tpot']

    return {
        'status': status,
        'optimal_clock': optimal_clock,
        'optimal_power': optimal_power,
        'ttft': optimal_ttft,
        'tpot': optimal_tpot,
        'feasible': True,
        'objective_value': prob.objective.value(),
        'ttft_slo': ttft_slo,
        'tpot_slo': tpot_slo,
        'ttft_slack': ttft_slo - optimal_ttft,
        'tpot_slack': tpot_slo - optimal_tpot,
        'selected_row': optimal_idx,
        'available_configs': len(df)
    }


def dynamo_solve(csv_path: str, in_category: str, out_category: str, tps: int,
                 tp: int, pp: int, ttft_slo: float,
                 tpot_slo: float) -> Dict[str, Any]:
    """
    Main function to solve the optimal clock selection problem.
    
    Args:
        csv_path: Path to CSV file with performance metrics
        in_category: Input token category ('S', 'M', 'L')
        out_category: Output token category ('S', 'M', 'L') 
        tps: Tokens per second
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        ttft_slo: Time to First Token SLO constraint (ms)
        tpot_slo: Token Per Output Time SLO constraint (ms)
    
    Returns:
        Dictionary containing optimization results
    """

    df = load_metrics_data(csv_path)

    try:
        filtered_df = filter_data_by_config(df, in_category, out_category, tps,
                                            tp, pp)
        result = solve_optimal_clock(filtered_df, ttft_slo, tpot_slo)

        return result

    except ValueError as e:
        return {'status': 'Error', 'feasible': False, 'message': str(e)}


def plot_load_vs_optimal_clock(csv_path: str,
                               in_category: str,
                               out_category: str,
                               tp: int,
                               pp: int,
                               ttft_slo: float,
                               tpot_slo: float,
                               output_path: Optional[str] = None,
                               tps_step: int = 25) -> None:
    """
    Plot load (TPS) vs optimal clock frequency with interpolation for fine-grained analysis.
    
    Args:
        csv_path: Path to CSV file with performance metrics
        in_category: Input token category ('S', 'M', 'L')
        out_category: Output token category ('S', 'M', 'L') 
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        ttft_slo: Time to First Token SLO constraint (ms)
        tpot_slo: Token Per Output Time SLO constraint (ms)
        output_path: Path to save plot image (optional)
        tps_step: Step size for fine-grained TPS interpolation (default: 25)
    """

    # Load data and get available TPS values for this configuration
    df = load_metrics_data(csv_path)

    # Filter to get all TPS values for this configuration
    config_df = df[(df['in_category'] == in_category)
                   & (df['out_category'] == out_category) & (df['tp'] == tp) &
                   (df['pp'] == pp)]

    if config_df.empty:
        print(f"No data found for configuration: in_category={in_category}, "
              f"out_category={out_category}, tp={tp}, pp={pp}")
        return

    available_tps = sorted(config_df['tps'].unique())
    print(f"Found TPS values: {available_tps}")

    # Step 1: Get discrete optimal points from MILP solver
    optimal_clocks = []
    optimal_powers = []
    feasible_tps = []
    infeasible_clocks = []
    infeasible_powers = []
    infeasible_tps = []

    # Get max clock frequency and corresponding power for infeasible solutions
    max_clock = config_df['clock'].max()
    max_clock_power = config_df[config_df['clock'] ==
                                max_clock]['gpu_power'].iloc[0]

    for tps in available_tps:
        result = dynamo_solve(csv_path, in_category, out_category, tps, tp, pp,
                              ttft_slo, tpot_slo)

        if result['feasible']:
            feasible_tps.append(tps)
            optimal_clocks.append(result['optimal_clock'])
            optimal_powers.append(result['optimal_power'])
            print(
                f"TPS {tps}: Optimal clock {result['optimal_clock']} MHz, Power {result['optimal_power']:.1f} W"
            )
        else:
            infeasible_tps.append(tps)
            infeasible_clocks.append(max_clock)
            infeasible_powers.append(max_clock_power)
            print(
                f"TPS {tps}: No feasible solution - using max clock {max_clock} MHz, Power {max_clock_power:.1f} W"
            )

    # Check if we have any solutions (feasible or infeasible)
    total_solutions = len(feasible_tps) + len(infeasible_tps)
    if total_solutions == 0:
        print("No solutions found")
        return

    # For interpolation, we need at least 2 feasible solutions
    if len(feasible_tps) < 2:
        print(
            f"Only {len(feasible_tps)} feasible solutions - skipping interpolation, showing discrete points only"
        )
        if len(feasible_tps) == 1:
            print(
                f"Single feasible solution: TPS {feasible_tps[0]}, Clock {optimal_clocks[0]} MHz, Power {optimal_powers[0]:.1f} W"
            )

    # Step 2: Create interpolation functions (only if we have enough feasible solutions)
    fine_tps = None
    fine_clocks = None
    fine_powers = None

    if len(feasible_tps) >= 2:
        print(
            "\nCreating interpolation functions for fine-grained analysis...")

        # Convert to numpy arrays for interpolation
        tps_array = np.array(feasible_tps)
        clock_array = np.array(optimal_clocks)
        power_array = np.array(optimal_powers)

        # Create interpolation functions
        clock_interp = interpolate.interp1d(tps_array,
                                            clock_array,
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value='extrapolate')
        power_interp = interpolate.interp1d(tps_array,
                                            power_array,
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value='extrapolate')

        # Step 3: Generate fine-grained TPS values
        tps_min, tps_max = min(feasible_tps), max(feasible_tps)
        fine_tps = np.arange(tps_min, tps_max + tps_step, tps_step)

        # Interpolate optimal values for fine-grained TPS
        fine_clocks = clock_interp(fine_tps)
        fine_powers = power_interp(fine_tps)

        # Ensure interpolated clocks are within reasonable bounds
        available_clocks = sorted(df['clock'].unique())
        fine_clocks = np.clip(fine_clocks, min(available_clocks),
                              max(available_clocks))

        print(
            f"Generated {len(fine_tps)} fine-grained points from {tps_min} to {tps_max} TPS (step: {tps_step})"
        )

    # Step 4: Create enhanced plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Load vs Optimal Clock
    # Smooth interpolated curve (only if available)
    if fine_tps is not None and fine_clocks is not None:
        ax1.plot(fine_tps,
                 fine_clocks,
                 'b-',
                 linewidth=2,
                 alpha=0.8,
                 label='Interpolated (Feasible)')

    # Feasible discrete points (normal color)
    if feasible_tps:
        ax1.plot(feasible_tps,
                 optimal_clocks,
                 'bo',
                 markersize=10,
                 label='MILP Optimal (Feasible)',
                 zorder=5)

    # Infeasible discrete points (dimmer color)
    if infeasible_tps:
        ax1.plot(infeasible_tps,
                 infeasible_clocks,
                 'bo',
                 markersize=10,
                 alpha=0.4,
                 label='Max Clock (Infeasible)',
                 zorder=4)

    ax1.set_xlabel('Load (Tokens Per Second)', fontsize=12)
    ax1.set_ylabel('Clock Frequency (MHz)', fontsize=12)
    ax1.set_title(
        f'Load vs Clock Frequency\n'
        f'Config: {in_category}->{out_category}, TP={tp}, PP={pp}\n'
        f'SLOs: TTFT≤{ttft_slo}ms, TPOT≤{tpot_slo}ms',
        fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    available_clocks = sorted(df['clock'].unique())
    ax1.set_ylim(bottom=min(available_clocks) - 100)

    # Add discrete point labels for feasible solutions
    for x, y in zip(feasible_tps, optimal_clocks):
        ax1.annotate(f'{y}', (x, y),
                     textcoords="offset points",
                     xytext=(0, 15),
                     ha='center',
                     fontsize=9,
                     fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="white",
                               alpha=0.8))

    # Add discrete point labels for infeasible solutions (dimmer)
    for x, y in zip(infeasible_tps, infeasible_clocks):
        ax1.annotate(f'{y}', (x, y),
                     textcoords="offset points",
                     xytext=(0, 15),
                     ha='center',
                     fontsize=9,
                     fontweight='normal',
                     alpha=0.6,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightgray",
                               alpha=0.6))

    # Plot 2: Load vs Power
    # Smooth interpolated curve (only if available)
    if fine_tps is not None and fine_powers is not None:
        ax2.plot(fine_tps,
                 fine_powers,
                 'r-',
                 linewidth=2,
                 alpha=0.8,
                 label='Interpolated (Feasible)')

    # Feasible discrete points (normal color)
    if feasible_tps:
        ax2.plot(feasible_tps,
                 optimal_powers,
                 'ro',
                 markersize=10,
                 label='MILP Optimal (Feasible)',
                 zorder=5)

    # Infeasible discrete points (dimmer color)
    if infeasible_tps:
        ax2.plot(infeasible_tps,
                 infeasible_powers,
                 'ro',
                 markersize=10,
                 alpha=0.4,
                 label='Max Power (Infeasible)',
                 zorder=4)

    ax2.set_xlabel('Load (Tokens Per Second)', fontsize=12)
    ax2.set_ylabel('Power Consumption (W)', fontsize=12)
    ax2.set_title('Load vs Power Consumption', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # Add discrete point labels for feasible solutions
    for x, y in zip(feasible_tps, optimal_powers):
        ax2.annotate(f'{y:.0f}W', (x, y),
                     textcoords="offset points",
                     xytext=(0, 15),
                     ha='center',
                     fontsize=9,
                     fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="white",
                               alpha=0.8))

    # Add discrete point labels for infeasible solutions (dimmer)
    for x, y in zip(infeasible_tps, infeasible_powers):
        ax2.annotate(f'{y:.0f}W', (x, y),
                     textcoords="offset points",
                     xytext=(0, 15),
                     ha='center',
                     fontsize=9,
                     fontweight='normal',
                     alpha=0.6,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightgray",
                               alpha=0.6))

    plt.tight_layout()

    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    # Print summary
    print(f"\nSummary for {in_category}->{out_category}, TP={tp}, PP={pp}:")

    all_tps = feasible_tps + infeasible_tps
    all_clocks = optimal_clocks + infeasible_clocks
    all_powers = optimal_powers + infeasible_powers

    if all_tps:
        print(f"TPS range: {min(all_tps)}-{max(all_tps)}")
        print(f"Clock range: {min(all_clocks)}-{max(all_clocks)} MHz")
        print(f"Power range: {min(all_powers):.1f}-{max(all_powers):.1f} W")

        if feasible_tps:
            print(
                f"Feasible solutions: {len(feasible_tps)}/{len(all_tps)} TPS values"
            )
        if infeasible_tps:
            print(
                f"Infeasible solutions: {len(infeasible_tps)}/{len(all_tps)} TPS values (using max clock)"
            )
    else:
        print("No solutions found")


if __name__ == "__main__":
    # Static configuration values
    csv_path = "../dynamollm_profiles/collected_metrics_shorttime_20s.csv"
    in_category = "M"
    out_category = "M"

    tp = 2
    pp = 2
    ttft_slo = 2000.0
    tpot_slo = 400.0
    output_path = "test_interpolation_plot.png"

    print("Running with static configuration:")
    print(f"  CSV path: {csv_path}")
    print(f"  Input category: {in_category}")
    print(f"  Output category: {out_category}")
    print(f"  Tensor parallelism: {tp}")
    print(f"  Pipeline parallelism: {pp}")
    print(f"  TTFT SLO: {ttft_slo} ms")
    print(f"  TPOT SLO: {tpot_slo} ms")
    print(f"  Output path: {output_path}")
    print()

    plot_load_vs_optimal_clock(csv_path, in_category, out_category, tp, pp,
                               ttft_slo, tpot_slo, output_path)
