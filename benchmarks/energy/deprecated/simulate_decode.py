# SPDX-License-Identifier: Apache-2.0
# dvfs_profile_b'<GPU_NAME>'_<MODEL_NAME>.csv

import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# Set this flag to control whether to show the time graph
SHOW_TIME = False  # Set to False to hide the time graph

CONSIDER_DECODE_DEBT = True
DECODE_DEBT_FACTOR = 0


def set_fig_size(plt, ncols=1, nrows=1):
    # Larger plot size, scale with number of subplots
    plt.figure(figsize=(7 * ncols, 6 * nrows))


def save_fig(plt, model_name, gpu_name, mode, suffix):
    # y start from 0
    plt.ylim(0, None)
    plt.xlim(0, 2100)
    plt.savefig(f"[{mode}]{model_name}_{gpu_name}_{suffix}.png",
                bbox_inches="tight")
    plt.close()


PREFILL_SIZE = 4096
NUM_DECODES = 64

# List of bound sizes to consider
bound_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
decode_batch_sizes = [8, 16, 32, 64, 128]

for file in os.listdir("."):
    if file.endswith(".csv"):
        # Parse the file name to get the GPU name and model name with regex
        match = re.search(
            r"dvfs_profile_([^_]+(?: [^_]+)*)_([^_]+(?:_[^_]+)*)_(.*)\.csv",
            file)
        if match:
            gpu_name = match.group(1).strip()
            model_name = match.group(2).strip()
            # Only allow A6000 and A100 in GPU name, else continue
            if "A6000" not in gpu_name:
                continue

            print(f"GPU: {gpu_name}, Model: {model_name}")
            df_orig = pd.read_csv(file)

            # mJ to J
            df_orig[
                "energy_consumption"] = df_orig["energy_consumption"] / 1000
            df_orig["energy-per-token"] = (df_orig["energy_consumption"] /
                                           df_orig["batch_size"])
            df_orig["power"] = df_orig["energy_consumption"] / df_orig[
                "time_taken"]
            df_orig["time-per-token"] = df_orig["time_taken"] / df_orig[
                "batch_size"]

            if "A100" not in gpu_name:
                standard_clock = 1770
            else:
                standard_clock = 1050

            df = df_orig

            # Helper: find smallest batch size >= bs
            def find_smallest_batch_size_ge(bs, batch_sizes_sorted):
                for b in batch_sizes_sorted:
                    if b >= bs:
                        return b
                raise ValueError(
                    f"No batch size >= {bs} found in available batch sizes: {batch_sizes_sorted}"
                )

            def get_time_taken(bs, clock):
                df_clock = df[df["clock"] == clock]
                batch_time_map_clock = (df_clock.groupby("batch_size")
                                        ["time_taken"].mean().to_dict())
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

            def get_energy(bs, clock):
                df_clock = df[df["clock"] == clock]
                batch_energy_map_clock = (df_clock.groupby(
                    "batch_size")["energy_consumption"].mean().to_dict())
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
                idle_csv = f"dvfs_profile_{gpu_name}_idle.csv"
                if os.path.exists(idle_csv):
                    df_idle = pd.read_csv(idle_csv)
                    if "energy_consumption" in df_idle.columns:
                        df_idle["energy_consumption"] = (
                            df_idle["energy_consumption"] / 1000)
                    if ("energy_consumption" in df_idle.columns
                            and "time_taken" in df_idle.columns):
                        df_idle["power"] = (df_idle["energy_consumption"] /
                                            df_idle["time_taken"])
                    else:
                        raise ValueError(
                            "Idle CSV missing required columns for power calculation"
                        )
                    if "clock" in df_idle.columns and (df_idle["clock"]
                                                       == clock).any():
                        return df_idle[df_idle["clock"] ==
                                       clock]["power"].mean()
                    else:
                        return df_idle["power"].mean()
                else:
                    raise ValueError(f"Idle CSV not found for {gpu_name}")

            # --- Prepare subplots: one column per bound_size, one decode_batch_size per stacked block ---
            ncols = len(bound_sizes)
            n_decode = len(decode_batch_sizes)
            if SHOW_TIME:
                nrows = 2 * n_decode  # 2 rows (energy, time) per decode_batch_size
                height_ratios = [2, 1] * n_decode
            else:
                nrows = n_decode  # Only one row (energy) per decode_batch_size
                height_ratios = [2] * n_decode

            # --- Find the union of all available clocks across all data ---
            all_clocks_set = set()
            for clock in df["clock"].unique():
                all_clocks_set.add(clock)
            all_clocks = sorted(all_clocks_set)

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(7 * ncols, 5 * nrows),
                sharex=True,  # <--- All have common x-axis
                squeeze=False,
                gridspec_kw={"height_ratios": height_ratios},
            )

            # Set the common x-axis limit for all subplots to [0, 2100]
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xlim(0, 2100)

            # --- Pass 1: Collect all energy values to determine global y-limits ---
            # We'll collect all energies for all decode_batch_sizes and all bounds
            all_decode_energies = []
            all_idle_energies = []
            all_total_energies = []
            all_dilation_energies = []

            # For each decode_batch_size, store per-bound energy data
            all_per_decode_bound_energy_data = []

            # For storing percentage savings for each subplot
            percent_savings_matrix = [[None for _ in bound_sizes]
                                      for _ in decode_batch_sizes]

            for decode_idx, decode_batch_size in enumerate(decode_batch_sizes):
                per_bound_energy_data = []
                for col_idx, bound_size in enumerate(bound_sizes):
                    num_chunks = PREFILL_SIZE // bound_size
                    # if decode_batch_size larger than bound_size, skip
                    if decode_batch_size > bound_size:
                        per_bound_energy_data.append(None)
                        print(
                            f"Decode batch size {decode_batch_size} larger than bound size {bound_size}, skipping bound_size={bound_size}, decode_batch_size={decode_batch_size}"
                        )
                        continue

                    # Use only standard clock rows for bound time
                    try:
                        bound_time = get_time_taken(
                            bound_size,
                            standard_clock) - 0.01  # 20ms overhead two sides
                    except Exception as e:
                        print(
                            f"Could not get bound time for {bound_size} at standard clock {standard_clock}: {e}"
                        )
                        per_bound_energy_data.append(None)
                        continue

                    # Use the global all_clocks for x-axis
                    available_clocks = all_clocks

                    # Prepare lists for plotting, but only for clocks where decode_time is valid
                    decode_times = []
                    decode_energies = []
                    idle_energies = []
                    total_energies = []
                    idle_times = []
                    dilation_energies = []

                    no_dvfs = False

                    # if bound_time is 0, skip
                    if bound_time < 0:
                        print(
                            f"Bound time is negative ({bound_time:.4f}s), NODVFS bound_size={bound_size}, decode_batch_size={decode_batch_size}"
                        )
                        per_bound_energy_data.append(None)
                        no_dvfs = True

                    for clock in available_clocks:
                        if no_dvfs:
                            # only continue at standard clock
                            if clock != standard_clock:
                                # Still append NaN for missing clocks to keep x-axis aligned
                                decode_times.append(float('nan'))
                                decode_energies.append(float('nan'))
                                idle_energies.append(float('nan'))
                                total_energies.append(float('nan'))
                                idle_times.append(float('nan'))
                                dilation_energies.append(float('nan'))
                                continue
                            else:
                                print(
                                    f"No DVFS for bound_size={bound_size}, decode_batch_size={decode_batch_size}"
                                )

                        try:
                            decode_time = get_time_taken(
                                decode_batch_size, clock)
                            # if decode-time exceeds bound-time, skip
                            if decode_time > bound_time and clock != standard_clock:
                                # Still append NaN for missing clocks to keep x-axis aligned
                                decode_times.append(float('nan'))
                                decode_energies.append(float('nan'))
                                idle_energies.append(float('nan'))
                                total_energies.append(float('nan'))
                                idle_times.append(float('nan'))
                                dilation_energies.append(float('nan'))
                                continue

                            decode_energy = get_energy(decode_batch_size,
                                                       clock)
                            idle_pwr = get_idle_power(gpu_name, clock)
                            idle_time = max(bound_time - decode_time, 0)
                            idle_energy = idle_pwr * idle_time
                            total_energy = decode_energy + idle_energy

                            if CONSIDER_DECODE_DEBT:
                                decode_time = decode_time * num_chunks
                                decode_energy = decode_energy * num_chunks
                                idle_energy = idle_energy * num_chunks
                                total_energy = total_energy * num_chunks

                                # Next up, I want to add dilation energy penalty.
                                num_chunks_not_decoded = NUM_DECODES - num_chunks
                                # Add dilation energy penalty, consider each decod is run at standard clock
                                dilation_energy = get_energy(
                                    decode_batch_size, standard_clock
                                ) * num_chunks_not_decoded * DECODE_DEBT_FACTOR
                                total_energy += dilation_energy
                            else:
                                dilation_energy = 0

                            decode_times.append(decode_time)
                            decode_energies.append(decode_energy)
                            idle_energies.append(idle_energy)
                            total_energies.append(total_energy)
                            idle_times.append(idle_time)
                            dilation_energies.append(dilation_energy)
                        except Exception as e:
                            print(
                                f"Skipping clock {clock} for bound_size={bound_size}, decode_batch_size={decode_batch_size} due to error: {e}"
                            )
                            # Still append NaN for missing clocks to keep x-axis aligned
                            decode_times.append(float('nan'))
                            decode_energies.append(float('nan'))
                            idle_energies.append(float('nan'))
                            total_energies.append(float('nan'))
                            idle_times.append(float('nan'))
                            dilation_energies.append(float('nan'))
                            continue

                    # Save for global y-limits
                    all_decode_energies.extend(
                        [v for v in decode_energies if pd.notna(v)])
                    all_idle_energies.extend(
                        [v for v in idle_energies if pd.notna(v)])
                    all_total_energies.extend(
                        [v for v in total_energies if pd.notna(v)])
                    all_dilation_energies.extend(
                        [v for v in dilation_energies if pd.notna(v)])

                    # --- Calculate percentage of energy saved via DVFS for this subplot ---
                    # Find total energy at standard clock (if available)
                    try:
                        std_idx = available_clocks.index(standard_clock)
                        std_total_energy = total_energies[std_idx]
                    except Exception:
                        std_total_energy = None

                    # Find minimum total energy (DVFS optimal)
                    total_energies_valid = [
                        v for v in total_energies if pd.notna(v)
                    ]
                    if total_energies_valid:
                        min_total_energy = min(total_energies_valid)
                    else:
                        min_total_energy = None

                    percent_saving = None
                    if std_total_energy is not None and min_total_energy is not None and std_total_energy > 0:
                        percent_saving = 100.0 * (
                            std_total_energy -
                            min_total_energy) / std_total_energy
                    percent_savings_matrix[decode_idx][
                        col_idx] = percent_saving

                    per_bound_energy_data.append({
                        "available_clocks":
                        available_clocks,
                        "decode_energies":
                        decode_energies,
                        "idle_energies":
                        idle_energies,
                        "total_energies":
                        total_energies,
                        "decode_times":
                        decode_times,
                        "idle_times":
                        idle_times,
                        "bound_time":
                        bound_time,
                        "decode_batch_size":
                        decode_batch_size,
                        "bound_size":
                        bound_size,
                        "percent_saving":
                        percent_saving,
                        "std_total_energy":
                        std_total_energy,
                        "min_total_energy":
                        min_total_energy,
                        "dilation_energies":
                        dilation_energies,
                    })
                all_per_decode_bound_energy_data.append(per_bound_energy_data)

            # Compute global y-limits for energy axes
            all_energy_vals = all_decode_energies + all_idle_energies + all_total_energies + all_dilation_energies
            if all_energy_vals:
                global_energy_min = min(0, min(all_energy_vals))
                global_energy_max = max(all_energy_vals)
            else:
                global_energy_min = 0
                global_energy_max = 1

            # --- Pass 2: Plot using shared y-axis for energy ---
            for decode_idx, decode_batch_size in enumerate(decode_batch_sizes):
                # Defensive: check if decode_idx is in range of all_per_decode_bound_energy_data
                if decode_idx >= len(all_per_decode_bound_energy_data):
                    print(
                        f"decode_idx {decode_idx} out of range for all_per_decode_bound_energy_data (len={len(all_per_decode_bound_energy_data)})"
                    )
                    continue
                per_bound_energy_data = all_per_decode_bound_energy_data[
                    decode_idx]
                for col_idx, bound_size in enumerate(bound_sizes):
                    # Defensive: check if col_idx is in range of per_bound_energy_data
                    if col_idx >= len(per_bound_energy_data):
                        print(
                            f"col_idx {col_idx} out of range for per_bound_energy_data (len={len(per_bound_energy_data)})"
                        )
                        continue
                    data = per_bound_energy_data[col_idx]
                    if data is None:
                        # tell what is none
                        print(
                            f"data is None for bound_size={bound_size}, decode_batch_size={decode_batch_size}"
                        )
                        continue

                    available_clocks = data["available_clocks"]
                    decode_energies = data["decode_energies"]
                    idle_energies = data["idle_energies"]
                    total_energies = data["total_energies"]
                    decode_times = data["decode_times"]
                    idle_times = data["idle_times"]
                    bound_time = data["bound_time"]
                    percent_saving = data.get("percent_saving", None)
                    std_total_energy = data.get("std_total_energy", None)
                    min_total_energy = data.get("min_total_energy", None)
                    dilation_energies = data.get("dilation_energies", [])

                    # Find the energy-minimal point (minimum total energy)
                    total_energies_valid = [
                        v for v in total_energies if pd.notna(v)
                    ]
                    if total_energies_valid:
                        try:
                            min_energy_idx = int(
                                pd.Series(total_energies).idxmin())
                            # Defensive: check index range
                            if min_energy_idx < 0 or min_energy_idx >= len(
                                    available_clocks):
                                print(
                                    f"min_energy_idx {min_energy_idx} out of range for available_clocks (len={len(available_clocks)})"
                                )
                                min_energy_clock = None
                                min_energy_val = None
                            else:
                                min_energy_clock = available_clocks[
                                    min_energy_idx]
                                min_energy_val = total_energies[min_energy_idx]
                                if pd.isna(min_energy_val):
                                    min_energy_clock = None
                                    min_energy_val = None
                        except Exception as e:
                            print(f"Error finding min_energy_idx: {e}")
                            min_energy_idx = None
                            min_energy_clock = None
                            min_energy_val = None
                    else:
                        min_energy_idx = None
                        min_energy_clock = None
                        min_energy_val = None

                    # --- Top plot: Energy ---
                    if SHOW_TIME:
                        ax1 = axes[2 * decode_idx, col_idx]
                    else:
                        ax1 = axes[decode_idx, col_idx]
                    color1 = "tab:blue"
                    color2 = "tab:orange"
                    color3 = "tab:green"
                    color4 = "tab:purple"  # For dilation energy

                    ax1.set_ylabel("Energy (Joules)")
                    ax1.plot(
                        available_clocks,
                        decode_energies,
                        marker="o",
                        label="Decode Energy (J)",
                        linewidth=3,
                        color=color1,
                    )
                    ax1.plot(
                        available_clocks,
                        idle_energies,
                        marker="s",
                        label="Idle Energy (J)",
                        linewidth=3,
                        color=color2,
                    )
                    ax1.plot(
                        available_clocks,
                        total_energies,
                        marker="^",
                        label="Total Energy (J)",
                        linewidth=3,
                        color=color3,
                    )
                    # Plot dilation energies
                    ax1.plot(
                        available_clocks,
                        dilation_energies,
                        marker="*",
                        label="Dilation Energy (J)",
                        linewidth=2,
                        color=color4,
                        linestyle="dashed",
                    )

                    # Highlight the energy-minimal point with a red dot and text (clock)
                    if min_energy_idx is not None and min_energy_clock is not None and min_energy_val is not None:
                        ax1.plot(
                            min_energy_clock,
                            min_energy_val,
                            marker="o",
                            color="red",
                            markersize=12,
                            label="Energy-Minimal Point",
                            linestyle="None",
                        )
                        ax1.annotate(
                            f"Min @ {min_energy_clock} MHz\n{min_energy_val:.2f} J",
                            xy=(min_energy_clock, min_energy_val),
                            xytext=(min_energy_clock, min_energy_val *
                                    1.05 if min_energy_val > 0 else 0.1),
                            textcoords="data",
                            ha="center",
                            color="red",
                            fontsize=12,
                            fontweight="bold",
                            arrowprops=dict(facecolor="red",
                                            shrink=0.05,
                                            width=2,
                                            headwidth=8),
                        )
                    ax1.set_ylim(global_energy_min, global_energy_max * 1.05)
                    ax1.set_xlim(0, 2100)
                    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
                    if col_idx == 0:
                        ax1.legend(loc="best")

                    # --- Add percentage of energy saved via DVFS on top of each graph ---
                    # Only show if both std_total_energy and min_total_energy are available and std_total_energy > 0
                    if percent_saving is not None:
                        percent_str = f"DVFS Energy Saved: {percent_saving:.1f}%"
                    else:
                        percent_str = "DVFS Energy Saved: N/A"
                    # Place the annotation at the top center of the axes
                    ax1.annotate(
                        percent_str,
                        xy=(0.5, 0.5),
                        xycoords="axes fraction",
                        ha="center",
                        va="bottom",
                        fontsize=13,
                        color="darkgreen"
                        if percent_saving is not None else "gray",
                        fontweight="bold",
                    )

                    ax1.set_title(
                        f"Bound Size={bound_size}\nDecode Batch={decode_batch_size}\nBound Time={bound_time:.2f}s"
                    )

                    # --- Bottom plot: Time ---
                    if SHOW_TIME:
                        ax2 = axes[2 * decode_idx + 1, col_idx]
                        color5 = "tab:red"
                        color6 = "tab:cyan"
                        ax2.plot(
                            available_clocks,
                            decode_times,
                            marker="d",
                            label="Decode Time (s)",
                            linewidth=2,
                            color=color5,
                            linestyle="--",
                        )
                        ax2.plot(
                            available_clocks,
                            idle_times,
                            marker="x",
                            label="Idle Time (s)",
                            linewidth=2,
                            color=color6,
                            linestyle="--",
                        )
                        ax2.set_ylabel("Time (Seconds)")
                        ax2.set_xlabel("Clock Frequency (MHz)")
                        ax2.set_ylim(bottom=0)
                        ax2.set_xlim(0, 2100)
                        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
                        if col_idx == 0:
                            ax2.legend(loc="best")

            # Set a super title for the whole figure
            if SHOW_TIME:
                fig.suptitle(
                    f"Clock vs Energy and Time for {model_name} on {gpu_name}\nDecode Batch Sizes: {decode_batch_sizes}",
                    fontsize=16,
                    fontweight="bold",
                    y=1.02,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(
                    f"clock_vs_energy_and_time_allbounds_decodeALL_{model_name}_{gpu_name}.png",
                    bbox_inches="tight",
                )
            else:
                fig.suptitle(
                    f"Clock vs Energy for {model_name} on {gpu_name}\nDecode Batch Sizes: {decode_batch_sizes}",
                    fontsize=16,
                    fontweight="bold",
                    y=1.02,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(
                    f"clock_vs_energy_allbounds_decodeALL_{model_name}_{gpu_name}.png",
                    bbox_inches="tight",
                )
            plt.close()
