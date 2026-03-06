#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import datetime
import signal
import sys
import time

import pandas as pd
from pynvml import (NVML_CLOCK_GRAPHICS, nvmlDeviceGetClockInfo,
                    nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetTotalEnergyConsumption, nvmlInit,
                    nvmlShutdown)

running = True

model_name = "huggyllama/llama-13b"
model_name_cleaned = model_name.replace("/", "_")


def signal_handler(sig, frame):
    """
    Handles termination signals (SIGINT, SIGTERM).
    Sets the running flag to False, breaking the main loop.
    """
    global running
    running = False
    print("\nTermination signal received. Stopping data collection...")


def main(result_dir: str):

    # Initialize NVML
    nvmlInit()

    # Get the number of GPUs
    device_count = nvmlDeviceGetCount()

    # Create column names for the DataFrame
    columns = ["timestamp"]
    for i in range(device_count):
        columns.extend([f"gpu_{i}_joules", f"gpu_{i}_freq_mhz"])

    # Prepare a list to hold per-sample data dictionaries
    data_records = []

    # Sampling interval in seconds
    sampling_interval = 0.2

    # Attach the custom signal handler for graceful shutdown
    # Handle both SIGINT (Ctrl+C) and SIGTERM (kill from another process)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(
        f"Starting GPU energy and frequency monitoring for {device_count} GPU(s)"
    )
    print("Collecting: energy (Joules), frequency (MHz)")
    print(f"Sampling interval: {sampling_interval}s")

    # Main data collection loop
    while running:
        start_time = datetime.datetime.now()

        # Build a data record (dictionary) for this sample
        record = {"timestamp": start_time}

        # For each GPU, measure energy consumption and current frequency
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)

            # Get energy consumption in millijoules
            energy_mili_joules = nvmlDeviceGetTotalEnergyConsumption(handle)

            # Get current graphics clock frequency in MHz
            graphics_clock_mhz = nvmlDeviceGetClockInfo(
                handle, NVML_CLOCK_GRAPHICS)

            # Add to record
            record[f"gpu_{i}_joules"] = energy_mili_joules
            record[f"gpu_{i}_freq_mhz"] = graphics_clock_mhz

        # Store the record
        data_records.append(record)

        # Sleep for the remainder of the sampling interval
        elapsed = datetime.datetime.now() - start_time
        elapsed_seconds = elapsed.total_seconds()
        time_to_sleep = sampling_interval - elapsed_seconds
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    # Once the loop ends, create DataFrame and save to CSV
    if data_records:
        df = pd.DataFrame(data_records, columns=columns)
        filename = f"{result_dir}/gpu_energy_and_frequency_{model_name_cleaned}.csv"

        try:
            df.to_csv(filename, index=False)
            print(f"Data successfully saved to: {filename}")
            print(f"Collected {len(df)} samples across {device_count} GPU(s)")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    else:
        print("No data collected.")

    # Shutdown NVML
    nvmlShutdown()
    print("Exiting gracefully.")


if __name__ == "__main__":
    # Get result directory as an argument
    result_dir = sys.argv[1]
    main(result_dir)
