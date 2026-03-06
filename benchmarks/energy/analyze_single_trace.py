import pandas as pd
import re
import json
import os
import glob


# Create a class TraceData
class TraceData:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.num_gpus = get_num_gpus(base_directory)
        self.ttft_slo, self.tbt_slo = get_slo(base_directory)
        self.start_time, self.finish_time = get_start_finish_time(base_directory)
        energy_csvs = glob.glob(f"{base_directory}/traced_dataset/gpu_energy_and_frequency_*.csv")
        if not energy_csvs:
            raise ValueError(f"No gpu_energy_and_frequency_*.csv found in {base_directory}/traced_dataset/")
        self.energy_consumption = get_energy_consumption(
            energy_csvs[0],
            self.start_time,
            self.finish_time,
            self.num_gpus,
        )
        self.ttfts, self.tpots, self.tbts, self.total_duration = analyze_ttft_tbt_duration(
            base_directory, self.ttft_slo, self.tbt_slo
        )


def get_num_gpus(base_directory):
    # Find all batch_log_GPU_X.csv files and get the largest X
    batch_log_files = glob.glob(f"{base_directory}/traced_dataset/batch_log_GPU_*.csv")
    if not batch_log_files:
        raise ValueError(f"No batch_log_GPU_*.csv files found in {base_directory}/traced_dataset/")
    
    # Extract GPU numbers from filenames
    gpu_numbers = []
    for file in batch_log_files:
        # Extract the number from batch_log_GPU_X.csv
        filename = os.path.basename(file)
        gpu_num = int(filename.split("_")[-1].replace(".csv", ""))
        gpu_numbers.append(gpu_num)
    
    # Number of GPUs is the largest GPU number + 1 (since indexing starts at 0)
    return max(gpu_numbers) + 1


def get_energy_consumption(gpu_energy_csv_path, start_time, end_time, num_gpus):
    # timestamp,gpu_0_joules,gpu_0_freq_mhz,gpu_1_joules,gpu_1_freq_mhz,gpu_2_joules,gpu_2_freq_mhz,gpu_3_joules,gpu_3_freq_mhz
    df = pd.read_csv(gpu_energy_csv_path)
    # 1. Cut off timestamp before start_time and after end_time
    df = df[df["timestamp"] >= start_time]
    df = df[df["timestamp"] <= end_time]
    # 2. Since the gpu_joules is cumulative, subtract last to first to get the energy consumption
    energy_consumption = []
    for i in range(num_gpus):
        energy_consumption.append(
            df[f"gpu_{i}_joules"].iloc[-1] - df[f"gpu_{i}_joules"].iloc[0]
        )
    sum_energy_consumption = sum(energy_consumption)
    # Convert mJ to J
    sum_energy_consumption = sum_energy_consumption / 1000
    return sum_energy_consumption


def get_slo(base_directory):
    # return tuple of ttft, tbt/tpot
    # read dataset_info.txt
    # If SLO_X_<TBT>_<TTFT> format found in base_directory, return the values
    # Otherwise, return the values from the file

    # regex match base directory to find SLO_<any alphabet>_<TBT>_<TTFT> format
    match = re.search(r"SLO_([A-Za-z])_(\d+\.\d+)_(\d+\.\d+)", base_directory)
    if match:
        slo_letter = match.group(1)
        tbt_slo = float(match.group(2))
        ttft_slo = float(match.group(3))
        return ttft_slo, tbt_slo
    else:
        import os

        dataset_info_path = os.path.join(base_directory, "..", "dataset_info.txt")
        slo_info_path = os.path.join(base_directory, "slo_info.txt")
        ttft_slo = None
        tbt_slo = None
        
        print(slo_info_path)

        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, "r") as f:
                for line in f:
                    if "TTFT_SLO" in line:
                        ttft_slo = line.split(":")[1].strip()
                        # remove "s" suffix and make it float
                        ttft_slo = float(ttft_slo.replace("s", ""))
                    if "TBT_SLO" in line:
                        tbt_slo = line.split(":")[1].strip()
                        # remove "s" suffix and make it float
                        tbt_slo = float(tbt_slo.replace("s", ""))
        elif os.path.exists(slo_info_path):
            with open(slo_info_path, "r") as f:
                for line in f:
                    # Try to find lines like "TBT: 0.15s" or "TTFT: 1.0s"
                    
                    if "TBT:" in line:
                        tbt_slo = line.split(":")[1].strip()
                        # remove "s" suffix and make it float
                        tbt_slo = float(tbt_slo.replace("s", ""))
                    elif "TTFT:" in line:
                        ttft_slo = line.split(":")[1].strip()
                        # remove "s" suffix and make it float
                        ttft_slo = float(ttft_slo.replace("s", ""))
        print(ttft_slo, tbt_slo)
        if ttft_slo is None or tbt_slo is None:
            raise ValueError("Could not parse TTFT_SLO and/or TBT_SLO from dataset_info.txt or slo_info.txt")
    return ttft_slo, tbt_slo


def get_start_finish_time(base_directory):
    # use batch_log_GPU_0.csv to get start_time and finish_time
    # step,phase,elapsed_time,batch_size,gpu_clock,start_time,current_time,curr_energy
    df = pd.read_csv(f"{base_directory}/traced_dataset/batch_log_GPU_0.csv")
    # first_time is first "current_time", last_time is last "current_time"
    start_time = df["current_time"].iloc[0]
    finish_time = df["current_time"].iloc[-1]
    return start_time, finish_time


def analyze_ttft_tbt_duration(base_directory, ttft_slo, tbt_slo):
    # parse only json file in /traced_dataset/
    for file in os.listdir(f"{base_directory}/traced_dataset/"):
        if file.endswith(".json") and "_estimations" not in file:
            with open(f"{base_directory}/traced_dataset/{file}", "r") as f:
                data = json.load(f)
                break

    num_requests = data["completed"]
    ttfts = data["ttfts"]
    tbts = data["itls"]  # list of lists
    e2els = data["e2el"]
    tpots = []
    # calculate tpot for each request by subtracting ttft from e2el
    for i in range(num_requests):
        if len(tbts[i]) > 1:
            tpot = (e2els[i] - ttfts[i]) / (len(tbts[i]) - 1)
            tpots.append(tpot)
        
    total_duration = data["duration"]
    
    return ttfts, tpots, tbts, total_duration


if __name__ == "__main__":
    base_directory = "/workspace/disagg/energy-inf-v1-disagg/benchmarks/energy/profiler_results/dataset_comprehensive_test_20251010_150400/fixed_chunk_128_clock_1830"
    trace_data = TraceData(base_directory)
    print(f"Number of GPUs detected: {trace_data.num_gpus}")
    print(f"Energy consumption: {trace_data.energy_consumption}")
    print(f"TTFTs: {trace_data.ttfts}")
    print(f"TPOTs: {trace_data.tpots}")
    print(f"TBTs: {trace_data.tbts}")
