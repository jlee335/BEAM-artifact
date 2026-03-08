# BEAM: Energy-Efficient LLM Inference Scheduling

This repository contains the artifact for **BEAM**, an energy-aware inference scheduling system for LLM serving. It is built on top of [vLLM](https://github.com/vllm-project/vllm) (v0.11+) with a modified V1 scheduler that implements DVFS-based scheduling algorithms (S1 and S2) to minimize GPU energy consumption while satisfying latency SLOs.

## Hardware & Software Requirements

- **GPU Setup**: 1-8 GPUs (A100 recommended), depending on your target parallelism configuration. The `run_*.sh` scripts derive GPU count automatically via `nvidia-smi`.
  - **Paper evaluation setup**: 8× A100-80GB SXM GPUs, running `meta-llama/Llama-3.3-70B-Instruct` with `--tp 2 --pp 4`.
- **System Requirements**:
  - Docker and Docker Compose (v2) installed.
  - NVIDIA Container Toolkit installed and properly mapped.
  - At least 200 GB disk space for model weights (typically stored in your Hugging Face cache).

## Setup Instructions

We use `docker-compose` to start the evaluator environment:

1. **Set your Hugging Face token** (required to download model weights):
   ```bash
   export HF_TOKEN=<your_huggingface_token>
   ```

2. **Start the Evaluator Container** — run from the **repository root**:
   ```bash
   docker compose up -d
   ```
   *Note: On the first run, this builds the Docker image from `docker/Dockerfile`, compiling the custom vLLM C++ extensions from source. This takes several minutes. Once built, the container automatically installs monitoring tools (`pynvml`, `bc`, etc.) at startup.*

3. **Enter the Container**:
   ```bash
   docker exec -it -w /workspace/benchmarks/energy vllm_evaluator_env bash
   ```

## Reproducing Results

**All benchmarks must be run inside the container** from `/workspace/benchmarks/energy`.

Each script orchestrates server boot, GPU clock-locking, dynamic `--use-s1` / `--use-s2` scheduling, and energy profiling automatically via `utils.sh`.

### Quick-Start: Full Pipeline (Recommended)

The simplest way to run the complete artifact evaluation is the unified `run_all.sh` script:

```bash
./run_all.sh \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tp 2 --pp 4 \
  --dataset-path /workspace/benchmarks/energy/datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv
```

This runs all steps below in sequence and exits on the first failure.

---

### Step-by-Step Execution

If you prefer to run individual steps:

#### 1. Offline Profiling (System Profiles)

Generates energy/latency lookup tables for your GPU and model configuration:
```bash
./run_offline_profile.sh --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4
```
Output: `offline_profile_results/dvfs_profile_<GPU>_<model>_tp<n>_pp<n>_one.csv`

#### 2. Offline Profiling (DynamoLLM Profiles)

Generates baseline profiles following the DynamoLLM paper methodology:
```bash
./run_dynamo_benchmark.sh --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4
```
Output: `dynamollm_profiles/dynamo_dvfs_profile_<GPU>_<model>.csv`

#### 3. Main End-to-End Evaluation

Tests the three main configurations (Vanilla vLLM, DynamoLLM, S1+S2) on the traced dataset:
```bash
./run_e2e.sh \
  --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4 \
  --dataset-path /workspace/benchmarks/energy/datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv
```
Output: `end_to_end/<dataset>_YYYYMMDD_HHMMSS/`

#### 4. Sensitivity Studies

- **Full Ablation Study** (Vanilla / DynamoLLM / S1-DVFS-only / S1-only / S1+S2):
  ```bash
  ./run_ablation.sh --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4
  ```

- **Burst Pattern Study** (L-L-H-L-L workload):
  ```bash
  ./run_burst.sh --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4
  ```

- **Stair Pattern Study** (gradually increasing load):
  ```bash
  ./run_stair.sh --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4
  ```

- **Fidelity Metrics**:
  ```bash
  ./run_fidelity.sh \
    --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --pp 4 
  ```

- **Energy-Performance Pareto Frontier** *(optional, not included in `run_all.sh`)*:

  Use `run_pareto_pipeline.sh` for the **full self-contained Pareto pipeline** (requires exactly 8 GPUs). It automatically runs offline profiling, DynamoLLM profiling, the Pareto sweep, and visualization for both `tp4_pp2` and `tp2_pp4` configurations:
  ```bash
  ./run_pareto_pipeline.sh --model meta-llama/Llama-3.3-70B-Instruct
  ```
  Pipeline steps executed internally:
  1. Offline profiling — `tp4_pp2` and `tp2_pp4`
  2. DynamoLLM profiling — `tp4_pp2` and `tp2_pp4`
  3. Pareto sweep — `run_pareto.sh`
  4. Visualization — `sensitivity_tests/visualize_pareto.py`

  Output plots and CSV are saved under the latest `sensitivity_tests/pareto/pareto_analysis_*/` directory.

  If you only want the sweep step (profiling already done), run `run_pareto.sh` directly:
  ```bash
  ./run_pareto.sh --model meta-llama/Llama-3.3-70B-Instruct
  ```

---

## Expected Runtimes

The following are approximate wall-clock times per step on the paper evaluation setup (8× A100-80GB SXM, `meta-llama/Llama-3.3-70B-Instruct`, `--tp 2 --pp 4`):

| Step | Script | Approx. Time |
|------|--------|-------------|
| System profiling | `run_offline_profile.sh` | ~30-60 min |
| DynamoLLM profiling | `run_dynamo_benchmark.sh` | ~60-90 min |
| End-to-end ablation | `run_e2e.sh` | ~30 min |
| Full ablation study | `run_ablation.sh` | ~60 min |
| Burst study | `run_burst.sh` | ~20 min |
| Stair study | `run_stair.sh` | ~20 min |
| Fidelity test | `run_fidelity.sh` | ~20 min |
| **Full pipeline** | `run_all.sh` | **~5-6 hours** |

---

## Visualizing Results

Three scripts are provided to plot results after benchmarks complete:

| Script | Used After | What It Shows |
|--------|-----------|---------------|
| `visualization/visualize_multiple_traces.py` | `run_e2e.sh`, `run_ablation.sh` | Bar charts + histograms comparing energy & latency across configs |
| `visualization/visualize_timeline.py` | `run_burst.sh`, `run_stair.sh` | Windowed timeline of TTFT, TBT, and energy over time |
| `visualization/visualize_fidelity.py` | `run_fidelity.sh` | Actual vs. estimated TTFT and TBT/ITL scatter and error plots |

**Quick examples:**
```bash
# After run_e2e or run_ablation
python3 visualization/visualize_multiple_traces.py end_to_end/dataset_YYYYMMDD_HHMMSS/

# After run_burst or run_stair
python3 visualization/visualize_timeline.py sensitivity_tests/burst/burst_study_YYYYMMDD_HHMMSS/ --window-size 3.0

# After run_fidelity
python3 visualization/visualize_fidelity.py sensitivity_tests/fidelity/fidelity_test_YYYYMMDD_HHMMSS/
```

See [benchmarks/energy/VISUALIZATION.md](benchmarks/energy/VISUALIZATION.md) for full argument reference, expected directory layouts, and additional examples.

---

## Validating Outputs

Upon running any `run_*.sh` script, a timestamped directory is created (e.g., `end_to_end/dataset_YYYYMMDD_HHMMSS/` or `sensitivity_tests/burst/burst_study_YYYYMMDD_HHMMSS/`).

Inside each directory you will find:

1. `*_summary.txt` — configuration metadata, request rates, input lengths, and which tests were run.
2. Sub-directories per experiment configuration (e.g., `A_vanilla_...`, `D_s1_s2_...`).
3. `experiment_info.txt` — configuration mapping for the run.
4. `gpu_energy_and_frequency_<model>.csv` — per-GPU energy (J) and clock frequency measured by `gpu_energy_monitor.py`.
5. `batch_log_GPU_*.csv`, `dvfs_log_GPU_*.csv`, `s1_log.csv`, `s2_log.csv` — profiling logs (when `--profile` is active).
6. `*.json` — raw benchmark latency/throughput results.

See [benchmarks/energy/PROFILING.md](benchmarks/energy/PROFILING.md) for a detailed description of each output file and the two-layer profiling architecture.

---

## Repository Structure

```
.
├── benchmarks/energy/       # Artifact evaluation scripts and datasets
│   ├── run_all.sh           # Unified pipeline entry point
│   ├── run_e2e.sh           # End-to-end evaluation
│   ├── run_ablation.sh      # Full ablation study
│   ├── run_burst.sh / run_stair.sh / run_pareto.sh / run_fidelity.sh
│   ├── run_pareto_pipeline.sh  # Full self-contained Pareto pipeline (8 GPUs)
│   ├── datasets/            # Traced workload datasets (CSV)
│   ├── visualization/       # Plotting scripts
│   └── offline_profile_results/ / dynamollm_profiles/  # Generated profiles
├── vllm/v1/core/sched/
│   ├── scheduler.py         # Modified V1 scheduler (S1/S2 algorithms)
│   ├── energy_model.py      # EnergySimulator (prefill/decode modeling)
│   └── dynamo_energy_model.py  # DynamoLLM-style clock selector
├── docker-compose.yml       # Evaluator container definition
└── docker/Dockerfile        # Container image build file
```

## License

This project is a fork of [vLLM](https://github.com/vllm-project/vllm) and inherits its Apache 2.0 license.