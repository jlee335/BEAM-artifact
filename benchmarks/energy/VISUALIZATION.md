# Visualizing Results

Three visualization scripts are provided to inspect artifact evaluation outputs. Each script targets a specific experiment type.

---

## Script Overview

| Script | Used After | Output |
|--------|-----------|--------|
| `visualization/visualize_multiple_traces.py` | `run_e2e.sh`, `run_ablation.sh` | Bar charts + distribution histograms comparing configurations |
| `visualization/visualize_timeline.py` | `run_burst.sh`, `run_stair.sh` | Windowed timeline plots of TTFT, TBT, and energy over time |
| `visualization/visualize_fidelity.py` | `run_fidelity.sh` | Actual vs. estimated TTFT and TBT/ITL scatter/error plots |

---

## 1. `visualize_multiple_traces.py` — Bar / Distribution Comparison

**Use after:** `run_e2e.sh`, `run_ablation.sh`

Compares energy consumption, TTFT, TBT, and TPOT across multiple experiment configurations side-by-side. Produces two figures:
- `comparison_visualizations/bar.png` — bar chart of mean, P90, and % under SLO for each config
- `comparison_visualizations/distributions.png` — per-config histograms with SLO reference lines

**Usage:**
```bash
python3 visualization/visualize_multiple_traces.py <result_dir> [--mode auto|single|multiple]
```

| Argument | Description |
|----------|-------------|
| `result_dir` | Timestamped result directory (e.g., `end_to_end/dataset_20260304_120000/`) |
| `--mode` | `auto` (default): detect single vs. multi-trace automatically; `single`: one config only; `multiple`: force multi-config comparison |

**Examples:**
```bash
# After run_e2e.sh — compare Vanilla, DynamoLLM, S1+S2
python3 visualization/visualize_multiple_traces.py \
    end_to_end/dataset_20260304_120000/

# After run_ablation.sh — compare all 5 ablation configs
python3 visualization/visualize_multiple_traces.py \
    sensitivity_tests/ablation/ablation_study_20260304_130000/

# Single config inspection
python3 visualization/visualize_multiple_traces.py \
    end_to_end/dataset_20260304_120000/D_s1_s2_.../ \
    --mode single
```

**Expected output directory layout** (the script expects subdirectories each containing a `traced_dataset/` folder with a `.json` result file):
```
end_to_end/dataset_YYYYMMDD_HHMMSS/
├── A_vanilla_.../
│   └── traced_dataset/
│       └── *.json
├── B2_dynamo_.../
│   └── traced_dataset/
│       └── *.json
└── D_s1_s2_.../
    └── traced_dataset/
        └── *.json
```

---

## 2. `visualize_timeline.py` — Windowed Timeline

**Use after:** `run_burst.sh`, `run_stair.sh`

Shows how TTFT, TBT, and GPU energy evolve over time using sliding windows. Useful for observing how each scheduling approach responds to load changes (e.g., the burst spike or the staircase ramp).

**Usage:**
```bash
python3 visualization/visualize_timeline.py <parent_dir> [--window-size <seconds>] [--save <path>]
```

| Argument | Description |
|----------|-------------|
| `parent_dir` | Timestamped result directory containing experiment subdirs (A, B, C, D, ...) |
| `--window-size` | Aggregation window in seconds (default: `3.0`) |
| `--save` | Output file path (default: auto-generated inside `parent_dir`) |

**Examples:**
```bash
# After run_burst.sh
python3 visualization/visualize_timeline.py \
    sensitivity_tests/burst/burst_study_20260304_140000/ \
    --window-size 5.0

# After run_stair.sh — finer windows to capture ramp transitions
python3 visualization/visualize_timeline.py \
    sensitivity_tests/stair/stair_study_20260304_150000/ \
    --window-size 3.0 \
    --save stair_timeline.png
```

**Expected output directory layout** (the script reads `batch_log_GPU_0.csv` and `gpu_energy_and_frequency_*.csv` from each subdirectory):
```
sensitivity_tests/burst/burst_study_YYYYMMDD_HHMMSS/
├── A_vanilla_.../
│   ├── batch_log_GPU_0.csv
│   ├── gpu_energy_and_frequency_*.csv
│   └── *.json
├── B2_dynamo_.../
│   └── ...
└── D_s1_s2_.../
    └── ...
```

---

## 3. `visualization/visualize_fidelity.py` — Energy Model Fidelity

**Use after:** `run_fidelity.sh`

Compares the energy model's estimated TTFT / TBT against actual measured values. Produces:
- `ttft_comparison.png` — request-by-request timeline, scatter plot, and error distribution for TTFT
- `tbt_itl_comparison.png` — token-level timeline, scatter plot, and error distribution for TBT/ITL

**Usage:**
```bash
python3 visualization/visualize_fidelity.py <fidelity_dir> [--output-dir <dir>] [--itl-all]
```

| Argument | Description |
|----------|-------------|
| `fidelity_dir` | Fidelity test result directory (e.g., `sensitivity_tests/fidelity/fidelity_test_YYYYMMDD_HHMMSS/`) |
| `--output-dir` | Where to save plots (default: same as `fidelity_dir`) |
| `--itl-all` | Plot all concatenated token indices instead of only the longest single request in the TBT timeline |

**Example:**
```bash
python3 visualization/visualize_fidelity.py \
    sensitivity_tests/fidelity/fidelity_test_20260304_160000/ \
    --output-dir sensitivity_tests/fidelity/fidelity_test_20260304_160000/plots/
```

**Expected output directory layout** (the script pairs `*.json` with `*_estimations.json` files):
```
sensitivity_tests/fidelity/fidelity_test_YYYYMMDD_HHMMSS/
└── <config_dir>/
    └── traced_dataset/
        ├── traced_dataset.json          ← actual measured results
        └── traced_dataset_estimations.json  ← model estimates
```

---

## Quick Reference

```bash
# E2E / ablation results → bar + distribution comparison
python3 visualization/visualize_multiple_traces.py <result_dir>

# Burst / stair results → timeline over time
python3 visualization/visualize_timeline.py <result_dir> --window-size 3.0

# Fidelity results → actual vs. estimated metrics
python3 visualization/visualize_fidelity.py <fidelity_dir>
```
