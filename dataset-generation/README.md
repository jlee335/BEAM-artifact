# Dataset Generation

This directory contains a copy of [**alibaba/ServeGen**](https://github.com/alibaba/ServeGen), the workload generator used to produce the LLM inference request traces for BEAM artifact evaluation. The code here is taken directly from that repository (with one path fix in `generate.py` to match the layout of this directory).

ServeGen generates realistic request arrival patterns and token-length distributions by replaying anonymized production traces from real LLM serving deployments.

## Setup (one-time)

```bash
cd dataset-generation/
pip install -r requirements.txt
pip install -e .
```

## Reproducing the Primary Benchmark Dataset

The default arguments reproduce the exact dataset used in the BEAM end-to-end and ablation benchmarks:

```bash
python generate.py
# Produces: datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv
```

Copy the output to the benchmarks directory:

```bash
cp datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv \
   ../benchmarks/energy/datasets/
```

Because generation is seeded (`--seed 42` by default), the output is deterministic and will match the pre-committed dataset byte-for-byte.

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--category` | `language` | Workload category: `language`, `reason`, or `multimodal` |
| `--model` | `m-small` | Model size identifier (must match a subdirectory under `data/<category>/`) |
| `--rate` | `3.0` | Target aggregate request rate (requests per second) |
| `--start` | `19:00` | Window start time (`HH:MM`) |
| `--end` | `20:00` | Window end time (`HH:MM`) |
| `--day` | `1` | Day index within the trace (0-based) |
| `--duration` | *(end−start)* | Override window duration in seconds |
| `--output-dir` | `./datasets` | Directory to write the output CSV |
| `--seed` | `42` | RNG seed for reproducibility |
| `--force` | *(off)* | Regenerate even if output file already exists |

## Output CSV Format

Each row in the generated CSV represents one inference request:

| Column | Description |
|---|---|
| `TIMESTAMP` | UTC arrival timestamp (`YYYY-MM-DD HH:MM:SS.ffffff+0000`) |
| `ContextTokens` | Number of input (prompt) tokens |
| `GeneratedTokens` | Number of output (generated) tokens |
| `timestamp` | Duplicate of `TIMESTAMP` (required by the benchmark harness) |

## Data Directory

The `data/language/m-small/` directory included here contains the chunk files required for the primary benchmark (language category, m-small model). Each chunk covers a contiguous time window and consists of:

- `chunk-N-dataset.json` — per-client token-length PDFs
- `chunk-N-trace.csv` — per-client arrival rate, CV, and inter-arrival-time distribution parameters

**Other categories and model sizes** (`reason`, `multimodal`, `m-large`, etc.) are available from the upstream [ServeGen repository](https://github.com/alibaba/ServeGen). Download and place the corresponding subdirectory under `data/<category>/<model>/` to use them.
