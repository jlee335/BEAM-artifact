#!/bin/bash
# smoke_test.sh — Quick in-container validation for BEAM artifact evaluation.
# Runs a short 3-minute e2e test (Vanilla, DynamoLLM, S1+S2) and checks outputs.
#
# Usage (inside container, from /workspace/benchmarks/energy):
#   ./smoke_test.sh
#   ./smoke_test.sh --model Qwen/Qwen2.5-14B --tp 1 --pp 4   # lighter model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="meta-llama/Llama-3.3-70B-Instruct"
TP=2
PP=4
DATASET="datasets/requests_lang_m-small_day1_19h00m-19h03m_200s_3rps.csv"
RUN_PROFILES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --dataset-path) DATASET="$2"; shift 2 ;;
        --run-profiles) RUN_PROFILES=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--run-profiles] [--model <name>] [--tp <n>] [--pp <n>] [--dataset-path <path>]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PASS=true

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; PASS=false; }
ok()    { echo "[OK]    $*"; }
fail()  { echo "[FAIL]  $*"; PASS=false; }

echo ""
echo "=============================================="
echo "  BEAM Smoke Test"
echo "=============================================="
echo "  Model:   $MODEL"
echo "  TP×PP:   ${TP}×${PP}"
echo "  Dataset: $DATASET"
echo "=============================================="
echo ""

# ── 1. Validate environment ─────────────────────────────────────────────────

info "Checking GPU count..."
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
NEEDED=$((TP * PP))
if [[ "$GPU_COUNT" -lt "$NEEDED" ]]; then
    fail "Need at least $NEEDED GPUs (TP=$TP × PP=$PP) but found $GPU_COUNT."
    echo ""
    echo "VERDICT: FAIL"
    exit 1
fi
ok "$GPU_COUNT GPU(s) available (need $NEEDED)."

info "Checking DVFS permissions (lock/unlock lowest clock)..."
# Try to lock GPU 0 to its lowest supported clock, then release
if nvidia-smi -i 0 -pm 1 &>/dev/null && nvidia-smi -i 0 -lgc 210 &>/dev/null; then
    nvidia-smi -i 0 -rgc &>/dev/null
    nvidia-smi -i 0 -pm 0 &>/dev/null
    ok "DVFS clock locking works."
else
    nvidia-smi -i 0 -rgc &>/dev/null 2>&1 || true
    nvidia-smi -i 0 -pm 0 &>/dev/null 2>&1 || true
    warn "DVFS clock locking failed. Container may need --privileged or persistence mode on the host."
fi

# ── 2. Check offline profiles ───────────────────────────────────────────────

info "Checking offline profiles..."

# Derive GPU name for profile lookup
GPU_NAME=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | xargs)
MODEL_CLEANED=$(echo "$MODEL" | sed 's/\//_/g')

SYSTEM_PROFILE="offline_profile_results/dvfs_profile_${GPU_NAME}_${MODEL_CLEANED}_tp${TP}_pp${PP}_one.csv"
DYNAMO_PROFILE="dynamollm_profiles/dynamo_dvfs_profile_${GPU_NAME}_${MODEL_CLEANED}.csv"

MISSING_PROFILES=()
if [[ -f "$SYSTEM_PROFILE" ]]; then
    ok "System profile found: $SYSTEM_PROFILE"
else
    MISSING_PROFILES+=("$SYSTEM_PROFILE")
fi

if [[ -f "$DYNAMO_PROFILE" ]]; then
    ok "DynamoLLM profile found: $DYNAMO_PROFILE"
else
    MISSING_PROFILES+=("$DYNAMO_PROFILE")
fi

if [[ ${#MISSING_PROFILES[@]} -gt 0 ]]; then
    if [[ "$RUN_PROFILES" == true ]]; then
        info "Generating missing profiles..."
        bash "$SCRIPT_DIR/run_offline_profile.sh" \
            --model "$MODEL" --tp "$TP" --pp "$PP"
        bash "$SCRIPT_DIR/run_dynamo_benchmark.sh" \
            --model "$MODEL" --tp "$TP" --pp "$PP"
        # Re-check after generation
        for p in "${MISSING_PROFILES[@]}"; do
            if [[ -f "$p" ]]; then
                ok "Profile generated: $p"
            else
                fail "Failed to generate profile: $p"
                echo ""
                echo "VERDICT: FAIL"
                exit 1
            fi
        done
    else
        for p in "${MISSING_PROFILES[@]}"; do
            fail "Profile not found: $p"
        done
        echo ""
        echo "  Missing profiles. Re-run with --run-profiles to generate them:"
        echo "    $0 --run-profiles --model $MODEL --tp $TP --pp $PP"
        echo ""
        exit 1
    fi
fi

# ── 3. Check dataset exists ─────────────────────────────────────────────────

if [[ ! -f "$DATASET" ]]; then
    fail "Dataset not found: $DATASET"
    echo ""
    echo "VERDICT: FAIL"
    exit 1
fi
ok "Dataset exists: $DATASET"

# ── 4. Run short e2e evaluation ──────────────────────────────────────────────

info "Running end-to-end evaluation with short dataset..."
echo "    This exercises Vanilla, DynamoLLM, and S1+S2 on a 3-minute trace."
echo ""

bash "$SCRIPT_DIR/run_e2e.sh" \
    --model "$MODEL" --tp "$TP" --pp "$PP" \
    --dataset-path "$SCRIPT_DIR/$DATASET"
E2E_RC=$?

if [[ $E2E_RC -ne 0 ]]; then
    fail "run_e2e.sh exited with code $E2E_RC."
fi

# ── 5. Validate outputs ─────────────────────────────────────────────────────

info "Validating outputs..."

# Find the most recent e2e output directory
LATEST_DIR=$(ls -dt end_to_end/*/ 2>/dev/null | head -1)
if [[ -z "$LATEST_DIR" ]]; then
    fail "No output directory found under end_to_end/."
else
    ok "Output directory: $LATEST_DIR"

    EXPERIMENT_PASS=true
    for exp_dir in "$LATEST_DIR"*/; do
        exp_name=$(basename "$exp_dir")

        # Check for energy CSV
        energy_csv=$(find "$exp_dir" -name "gpu_energy_and_frequency_*.csv" 2>/dev/null | head -1)
        if [[ -n "$energy_csv" ]] && [[ -s "$energy_csv" ]]; then
            ok "  $exp_name: energy CSV present ($(wc -l < "$energy_csv") lines)"
        else
            fail "  $exp_name: energy CSV missing or empty"
            EXPERIMENT_PASS=false
        fi

        # Check for JSON results
        json_file=$(find "$exp_dir" -name "*.json" 2>/dev/null | head -1)
        if [[ -n "$json_file" ]] && [[ -s "$json_file" ]]; then
            ok "  $exp_name: JSON results present"
        else
            fail "  $exp_name: JSON results missing or empty"
            EXPERIMENT_PASS=false
        fi
    done
fi

# ── 6. Visualize results ────────────────────────────────────────────────────

RESULTS_DIR="$SCRIPT_DIR/results_smoke_$(date +%Y%m%d_%H%M%S)"
bash "$SCRIPT_DIR/collect_visualizations.sh" --output-dir "$RESULTS_DIR"

# ── 7. Verdict ───────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
if [[ "$PASS" == true ]]; then
    echo "  VERDICT: PASS"
    echo "=============================================="
    echo ""
    echo "  The environment is correctly configured."
    echo "  Visualizations saved to: $RESULTS_DIR"
    echo ""
    echo "  Proceed with the full evaluation:"
    echo ""
    echo "    ./run_all.sh --skip-profiling \\"
    echo "      --model meta-llama/Llama-3.3-70B-Instruct \\"
    echo "      --tp 2 --pp 4 \\"
    echo "      --dataset-path /workspace/benchmarks/energy/datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv"
    echo ""
    exit 0
else
    echo "  VERDICT: FAIL"
    echo "=============================================="
    echo "  Review the warnings/failures above before proceeding."
    echo ""
    exit 1
fi
