#!/bin/bash
# setup_reviewer.sh — One-command host-side setup for AE reviewers.
# Run from the repository root: ./setup_reviewer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="vllm_evaluator_env"
IMAGE_NAME="jlee335/beam-volume-260307:latest"
STARTUP_TIMEOUT=120
DEFAULT_MODEL="meta-llama/Llama-3.3-70B-Instruct"
MODEL="$DEFAULT_MODEL"

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
err()   { echo "[ERROR] $*" >&2; }
ok()    { echo "[OK]    $*"; }

fail() { err "$*"; exit 1; }

check_cmd() {
    command -v "$1" &>/dev/null || fail "'$1' is not installed. Please install it first."
}

usage() {
    cat <<EOF
Usage: $0 [--model MODEL_ID]

Options:
  --model MODEL_ID   Hugging Face model ID to use (default: $DEFAULT_MODEL)
  -h, --help         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || fail "--model requires a value."
            MODEL="$2"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            [[ -n "$MODEL" ]] || fail "--model requires a non-empty value."
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1. Use --help for usage."
            ;;
    esac
done

# ── 1. Prerequisites ────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo "  BEAM Artifact — Reviewer Setup"
echo "=============================================="
echo ""

info "Checking prerequisites..."
check_cmd docker
check_cmd nvidia-smi

# Docker Compose v2 (docker compose) or v1 (docker-compose)
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
else
    fail "Docker Compose (v2) is required. Install with: sudo apt-get install docker-compose-plugin"
fi
ok "Docker ($COMPOSE) found."

# nvidia-container-toolkit
if docker info 2>/dev/null | grep -qi nvidia; then
    ok "NVIDIA Container Toolkit detected."
else
    warn "NVIDIA Container Toolkit may not be installed/configured."
    echo "    Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "    Continuing anyway — GPU passthrough may still work if configured differently."
fi

# ── 2. NVIDIA system services ───────────────────────────────────────────────

info "Running NVIDIA system service checks..."
bash "$SCRIPT_DIR/check_fabricmanager.sh" || {
    warn "Some NVIDIA services are not running. Review the warnings above."
    echo "    You can re-run:  ./check_fabricmanager.sh"
}

# ── 3. Hugging Face token ───────────────────────────────────────────────────

if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN is set in environment."
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    ok "HF_TOKEN loaded from ~/.cache/huggingface/token."
else
    echo ""
    warn "HF_TOKEN not found. It is required to download model weights."
    echo "    Get a token at: https://huggingface.co/settings/tokens"
    read -rp "    Enter your HF_TOKEN (or press Enter to skip): " HF_TOKEN
    if [[ -z "$HF_TOKEN" ]]; then
        warn "Skipping HF_TOKEN — model download will fail unless token is provided later."
        echo "    Set it before running benchmarks:  export HF_TOKEN=<your_token>"
    else
        export HF_TOKEN
        ok "HF_TOKEN set."
    fi
fi

# ── 4. Model cache check ────────────────────────────────────────────────────

MODEL_CACHE_KEY="${MODEL//\//--}"
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--${MODEL_CACHE_KEY}/snapshots"
if [[ -d "$MODEL_CACHE" ]] && [[ -n "$(ls -A "$MODEL_CACHE" 2>/dev/null)" ]]; then
    ok "Model cache found for $MODEL at $MODEL_CACHE."
else
    warn "Model cache not found at $MODEL_CACHE."
    echo "    The selected model ($MODEL) will be"
    echo "    downloaded on the first benchmark run. Ensure sufficient disk space."
fi

# ── 5. Docker image ─────────────────────────────────────────────────────────

info "Pulling Docker image ($IMAGE_NAME)..."
if $COMPOSE pull 2>/dev/null; then
    ok "Docker image pulled."
else
    warn "Pull failed — building from source (this may take several minutes)..."
    $COMPOSE build || fail "Docker image build failed."
    ok "Docker image built."
fi

# ── 6. Start container ──────────────────────────────────────────────────────

info "Starting container..."
$COMPOSE up -d || fail "Failed to start container."

info "Waiting for container startup (up to ${STARTUP_TIMEOUT}s)..."
elapsed=0
while [[ $elapsed -lt $STARTUP_TIMEOUT ]]; do
    if docker exec "$CONTAINER_NAME" echo ready &>/dev/null; then
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [[ $elapsed -ge $STARTUP_TIMEOUT ]]; then
    fail "Container did not become ready within ${STARTUP_TIMEOUT}s. Check:  docker logs $CONTAINER_NAME"
fi
ok "Container is running."

# ── 7. Verify GPU access inside container ────────────────────────────────────

info "Verifying GPU access inside container..."
CONTAINER_GPUS=$(docker exec "$CONTAINER_NAME" nvidia-smi --list-gpus 2>/dev/null | wc -l)
HOST_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [[ "$CONTAINER_GPUS" -eq "$HOST_GPUS" ]]; then
    ok "Container sees $CONTAINER_GPUS GPU(s) (matches host)."
else
    warn "Container sees $CONTAINER_GPUS GPU(s) but host has $HOST_GPUS."
fi

# ── 8. In-container smoke test ───────────────────────────────────────────────

info "Running in-container import checks..."
docker exec "$CONTAINER_NAME" python3 -c "
import pynvml, pandas, torch
pynvml.nvmlInit()
gpu_count = pynvml.nvmlDeviceGetCount()
print(f'pynvml OK — {gpu_count} GPU(s)')
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch OK — {torch.cuda.device_count()} CUDA device(s)')
pynvml.nvmlShutdown()
" || fail "In-container import checks failed. Check container logs:  docker logs $CONTAINER_NAME"
ok "In-container imports verified (pynvml, pandas, torch)."

# ── 9. Next steps ───────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Enter the container:"
echo "    docker exec -it -w /workspace/benchmarks/energy $CONTAINER_NAME bash"
echo ""
echo "  Quick validation (~30 min, 3-minute dataset):"
echo "    ./smoke_test.sh"
echo ""
echo "  Full evaluation (~5-6 hours, 1-hour dataset):"
echo "    ./run_all.sh --skip-profiling \\"
echo "      --model $MODEL \\"
echo "      --tp 2 --pp 4 \\"
echo "      --dataset-path /workspace/benchmarks/energy/datasets/requests_lang_m-small_day1_19h00m-20h00m_3600s_3rps.csv"
echo ""
echo "=============================================="
