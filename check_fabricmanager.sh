#!/bin/bash
# check_fabricmanager.sh — Verify NVIDIA system services for SXM4 multi-GPU operation.
# Safe to re-run (idempotent). Does NOT install packages automatically.

set -euo pipefail

EXPECTED_GPUS="${EXPECTED_GPUS:-8}"

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
err()   { echo "[ERROR] $*"; }
ok()    { echo "[OK]    $*"; }

service_active() {
    local svc="$1"
    if command -v systemctl &>/dev/null; then
        systemctl is-active --quiet "$svc" 2>/dev/null
    else
        service "$svc" status &>/dev/null
    fi
}

start_service() {
    local svc="$1"
    info "Attempting to start $svc ..."
    if command -v systemctl &>/dev/null; then
        sudo systemctl start "$svc" && sudo systemctl enable "$svc"
    else
        sudo service "$svc" start
    fi
}

# ── 1. nvidia-smi availability & GPU count ───────────────────────────────────

if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Install the NVIDIA driver first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [[ "$GPU_COUNT" -lt "$EXPECTED_GPUS" ]]; then
    warn "Expected $EXPECTED_GPUS GPUs but nvidia-smi reports $GPU_COUNT."
else
    ok "nvidia-smi reports $GPU_COUNT GPU(s) (expected $EXPECTED_GPUS)."
fi

# Detect driver version for install hints
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
info "NVIDIA driver version: $DRIVER_VERSION (major: $DRIVER_MAJOR)"

# ── 2. nvidia-persistenced ───────────────────────────────────────────────────

if service_active nvidia-persistenced; then
    ok "nvidia-persistenced is active."
else
    warn "nvidia-persistenced is NOT active (needed for DVFS clock locking)."
    start_service nvidia-persistenced && ok "nvidia-persistenced started." || {
        err "Could not start nvidia-persistenced."
        echo "    Install with:  sudo apt-get install -y nvidia-utils-${DRIVER_MAJOR}"
        echo "    Then run:      sudo systemctl enable --now nvidia-persistenced"
    }
fi

# ── 3. nvidia-fabricmanager ──────────────────────────────────────────────────

if service_active nvidia-fabricmanager; then
    ok "nvidia-fabricmanager is active."
else
    warn "nvidia-fabricmanager is NOT active (needed for NVSwitch on SXM4)."
    start_service nvidia-fabricmanager && ok "nvidia-fabricmanager started." || {
        err "Could not start nvidia-fabricmanager."
        echo "    Install the version matching your driver ($DRIVER_VERSION):"
        echo "      sudo apt-get install -y nvidia-fabricmanager-${DRIVER_MAJOR}=${DRIVER_VERSION}-1"
        echo "    Then run:"
        echo "      sudo systemctl enable --now nvidia-fabricmanager"
    }
fi

# ── 4. GPU health summary ───────────────────────────────────────────────────

echo ""
echo "=== GPU Health Summary ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo ""

ok "All checks complete."
