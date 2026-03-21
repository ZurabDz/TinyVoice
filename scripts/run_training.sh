#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

usage() {
    cat <<'EOF'
Usage: scripts/run_training.sh <mode> [options]

Modes:
  train              Normal full training run
  dev                Dev mode — short run (default 10k steps, configurable)
  profile            Training with JAX profiler tracing
  debug              Training with JAX debug flags (nan checks, disable JIT, log compiles)
  debug-nans         Training with only NaN checking enabled
  dev-profile        Dev mode + profiler tracing combined
  cpu                Force CPU backend (for testing without GPU)

Options (via environment variables):
  TINYVOICE_DEV_STEPS=N       Number of steps in dev mode (default: 10000)
  TINYVOICE_PROFILE_START=N   Step to start profiling (default: 10)
  TINYVOICE_PROFILE_STEPS=N   Number of steps to profile (default: 5)

Examples:
  scripts/run_training.sh train
  scripts/run_training.sh dev
  TINYVOICE_DEV_STEPS=500 scripts/run_training.sh dev
  scripts/run_training.sh profile
  TINYVOICE_PROFILE_START=20 TINYVOICE_PROFILE_STEPS=10 scripts/run_training.sh profile
  scripts/run_training.sh debug
  scripts/run_training.sh debug-nans
  scripts/run_training.sh dev-profile
  scripts/run_training.sh cpu
EOF
    exit 1
}

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    usage
fi

case "$MODE" in
    train)
        echo "==> Normal training run"
        exec uv run python train_minimal.py
        ;;

    dev)
        DEV_STEPS="${TINYVOICE_DEV_STEPS:-10000}"
        echo "==> Dev mode ($DEV_STEPS steps)"
        exec env \
            TINYVOICE_DEV=1 \
            TINYVOICE_DEV_STEPS="$DEV_STEPS" \
            uv run python train_minimal.py
        ;;

    profile)
        START="${TINYVOICE_PROFILE_START:-10}"
        STEPS="${TINYVOICE_PROFILE_STEPS:-5}"
        echo "==> Training with profiler (steps $START-$((START + STEPS - 1)))"
        echo "    Traces will be in ./runs/profile/"
        exec env \
            TINYVOICE_PROFILE_START="$START" \
            TINYVOICE_PROFILE_STEPS="$STEPS" \
            uv run python train_minimal.py
        ;;

    debug)
        echo "==> Debug mode (NaN checks + log compiles + disable JIT)"
        exec env \
            JAX_DEBUG_NANS=1 \
            JAX_LOG_COMPILES=1 \
            JAX_DISABLE_JIT=1 \
            TINYVOICE_DEV=1 \
            TINYVOICE_DEV_STEPS="${TINYVOICE_DEV_STEPS:-50}" \
            uv run python train_minimal.py
        ;;

    debug-nans)
        echo "==> Training with NaN checking enabled"
        exec env \
            JAX_DEBUG_NANS=1 \
            uv run python train_minimal.py
        ;;

    dev-profile)
        DEV_STEPS="${TINYVOICE_DEV_STEPS:-1000}"
        START="${TINYVOICE_PROFILE_START:-10}"
        STEPS="${TINYVOICE_PROFILE_STEPS:-5}"
        echo "==> Dev mode ($DEV_STEPS steps) + profiler (steps $START-$((START + STEPS - 1)))"
        exec env \
            TINYVOICE_DEV=1 \
            TINYVOICE_DEV_STEPS="$DEV_STEPS" \
            TINYVOICE_PROFILE_START="$START" \
            TINYVOICE_PROFILE_STEPS="$STEPS" \
            uv run python train_minimal.py
        ;;

    cpu)
        echo "==> CPU-only training (dev mode)"
        exec env \
            JAX_PLATFORMS=cpu \
            TINYVOICE_DEV=1 \
            TINYVOICE_DEV_STEPS="${TINYVOICE_DEV_STEPS:-100}" \
            uv run python train_minimal.py
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo
        usage
        ;;
esac
