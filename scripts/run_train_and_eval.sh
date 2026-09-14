#!/usr/bin/env bash
# Day-2 GPU-box run: trains both DPO conditions (generic_dpo and
# constitutional_dpo), evaluates all three conditions (baseline +
# both trained adapters) against the held-out eval set, then plots the
# comparison. See docs/NEXT_STEPS.md for the full runbook this automates
# (steps 7, 9, 10 -- the optional 2-GPU distributed-training demo, step 8,
# is a separate one-off artifact and is NOT part of this script).
#
# Run this on the rented GPU pod, after (one-time, not scripted here since
# it needs interactive key entry):
#   git clone <your-repo-url> && cd sycophancy-constitutional-dpo
#   curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
#   uv sync --extra train
#   cp .env.example .env   # fill in ANTHROPIC_API_KEY
#   uv run python scripts/smoke_test_pipeline.py   # sanity check first
#
# Total wall time: training is well under an hour per condition; eval
# across all three conditions is roughly 45-90 minutes on a single RTX
# 4090 (docs/NEXT_STEPS.md's estimates). Budget ~2-4 hours total.
#
# Terminate the pod as soon as this finishes -- everything after (plotting
# is already done here, but spot-checking transcripts and writing up
# results) is local/CPU work that doesn't need the GPU running.
set -euo pipefail

# Python buffers stdout in large blocks (not line-by-line) when it isn't
# writing to an interactive terminal -- which it isn't here, since it's
# piped through tee below. Without this, output (including tqdm progress
# bars) can sit invisible for many minutes before appearing, even though
# the underlying script is actually running fine.
export PYTHONUNBUFFERED=1

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="Qwen/Qwen3-4B-Instruct-2507"
EVAL_JUDGE_CONCURRENCY=4  # bound by Anthropic rate-limit tier; only affects run_eval.py's judging phase, not generation

mkdir -p logs
LOG_FILE="logs/train_and_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
TEE_PID=$!
trap 'exec 1>&- 2>&-; wait "$TEE_PID" 2>/dev/null || true' EXIT
echo "Logging this run to ${LOG_FILE} (tail -f it from another terminal to follow along)"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH. Install it first:" >&2
    echo '  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env' >&2
    exit 1
fi

echo "=============================================================="
echo "STEP 1/6: Training generic_dpo (control)"
echo "=============================================================="
uv run scripts/train_dpo.py --config configs/train_generic_dpo.yaml

echo
echo "=============================================================="
echo "STEP 2/6: Training constitutional_dpo (treatment)"
echo "=============================================================="
uv run scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml

echo
echo "=============================================================="
echo "STEP 3/6: Evaluating baseline (untouched base model)"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --condition-name baseline \
    --concurrency "$EVAL_JUDGE_CONCURRENCY"

echo
echo "=============================================================="
echo "STEP 4/6: Evaluating generic_dpo"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --adapter outputs/generic_dpo/ \
    --condition-name generic_dpo \
    --concurrency "$EVAL_JUDGE_CONCURRENCY"

echo
echo "=============================================================="
echo "STEP 5/6: Evaluating constitutional_dpo"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --adapter outputs/constitutional_dpo/ \
    --condition-name constitutional_dpo \
    --concurrency "$EVAL_JUDGE_CONCURRENCY"

echo
echo "=============================================================="
echo "STEP 6/6: Plotting the comparison"
echo "=============================================================="
uv run scripts/plot_comparison.py

echo
echo "=============================================================="
echo "Training + eval complete. Terminate this GPU pod now --"
echo "everything left (spot-checking transcripts, the write-up) is local/CPU work."
echo "=============================================================="
