#!/usr/bin/env bash
# Day-2 GPU-box run, part 2: trains both DPO conditions on the filtered
# (drop-only, no regeneration) preference-pairs variant
# (data/preference_pairs_v2_clean/, 423 pairs/condition -- see that
# directory's MANIFEST.md and README.md's "Debugging incident" section for
# what this variant is and why), evaluates just those two new conditions,
# and re-plots the comparison alongside the original baseline/generic_dpo/
# constitutional_dpo results already sitting in outputs/eval/. This is the
# direct "filter vs no-filter" comparison run.
#
# Prerequisites (one-time on a fresh pod, assumed already done):
#   git clone https://github.com/Luca-Engel/sycophancy_constitutional_dpo.git
#   cd sycophancy_constitutional_dpo   # or: cd it, then `git pull` if already cloned
#   curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
#   uv sync --extra train
#   cp .env.example .env   # fill in ANTHROPIC_API_KEY
#
# Does NOT re-run baseline eval -- it doesn't depend on training data, so
# the existing outputs/eval/baseline/ from the original run is reused as-is.
#
# Total wall time: similar ballpark to the original run_train_and_eval.sh,
# a bit less given the dataset is ~10% smaller (423 vs 466 pairs/condition).
#
# Terminate the pod once this finishes -- copying results back and any
# further analysis is local/CPU work that doesn't need the GPU running.
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
LOG_FILE="logs/train_and_eval_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
TEE_PID=$!
trap 'exec 1>&- 2>&-; wait "$TEE_PID" 2>/dev/null || true' EXIT
echo "Logging this run to ${LOG_FILE} (tail -f it from another terminal to follow along)"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH. Install it first:" >&2
    echo '  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env' >&2
    exit 1
fi

if [ ! -f data/preference_pairs_v2_clean/generic_dpo.jsonl ] || [ ! -f data/preference_pairs_v2_clean/constitutional_dpo.jsonl ]; then
    echo "ERROR: data/preference_pairs_v2_clean/*.jsonl not found. Did 'git pull' actually bring the" >&2
    echo "new data files down? Check with: git log --oneline -- data/preference_pairs_v2_clean/" >&2
    exit 1
fi

if [ ! -f outputs/eval/baseline/summary.json ]; then
    echo "WARNING: outputs/eval/baseline/summary.json not found on this pod -- plot_comparison.py" >&2
    echo "will only show the two new conditions below, not a baseline comparison. Continuing anyway." >&2
fi

echo "=============================================================="
echo "STEP 1/5: Training generic_dpo_v2 (filtered dataset, no regeneration)"
echo "=============================================================="
uv run scripts/train_dpo.py --config configs/train_generic_dpo.yaml \
    --dataset-path data/preference_pairs_v2_clean/generic_dpo.jsonl \
    --output-dir outputs/generic_dpo_v2

echo
echo "=============================================================="
echo "STEP 2/5: Training constitutional_dpo_v2 (filtered dataset, no regeneration)"
echo "=============================================================="
uv run scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml \
    --dataset-path data/preference_pairs_v2_clean/constitutional_dpo.jsonl \
    --output-dir outputs/constitutional_dpo_v2

echo
echo "=============================================================="
echo "STEP 3/5: Evaluating generic_dpo_v2"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --adapter outputs/generic_dpo_v2/ \
    --condition-name generic_dpo_v2 \
    --concurrency "$EVAL_JUDGE_CONCURRENCY"

echo
echo "=============================================================="
echo "STEP 4/5: Evaluating constitutional_dpo_v2"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --adapter outputs/constitutional_dpo_v2/ \
    --condition-name constitutional_dpo_v2 \
    --concurrency "$EVAL_JUDGE_CONCURRENCY"

echo
echo "=============================================================="
echo "STEP 5/5: Plotting the comparison (every condition found under outputs/eval/)"
echo "=============================================================="
uv run scripts/plot_comparison.py

echo
echo "=============================================================="
echo "Training + eval complete. Terminate this GPU pod now --"
echo "everything left (copying results back, analysis) is local/CPU work."
echo "=============================================================="
