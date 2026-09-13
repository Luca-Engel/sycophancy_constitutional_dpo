#!/usr/bin/env bash
# Smoke tests to run on the rented GPU box BEFORE committing to the full,
# multi-hour scripts/run_train_and_eval.sh. Cheapest/fastest checks first;
# each stage catches a different class of failure that a paid full run
# would otherwise hit partway through:
#
#   1. scripts/smoke_test_pipeline.py  -- script I/O schemas line up
#      (seconds, no GPU, no heavy deps)
#   2. train_dpo.py --smoke-test       -- transformers/peft/trl/DPOTrainer
#      wiring is correct (seconds, CPU-only, tiny stand-in model + synthetic
#      data -- does NOT touch the real model, LoRA target names, or dataset)
#   3. train_dpo.py with the REAL generic_dpo config, capped at 2 steps
#      -- forces the real `uv sync --extra train` deps to import, downloads
#      the real Qwen3-4B-Instruct-2507 weights, and validates the real LoRA
#      target modules + the real dataset file + bf16 + batch size on THIS
#      GPU (a minute or two, not the better part of an hour)
#   4/5. run_eval.py against 2 real eval items, once with no adapter and
#      once loading the adapter step 3 just produced -- validates real
#      generation, adapter loading, and a live ANTHROPIC_API_KEY
#
# If all 5 pass, scripts/run_train_and_eval.sh should run to completion
# without an environment/dependency/schema surprise partway through.
#
# Outputs land in outputs/smoke_generic_dpo/ and outputs/eval/smoke_*/ --
# distinct from the real run's outputs/{generic_dpo,constitutional_dpo}/ and
# outputs/eval/{baseline,generic_dpo,constitutional_dpo}/, so nothing here
# collides with or gets reused by the real run. Safe to delete afterward.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="Qwen/Qwen3-4B-Instruct-2507"
SMOKE_ADAPTER_DIR="outputs/smoke_generic_dpo"

mkdir -p logs
LOG_FILE="logs/smoke_tests_$(date +%Y%m%d_%H%M%S).log"
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
echo "SMOKE 1/5: Pipeline script I/O schema check (no GPU, no heavy deps)"
echo "=============================================================="
uv run python scripts/smoke_test_pipeline.py

echo
echo "=============================================================="
echo "SMOKE 2/5: Training-loop wiring check (tiny stand-in model, CPU-only)"
echo "=============================================================="
uv run scripts/train_dpo.py --smoke-test

echo
echo "=============================================================="
echo "SMOKE 3/5: Real model + real LoRA config + real dataset, 2 steps"
echo "  -> ${SMOKE_ADAPTER_DIR}"
echo "=============================================================="
uv run scripts/train_dpo.py --config configs/train_generic_dpo.yaml \
    --max-steps 2 --output-dir "$SMOKE_ADAPTER_DIR"

echo
echo "=============================================================="
echo "SMOKE 4/5: Real eval generation, no adapter, 2 items"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --condition-name smoke_baseline --limit 2

echo
echo "=============================================================="
echo "SMOKE 5/5: Real eval generation with the smoke adapter, 2 items"
echo "=============================================================="
uv run scripts/run_eval.py --model "$MODEL" \
    --adapter "${SMOKE_ADAPTER_DIR}/" --condition-name smoke_adapter --limit 2

echo
echo "=============================================================="
echo "All smoke tests passed. Safe to run scripts/run_train_and_eval.sh next."
echo "(outputs/smoke_generic_dpo/ and outputs/eval/smoke_*/ are throwaway --"
echo " delete them any time, they don't feed into the real run.)"
echo "=============================================================="
