#!/usr/bin/env bash
# Full-scale run of the candidate-generation -> judge-ranking pipeline over
# the entire pushback_prompts.jsonl dataset (no --limit): generate all
# candidates via the hosted HF endpoint, rank them into DPO preference
# pairs with the judge, then check how much the generic_dpo and
# constitutional_dpo rubrics already agree.
#
# Before running: make sure the HF Inference Endpoint's autoscaling max
# replicas is actually raised to (at least) GENERATE_CONCURRENCY below --
# concurrency only helps as far as replicas are available to serve it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENDPOINT_URL="https://tjvloo7l2rzg91h4.us-east-1.aws.endpoints.huggingface.cloud"
GENERATE_CONCURRENCY=10
JUDGE_CONCURRENCY=4  # bound by Anthropic rate-limit tier

CANDIDATES_OUT="data/generated/candidates.jsonl"
PAIRS_OUT="data/preference_pairs/"

# Tee everything (stdout + stderr, including uv's own output and tqdm bars)
# to a timestamped log file under logs/, while still showing it live in the
# terminal. *.log is already gitignored. The EXIT trap closes our end of the
# pipe to tee (so it sees EOF) before waiting on it, so the file is
# guaranteed fully flushed by the time this script actually exits -- without
# that explicit close, waiting on tee here would deadlock, and without the
# wait at all, a fast failure could exit before tee finishes writing.
mkdir -p logs
LOG_FILE="logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
TEE_PID=$!
trap 'exec 1>&- 2>&-; wait "$TEE_PID" 2>/dev/null || true' EXIT
echo "Logging this run to ${LOG_FILE} (tail -f it from another terminal to follow along)"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH in this shell." >&2
    echo "On Windows, 'bash' often resolves to WSL's launcher (C:\\Windows\\System32\\bash.exe)" >&2
    echo "instead of Git for Windows' bash -- WSL is a separate Linux environment that never" >&2
    echo "sees a Windows-installed uv. Check with 'where.exe bash' in PowerShell; if" >&2
    echo "System32\\bash.exe is listed first, run this from a Git Bash window instead, or:" >&2
    echo '  & "C:\Program Files\Git\bin\bash.exe" scripts/run_full_pipeline.sh' >&2
    exit 1
fi

echo "=============================================================="
echo "STEP 1/3: Generating candidates for the full dataset via HF endpoint"
echo "  concurrency=${GENERATE_CONCURRENCY} -> ${CANDIDATES_OUT}"
echo "=============================================================="
uv run scripts/generate_candidates.py \
    --concurrency "$GENERATE_CONCURRENCY" \
    --endpoint-url "$ENDPOINT_URL" \
    --endpoint-api text-generation \
    --out "$CANDIDATES_OUT"

echo
echo "=============================================================="
echo "STEP 2/3: Judging candidates -> DPO preference pairs"
echo "  concurrency=${JUDGE_CONCURRENCY} -> ${PAIRS_OUT}"
echo "=============================================================="
uv run scripts/judge_rank.py \
    --in "$CANDIDATES_OUT" \
    --out "$PAIRS_OUT" \
    --concurrency "$JUDGE_CONCURRENCY"

echo
echo "=============================================================="
echo "STEP 3/3: Checking generic_dpo vs constitutional_dpo agreement"
echo "=============================================================="
uv run scripts/check_condition_agreement.py \
    --generic "${PAIRS_OUT}generic_dpo.jsonl" \
    --constitutional "${PAIRS_OUT}constitutional_dpo.jsonl"

echo
echo "=============================================================="
echo "Full pipeline complete."
echo "=============================================================="
