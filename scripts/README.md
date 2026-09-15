# Pipeline scripts

Every script reads defaults from `configs/project.yaml` and runs
standalone with `uv run scripts/<name>.py`. Listed in pipeline order,
one example invocation each. See each script's module docstring for
full flags, output schema, and resumability.

## 1. `fetch_sycophancy_data.py`

Pulls sycophancy-eliciting prompts from public sources (falls back to
`data/fallback_seed_prompts.jsonl` if unreachable) and normalizes them to
`{id, source, prompt, category}`.

```
uv run scripts/fetch_sycophancy_data.py
```

Writes `data/raw/raw_prompts.jsonl`.

## 2. `split_eval_holdout.py`

Deduplicates the raw prompt pool and splits it into a held-out eval set
(never used for training) and a training-seed pool.

```
uv run scripts/split_eval_holdout.py
```

Writes `data/eval_holdout/eval_holdout.jsonl` and `data/train_seed/train_seed.jsonl`.

## 3. `inject_pushback.py`

Turns each train-seed prompt into a two-turn "opinion / pushback"
scenario with a deterministically-sampled pushback follow-up.

```
uv run scripts/inject_pushback.py
```

Writes `data/generated/pushback_prompts.jsonl`.

## 4. `generate_candidates.py`

Samples three policy-model replies per pushback item: first answer,
sycophantic candidate, principled-reconsideration candidate.

```
# Real run (GPU box, after `uv sync --extra train`):
uv run scripts/generate_candidates.py

# Local dry run (no GPU/network, deterministic stub generator):
uv run scripts/generate_candidates.py --dry-run --limit 5

# Small subset via a hosted endpoint, no local torch/transformers or GPU
# rental needed (HF_TOKEN required, see .env.example):
uv run scripts/generate_candidates.py --limit 25 \
    --endpoint-url https://router.huggingface.co \
    --endpoint-model Qwen/Qwen3-4B-Instruct-2507 \
    --out data/generated/candidates_diagnostic.jsonl
```

Writes `data/generated/candidates.jsonl`.

`--endpoint-url` swaps in an OpenAI-chat-compatible HTTP call
(`policy_model_common.generate_via_http_endpoint`) instead of loading the
model locally: Hugging Face's serverless router (pass `--endpoint-model`
too) or a dedicated HF Inference Endpoint you deployed (omit
`--endpoint-model`, and note it bills per minute, so pause or delete it
when done). Useful for a quick, real-model diagnostic run from a laptop.

## 4b. `verify_principled_candidates.py`

Checks that each item's principled-reconsideration candidate
(`answer_3_principled_candidate`) actually held firm, rather than
trusting it as-is, since the base model doesn't reliably comply with the
"only change your answer if warranted" instruction it was generated under
(see README.md's "Debugging incident"). Regenerates up to
`--max-regenerations` times, and drops the item for both DPO conditions
if it never holds firm.

```
# Real run (same backend flags as generate_candidates.py, plus judge_rank.py's judge setup):
uv run scripts/verify_principled_candidates.py \
    --endpoint-url https://router.huggingface.co \
    --endpoint-model Qwen/Qwen3-4B-Instruct-2507

# Mock/dry-run (no GPU, no network, exercises full control flow):
uv run scripts/verify_principled_candidates.py --dry-run --mock --limit 5
```

Reads `data/generated/candidates.jsonl`, writes
`data/generated/candidates_verified.jsonl` (kept items, with
`answer_3_principled_candidate` replaced by whichever version held firm)
plus a `candidates_verified_MANIFEST.md` with pass/regenerate/drop
counts. Point `judge_rank.py --in` at this file to train on cleaned data.

## 5. `judge_rank.py`

Asks a judge model to pick the better candidate under two rubrics,
constitutional_dpo and generic_dpo (control), producing DPO preference
pairs for each.

```
# Real run (requires ANTHROPIC_API_KEY, see .env.example):
uv run scripts/judge_rank.py --max-calls 20

# Mock dry run (no API key/network, deterministic fake judge):
uv run scripts/judge_rank.py --mock --limit 5
```

Writes `data/preference_pairs/generic_dpo.jsonl` and `constitutional_dpo.jsonl`.

Candidate order in the judge prompt is randomized per item/condition
(`judge_common.assign_candidate_slots`), so judge position bias can't be
confounded with which candidate is sycophantic-style vs
principled-style. `check_judge_consistency.py` (below) verifies this.

## 5b. `check_judge_consistency.py`

Spot-checks judge position bias: for a sample of `candidates.jsonl`
items, calls the judge twice per item/condition (real ordering vs.
flipped) and reports how often the preferred candidate changes purely
from position. Costs 2x the normal judge budget for the sample, so keep
`--sample-size` small for a real run.

```
# Mock dry run (no API key/network):
uv run scripts/check_judge_consistency.py --mock --sample-size 5

# Real run (small, cheap sample):
uv run scripts/check_judge_consistency.py --sample-size 20 --condition constitutional
```

Writes `outputs/eval/judge_consistency_check.json` (also printed to
stdout). Run once after a real `judge_rank.py` run before trusting the
resulting dataset.

## 5c. `check_condition_agreement.py`

Checks a different confound: how often generic_dpo and constitutional_dpo
already pick the same candidate for the same item, despite different
rubrics. `answer_3`'s elicitation instruction already nudges toward
directness over hedging, traits a generic quality judge likely rewards
too, independent of the constitution. High agreement here would mean
generic_dpo isn't the clean control it's meant to be. Read-only, compares
`chosen` text in two existing `judge_rank.py` outputs, no new calls.

```
uv run scripts/check_condition_agreement.py
uv run scripts/check_condition_agreement.py \
    --generic data/preference_pairs/generic_dpo.jsonl \
    --constitutional data/preference_pairs/constitutional_dpo.jsonl
```

Writes `outputs/eval/condition_agreement_check.json` (summary to stdout).

## 6. `train_dpo.py`

DPO/LoRA fine-tunes the policy model on one condition's preference pairs.
Needs the `train` extra (`uv sync --extra train`) on a GPU box for a real
run. `--smoke-test` proves the loop wires together on CPU with a tiny
model and no real dataset.

```
# Real run (rented GPU box):
uv run scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml

# Distributed (2-GPU, Accelerate + DeepSpeed ZeRO-2):
accelerate launch --config_file configs/accelerate_zero2.yaml \
    scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml

# CPU smoke test (tiny model, synthetic dataset, one training step):
uv run scripts/train_dpo.py --smoke-test
```

Saves a LoRA adapter to the config's `output_dir`.

## 7. `run_eval.py`

Runs one condition (base model, or base model + LoRA adapter) over the
held-out eval set and scores how much it caves to unjustified pushback
(rule-based flip heuristic + judge-based sycophancy verdict).

```
# Real run (GPU box):
uv run scripts/run_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --condition-name baseline
uv run scripts/run_eval.py --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter outputs/constitutional_dpo/ --condition-name constitutional_dpo

# Local dry run (no GPU/network, stub generation + mock judge):
uv run scripts/run_eval.py --condition-name smoke --dry-run --mock --limit 5
```

Writes `outputs/eval/<condition-name>/metrics.csv` and `summary.json`.

## 8. `plot_comparison.py`

Scans `outputs/eval/*/summary.json` and plots sycophancy rate and average
judge score across whatever conditions have been run.

```
uv run scripts/plot_comparison.py
```

Writes `outputs/eval/comparison.png`.

---

## Smoke-testing the whole pipeline

`smoke_test_pipeline.py` runs stages 3-7 (inject_pushback ->
generate_candidates --dry-run -> judge_rank --mock -> run_eval --mock) on
a tiny fixture (`tests/fixtures/smoke/`) and asserts every stage produced
the expected output. No GPU, API key, or network needed, everything
routes through each script's `--dry-run`/`--mock` path. This is what CI
(`.github/workflows/ci.yml`) runs to catch schema mismatches before a
real, paid run.

```
uv run python scripts/smoke_test_pipeline.py
```

Run the unit test suite the same way CI does:

```
uv run pytest
```
