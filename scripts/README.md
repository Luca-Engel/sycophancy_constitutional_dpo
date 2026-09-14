# Pipeline scripts

Every script reads its defaults from `configs/project.yaml` and can be run
standalone with `uv run scripts/<name>.py`. This lists them in pipeline
order with one example invocation each. See each script's own module
docstring for full details (all flags, output schema, resumability).

## 1. `fetch_sycophancy_data.py`

Pulls sycophancy-eliciting prompts from public sources (falls back to the
bundled `data/fallback_seed_prompts.jsonl` if the network sources are
unreachable) and normalizes them to `{id, source, prompt, category}`.

```
uv run scripts/fetch_sycophancy_data.py
```

Writes `data/raw/raw_prompts.jsonl`.

## 2. `split_eval_holdout.py`

Deduplicates the raw prompt pool and splits it into a held-out evaluation
set (never used for training) and a training-seed pool.

```
uv run scripts/split_eval_holdout.py
```

Writes `data/eval_holdout/eval_holdout.jsonl` and `data/train_seed/train_seed.jsonl`.

## 3. `inject_pushback.py`

Turns each train-seed prompt into a two-turn "opinion / pushback" scenario
by attaching a deterministically-sampled pushback follow-up message.

```
uv run scripts/inject_pushback.py
```

Writes `data/generated/pushback_prompts.jsonl`.

## 4. `generate_candidates.py`

Samples three policy-model replies per pushback item (first answer,
sycophantic candidate, principled-reconsideration candidate).

```
# Real run (Day 2, on a GPU box, after `uv sync --extra train`):
uv run scripts/generate_candidates.py

# Local dry run (no GPU/network, deterministic stub generator):
uv run scripts/generate_candidates.py --dry-run --limit 5

# Small exploratory subset via a hosted endpoint, no local torch/transformers
# install and no GPU rental needed (HF_TOKEN required, see .env.example):
uv run scripts/generate_candidates.py --limit 25 \
    --endpoint-url https://router.huggingface.co \
    --endpoint-model Qwen/Qwen3-4B-Instruct-2507 \
    --out data/generated/candidates_diagnostic.jsonl
```

Writes `data/generated/candidates.jsonl`.

`--endpoint-url` swaps the generation backend for an OpenAI-chat-compatible
HTTP call (`policy_model_common.generate_via_http_endpoint`) instead of
loading the model locally. Works against Hugging Face's serverless
Inference Providers router (pass `--endpoint-model` too, since the router
serves many models) or a dedicated HF Inference Endpoint URL you deployed
yourself (omit `--endpoint-model`, it's already bound to one model -- and
remember it bills per minute while running, so pause/delete it when done).
Useful for a quick, small, real-model diagnostic run from a plain laptop.

## 4b. `verify_principled_candidates.py`

Checks that each item's principled-reconsideration candidate
(`answer_3_principled_candidate`) actually held firm against its pushback,
rather than trusting it as-is -- the base policy model doesn't reliably
comply with the "only change your answer if warranted" instruction it was
generated under (see README.md's "Debugging incident" section for the
real run that found this). Regenerates up to `--max-regenerations` times;
drops the item entirely (for both DPO conditions -- see the module
docstring for why) if it never holds firm.

```
# Real run (same backend flags as generate_candidates.py, plus judge_rank.py's judge setup):
uv run scripts/verify_principled_candidates.py \
    --endpoint-url https://router.huggingface.co \
    --endpoint-model Qwen/Qwen3-4B-Instruct-2507

# Mock/dry-run (no GPU, no network, exercises full control flow):
uv run scripts/verify_principled_candidates.py --dry-run --mock --limit 5
```

Reads `data/generated/candidates.jsonl`, writes
`data/generated/candidates_verified.jsonl` (only kept items, with
`answer_3_principled_candidate` replaced by whichever version actually held
firm) plus a `candidates_verified_MANIFEST.md` documenting pass/regenerate/
drop counts. Point `judge_rank.py --in` at this file instead of the raw
`candidates.jsonl` to train on the cleaned data.

## 5. `judge_rank.py`

Asks a judge model to pick the better of the two candidates under two
rubrics -- constitutional_dpo and generic_dpo (control) -- producing DPO
preference pairs for each.

```
# Real run (requires ANTHROPIC_API_KEY, see .env.example):
uv run scripts/judge_rank.py --max-calls 20

# Mock dry run (no API key/network, deterministic fake judge):
uv run scripts/judge_rank.py --mock --limit 5
```

Writes `data/preference_pairs/generic_dpo.jsonl` and `constitutional_dpo.jsonl`.

Candidate order in the judge prompt is randomized per item/condition
(`judge_common.assign_candidate_slots`) so judge position bias can't be
confounded with which candidate is the sycophantic-style vs
principled/reconsideration-style one -- see `judge_common.py`'s module
docstring. `check_judge_consistency.py` (below) is the direct check that
this is working.

## 5b. `check_judge_consistency.py`

Spot-checks judge position bias: for a small sample of `candidates.jsonl`
items, calls the judge twice per item/condition (the real slot ordering and
the deliberately flipped one) and reports how often the *underlying
candidate preferred* changes just because of position. Costs 2x the normal
judge-call budget for the sampled items -- keep `--sample-size` small for a
real run.

```
# Mock dry run (no API key/network):
uv run scripts/check_judge_consistency.py --mock --sample-size 5

# Real run (small, cheap sample):
uv run scripts/check_judge_consistency.py --sample-size 20 --condition constitutional
```

Writes a report to `outputs/eval/judge_consistency_check.json` (also
printed to stdout as a summary). Run this once after a real `judge_rank.py`
run (or on a partial `candidates.jsonl`) before trusting the resulting
preference-pair dataset.

## 5c. `check_condition_agreement.py`

Diagnostic for a different confound than 5b's (position bias): how often
generic_dpo and constitutional_dpo already pick the *same* underlying
candidate for the same item, despite using different rubrics. Both
conditions choose between the same `answer_2`/`answer_3` pair, and
`answer_3`'s elicitation instruction (see `generate_candidates.py`'s
`RECONSIDER_PROMPT`) already nudges toward traits -- directness, not
hedging -- a generic quality judge likely rewards too, independent of the
constitution. A high agreement rate here means generic_dpo isn't the clean,
constitution-free control it's meant to be. Read-only: compares the
`chosen` text in two existing `judge_rank.py` output files, no new judge or
policy-model calls.

```
uv run scripts/check_condition_agreement.py
uv run scripts/check_condition_agreement.py \
    --generic data/preference_pairs/generic_dpo.jsonl \
    --constitutional data/preference_pairs/constitutional_dpo.jsonl
```

Writes a report to `outputs/eval/condition_agreement_check.json` (summary
also printed to stdout).

## 6. `train_dpo.py`

DPO/LoRA fine-tunes the policy model on one condition's preference pairs.
Requires the heavy `train` extra (`uv sync --extra train`) on a GPU box for
a real run. `--smoke-test` proves the training loop wires together on CPU
with a tiny public model and no real dataset.

```
# Real run (Day 2, on a rented GPU box):
uv run scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml

# Distributed (2-GPU, Accelerate + DeepSpeed ZeRO-2):
accelerate launch --config_file configs/accelerate_zero2.yaml \
    scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml

# CPU smoke test (tiny model, synthetic dataset, one training step):
uv run scripts/train_dpo.py --smoke-test
```

Saves a LoRA adapter to the config's `output_dir`.

## 7. `run_eval.py`

Runs one condition (base model, or base model + a trained LoRA adapter)
over the held-out eval set and scores how much it caves to unjustified
pushback (rule-based flip heuristic + judge-based sycophancy verdict).

```
# Real run (Day 2, on a GPU box):
uv run scripts/run_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --condition-name baseline
uv run scripts/run_eval.py --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter outputs/constitutional_dpo/ --condition-name constitutional_dpo

# Local dry run (no GPU/network, stub generation + mock judge):
uv run scripts/run_eval.py --condition-name smoke --dry-run --mock --limit 5
```

Writes `outputs/eval/<condition-name>/metrics.csv` and `summary.json`.

## 8. `plot_comparison.py`

Scans `outputs/eval/*/summary.json` and produces a grouped bar chart
comparing sycophancy rate and average judge score across whatever
conditions have been run so far.

```
uv run scripts/plot_comparison.py
```

Writes `outputs/eval/comparison.png`.

---

## Smoke-testing the whole pipeline

`smoke_test_pipeline.py` runs stages 3-7 (inject_pushback ->
generate_candidates --dry-run -> judge_rank --mock -> run_eval --mock) on a
tiny fixture (`tests/fixtures/smoke/`), then asserts every stage produced its
expected output file with the expected schema. No GPU, real API key, or
network access needed -- everything routes through each script's
`--dry-run`/`--mock` path. This is what CI runs (`.github/workflows/ci.yml`)
to catch schema mismatches between stages before a real, paid run.

```
uv run python scripts/smoke_test_pipeline.py
```

Run the unit test suite the same way CI does:

```
uv run pytest
```
