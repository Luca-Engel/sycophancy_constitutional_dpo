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
```

Writes `data/generated/candidates.jsonl`.

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
uv run scripts/run_eval.py --model Qwen/Qwen2.5-3B-Instruct --condition-name baseline
uv run scripts/run_eval.py --model Qwen/Qwen2.5-3B-Instruct \
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
