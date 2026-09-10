# Constitutional DPO: Mitigating Sycophancy in a Small LLM via AI Feedback

A Constitutional-AI-style post-training pipeline that reduces sycophancy
(caving to unsubstantiated user pushback, or flattering a user's stated
opinion) in a small open instruction-tuned LLM, using AI-feedback preference
data and DPO fine-tuning. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the
full methodology, dataset, compute/budget, and day-by-day plan.

## Status

Under construction. This README currently documents only the repo
scaffolding, dependencies, and constitution — data generation, training,
and evaluation scripts land in later stages of the pipeline.

## Setup

```bash
# Install core dependencies (no heavy ML libs -- those are a separate, optional group)
uv sync

# Copy the environment template and fill in real values
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY (required) and HF_TOKEN (optional)
```

The heavy training dependencies (`torch`, `transformers`, `peft`, `trl`,
`accelerate`, `deepspeed`, `bitsandbytes`) are declared under the `train`
optional extra and are meant to be installed later, on a rented Linux GPU
box, via:

```bash
uv sync --extra train
```

They are intentionally not installed on this development machine.

## Project structure

```
configs/                   Config files: constitution.md (judge rubric), project.yaml (paths/model/seed settings)
data/eval_holdout/          Held-out sycophancy eval prompts -- never used for training
data/train_seed/             Seed prompts used to generate training preference data
data/generated/               Raw candidate model generations before judging
data/preference_pairs/         Final {prompt, chosen, rejected} DPO-format datasets (conditions B and C)
scripts/                    Data generation, judging, training, and evaluation scripts
tests/                      Automated tests for the scripts above
outputs/                    Training/eval run artifacts (checkpoints, logs, metrics, plots)
```
