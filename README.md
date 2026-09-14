# Constitutional DPO: Mitigating Sycophancy in a Small LLM via AI Feedback

This repo is a Constitutional-AI post-training pipeline: 
- sample candidate responses from a small instruction-tuned LLM on prompts that invite sycophancy (a stated opinion, or unsubstantiated pushback on a correct answer), 
- have a larger judge model critique and rank those candidates against a hand-written constitution, 
- turn the result into DPO preference pairs, 
- and fine-tune the small model on them. 

The core scientific question is: 
> Can the constitution specifically reduce sycophancy versus a plain-preference DPO control that uses the same pipeline but no constitution?
## Pipeline

```mermaid
flowchart TD
    A[fetch_sycophancy_data.py] --> B[split_eval_holdout.py]
    B -->|eval_holdout.jsonl, held out| H[run_eval.py]
    B -->|train_seed.jsonl| C[inject_pushback.py]
    C --> D[generate_candidates.py]
    D -->|answer_1, sycophantic, principled candidates| E[judge_rank.py]
    Const[configs/constitution.md] -.rubric.-> E
    E -->|generic_dpo.jsonl, no constitution| F[train_dpo.py]
    E -->|constitutional_dpo.jsonl, constitutional| G[train_dpo.py]
    F -->|LoRA adapter: generic_dpo| H
    G -->|LoRA adapter: constitutional_dpo| H
    H --> I[plot_comparison.py]
```

Three conditions get compared on the held-out set: **baseline** (base model,
untouched), **generic_dpo** (DPO on plain-quality preference pairs, no
constitution, which is the control), and **constitutional_dpo** (DPO on
constitutional AI-feedback pairs, which is the real treatment). generic_dpo vs.
constitutional_dpo isolates whether the constitution itself moves the
result, not just DPO in general.

## Setup

```bash
# Install core dependencies (no heavy ML libs -- those are a separate, optional group)
uv sync

# Copy the environment template and fill in real values
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY (required for the judge) and HF_TOKEN (optional)
```

Training dependencies (`torch`, `transformers`, `peft`, `trl`,
`accelerate`, `deepspeed`, `bitsandbytes`) live under the `train` optional
extra and are meant to be installed later on a GPU machine:

```bash
uv sync --extra train
```

## Running the pipeline

Every stage is a standalone script under `scripts/`, run in order, reading
its defaults from `configs/project.yaml`. This is a quick-start map with
one example command per stage. See [`scripts/README.md`](scripts/README.md)
for the full list of flags, output schemas, and dry-run/mock options for
every script.

```bash
uv run scripts/fetch_sycophancy_data.py        # -> data/raw/raw_prompts.jsonl
uv run scripts/split_eval_holdout.py           # -> data/eval_holdout/, data/train_seed/
uv run scripts/inject_pushback.py              # -> data/generated/pushback_prompts.jsonl
uv run scripts/generate_candidates.py          # -> data/generated/candidates.jsonl (needs GPU/train extra, though --dry-run works without one)
uv run scripts/judge_rank.py --max-calls 20    # -> data/preference_pairs/{generic,constitutional}_dpo.jsonl (needs ANTHROPIC_API_KEY, though --mock works without one)
uv run scripts/train_dpo.py --config configs/train_constitutional_dpo.yaml   # -> outputs/constitutional_dpo/ (needs GPU/train extra, though --smoke-test works without one)
uv run scripts/run_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --adapter outputs/constitutional_dpo/ --condition-name constitutional_dpo
uv run scripts/plot_comparison.py              # -> outputs/eval/comparison.png
```

Everything through stage 3 (`inject_pushback.py`) needs only the core
dependency group and can be run on a development machine. Stages 4-7 have a
`--dry-run`/`--mock`/`--smoke-test` path each, so the whole pipeline's
control flow and output schemas can be verified without a GPU, a real
judge API call, or network access:

```bash
uv run python scripts/smoke_test_pipeline.py   # runs stages 3-7 end to end on a tiny fixture
uv run pytest                                  # unit tests for every script
```

This is exactly what CI (`.github/workflows/ci.yml`) runs on every push.

## Data exploration

[`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb)
explores the input prompt corpus before any GPU stage runs: the
raw→dedup→split funnel, source/category balance between the train-seed and
eval-holdout pools, prompt length and the `opinion_agreement` persona
template, the pushback-injection taxonomy (generic vs. light-touch
template banks), reproduced data-quality checks, and a short scikit-learn
pass (TF-IDF distinctive vocabulary, PCA projections, a category
classifier baseline, and k-means clustering of pushback text against the
hand-assigned template banks).

## The constitution

[`configs/constitution.md`](configs/constitution.md) is the rubric the
judge model is shown when producing constitutional_dpo's preference data: 14
original principles (inspired by, not copied from, the appendix of
Anthropic's public Constitutional AI paper) about not flipping a correct
answer under unsubstantiated pressure, not flattering stated opinions,
updating promptly when the user is actually right, and weighing stakes
appropriately. generic_dpo's judge prompt asks for generic
helpfulness/correctness/clarity ranking and never sees this file. The gap
between generic_dpo and constitutional_dpo is the experiment.

**Candidate order is randomized to isolate that gap.** The two candidates a
judge compares (`answer_2_sycophantic_candidate` and
`answer_3_principled_candidate`) aren't neutral, interchangeable samples --
`answer_3` is elicited with an extra "pause and reconsider" turn
(`generate_candidates.py`'s `RECONSIDER_PROMPT`) that `answer_2` never sees,
so the two differ systematically in *how they were produced*, not just in
content. If they were always shown to the judge in the same fixed
answer_2/answer_3 order, any position bias the judge has (a documented
LLM-judge failure mode) would be indistinguishable from a genuine
preference for one elicitation style, and generic_dpo -- the whole point of
which is to be a clean, constitution-free control -- could end up absorbing
the same confound instead of being a true null. `judge_common.py`'s
`assign_candidate_slots` fixes this: which slot ("answer_2" or "answer_3")
each candidate lands in is randomized independently per item and per
condition (deterministically, from the project seed), and
`judge_rank.py`'s `build_dpo_record` un-shuffles the judge's verdict back to
the correct candidate before writing the preference pair.
`scripts/check_judge_consistency.py` spot-checks that this is actually
working, by calling the judge twice per sampled item (real ordering vs.
deliberately flipped) and reporting how often the preferred candidate
changes purely because of position.

## Compute & budget

Target: 20-50 CHF. LoRA/DPO training on a 3B
model runs on a single rented consumer GPU (RTX 4090 / A6000, well under an
hour per condition), and a short 2-GPU Accelerate + DeepSpeed ZeRO-2 run
produces a genuine distributed-training artifact. Judge API calls
(candidate critique + eval scoring, a few thousand Haiku calls) are the
other real cost, expected under $5. Full reasoning and a day-by-day
breakdown are in [`PROJECT_PLAN.md`](PROJECT_PLAN.md#4-compute--budget).

## Results

Evaluated on the full 125-item held-out eval set (`data/eval_holdout/eval_holdout.jsonl`),
`Qwen/Qwen3-4B-Instruct-2507`, five conditions -- the original three, plus a
follow-up pair (`_v2`) trained on a filtered dataset with 46 mislabeled
"principled" candidates removed (see "Debugging incident" below for why):

| Condition | n | Sycophancy rate | 95% Wilson CI |
|---|---|---|---|
| baseline | 125 | 54.4% | [45.7%, 62.9%] |
| generic_dpo | 125 | 54.4% | [45.7%, 62.9%] |
| constitutional_dpo | 125 | 55.2% | [46.5%, 63.6%] |
| generic_dpo_v2 | 125 | 54.4% | [45.7%, 62.9%] |
| constitutional_dpo_v2 | 125 | **50.4%** | [41.8%, 59.0%] |

![Sycophancy rate by condition](docs/comparison.png)

**The headline read: `constitutional_dpo_v2` has the lowest point estimate of any
condition, and is the only one that beats baseline at all -- but none of these
differences are statistically distinguishable from noise at this sample size.**
A paired bootstrap (5,000 resamples, resampling eval items with replacement --
paired because all five conditions were scored on the *same* 125 items) on the
differences that matter most:

| Comparison | Observed diff | 95% CI | Significant at 95%? |
|---|---|---|---|
| constitutional_dpo_v2 − baseline | −4.0pp | [−12.8pp, +4.8pp] | No |
| constitutional_dpo_v2 − constitutional_dpo (v1) | −4.8pp | [−14.4pp, +4.0pp] | No |
| constitutional_dpo_v2 − generic_dpo_v2 | −4.0pp | [−13.6pp, +4.8pp] | No |
| constitutional_dpo − baseline | +0.8pp | [−7.2pp, +8.8pp] | No |
| generic_dpo − baseline | 0.0pp | [−9.6pp, +9.6pp] | No |

Every 95% CI comfortably includes zero. **At n=125, this study does not have the
statistical power to confirm any of these effects are real rather than sampling
noise** -- a proper follow-up would need a substantially larger eval set (or many
more independent training/eval runs per condition) before treating any of these
percentages as a confirmed result. This is stated plainly rather than smoothed
over: the honest conclusion from this run is "directionally suggestive, not
statistically confirmed," not "constitutional training works."

### Per-category breakdown (v1: original 469-item, unfiltered dataset)

| Category | n | baseline | generic_dpo | constitutional_dpo |
|---|---|---|---|---|
| math_mc_cot | 17 | 41.2% | 23.5% | **17.6%** |
| truthful_qa | 21 | 57.1% | 42.9% | 52.4% |
| truthful_qa_mc | 21 | 66.7% | 61.9% | 61.9% |
| trivia_qa | 21 | 61.9% | 76.2% | 66.7% |
| mmlu_mc_cot | 21 | 23.8% | 33.3% | **42.9%** |
| opinion_agreement | 20 | 85.0% | 90.0% | **90.0%** |

DPO training produced a large reduction in sycophancy on tasks with a single
checkable answer (`math_mc_cot`), and made things measurably worse on
subjective/opinion pushback (`opinion_agreement`) and on `mmlu_mc_cot` recall
questions -- opposite-signed effects that cancel out in the blended aggregate
number above. See "Debugging incident" for the mechanism this points to, and
for what changed (and didn't) after filtering.

## Known limitations / approximations

- **Rule-based answer-flip heuristic**: `run_eval.py`'s flip-rate metric
  uses a rule-based comparison between an item's first answer and its
  post-pushback answer (e.g. multiple-choice letter extraction). This is a
  cheap, fast proxy and will misfire on free-form answers that are
  semantically equivalent but lexically different, or vice versa, so it's a
  signal to read alongside the judge-based verdict, not a ground truth on
  its own.
- **Judge-based scoring noise**: sycophancy verdicts and quality rankings
  come from a single judge-model call per item with no self-consistency
  sampling or human validation. Judge disagreement/inconsistency across
  reruns is expected and is itself a candidate topic for the debugging
  incident below. `scripts/check_judge_consistency.py` covers one specific
  slice of this (position/order sensitivity, see "The constitution" above)
  but is not a substitute for the human spot-check `docs/NEXT_STEPS.md`
  already calls for.
- **Eval rubric shares an author and some concepts with the constitution**:
  `run_eval.py`'s sycophancy-verdict rubric
  (`judge_common.SYCOPHANCY_EVAL_SYSTEM_PROMPT` /
  `build_sycophancy_eval_user_prompt`) never shows the judge
  `configs/constitution.md`, but both documents were written by the same
  person and lean on overlapping ideas ("unjustified pressure," "a
  legitimate reason to update"). That's intentional -- the eval needs its
  own definition of sycophancy independent of training -- but it means a
  constitutional_dpo win on this eval is not fully independent evidence
  from the constitution itself; a reader should not treat the eval rubric
  as a neutral third party without having actually compared its wording
  against the constitution's.
- **Train/eval category-distribution mismatch**: per
  `data/train_seed/MANIFEST.md`, the training-seed pool is ~60%
  `opinion_agreement` items, while the eval-holdout set is only ~15%
  (round-robin category balancing caps any one category from dominating
  the smaller holdout set -- see `split_eval_holdout.py`). So the model is
  trained mostly on stated-opinion-style pushback and evaluated mostly on
  factual-pushback-style items. This may or may not matter for
  generalization; it hasn't been checked, and the results write-up should
  say whether the per-category breakdown (already planned in
  `docs/NEXT_STEPS.md` §4-5) shows a gap between these two regimes.
- **Synthetic fallback data**: `fetch_sycophancy_data.py` falls back to a
  small bundled synthetic prompt set (`data/fallback_seed_prompts.jsonl`)
  only if every public network source is unreachable. The actual data run
  used the real public sources (see `data/train_seed/MANIFEST.md` and
  `data/eval_holdout/MANIFEST.md`, "Fallback synthetic set used: False"),
  but any rerun in a network-restricted environment should check this flag
  before trusting the resulting split.
- **Small preference-dataset size**: target size is 300-600 pairs per
  condition (`configs/project.yaml`'s `target_preference_pairs: 400`),
  chosen to keep judge API cost low. This is enough to see a training
  effect at this model scale but is far smaller than a production
  preference dataset, so results should be read as a directional
  demonstration, not a rigorously powered study.
- **125-item eval set is underpowered, confirmed (not just suspected)**: the
  paired-bootstrap CIs in `Results` show every condition-vs-baseline
  difference actually observed in this project -- including the largest
  one, `constitutional_dpo_v2`'s 4-point improvement -- has a 95% CI that
  includes zero. This isn't a theoretical caveat; it's a directly measured
  consequence of n=125. Any future rerun of this project should budget for
  a substantially larger eval set (or several independent training/eval
  runs per condition, to average out run-to-run training variance) before
  treating a point-estimate difference of this size as confirmed rather
  than suggestive.

## Debugging incident

**Symptom**: the headline `Results` numbers above show no aggregate improvement from
either DPO condition, and `constitutional_dpo` (55.2%) doesn't even beat `generic_dpo`
(54.4%) -- on first look, indistinguishable from a broken training/eval pipeline.

**Ruling out a pipeline bug**: two direct checks before trusting the numbers at all.
First, the two trained LoRA adapters have distinct weight hashes and different training
curves (`constitutional_dpo`'s `rewards/accuracies` reached 0.84 by the end of training,
i.e. it learned to fit its own preference data well) -- so training genuinely diverged
per condition, it isn't producing the same model twice. Second, comparing generated text
directly: 0 of 125 post-pushback answers are identical between `baseline` and
`constitutional_dpo`. The adapters are doing something real; the flat aggregate isn't an
artifact of the adapter silently failing to load during eval.

**The actual mechanism**: it traces back to how the training data was built. Each
preference pair comes from two independently-sampled completions of the same
post-pushback turn -- a plain continuation (`answer_2`), and a completion elicited with
an extra instruction: *"only change your answer if the pushback gave you a real new
fact; otherwise hold your position"* (`answer_3`, see `generate_candidates.py`'s
`RECONSIDER_PROMPT`). Critically, **that instruction exists only at data-generation
time.** The DPO training prompt -- and the eval prompt -- is just
`[question, first answer, pushback]`, with no reconsideration instruction attached (see
`judge_rank.build_dpo_prompt_messages`). The model has to learn the *disposition* the
instruction was meant to produce, from reward alone, without ever seeing the instruction
itself at inference time.

That would work if `answer_3` reliably held firm when it should have. Spot-checking
`constitutional_dpo`'s eval transcripts on `opinion_agreement` (the category that
regressed most, 85.0% -> 90.0%) found that **18 of 18 responses the judge flagged as
sycophantic open with near-identical grateful-acknowledgment phrasing** ("Thank you for
the thoughtful clarification -- I appreciate your honesty...") -- a style the model
adopted almost universally on this category -- **but then caves anyway** on pushback
that supplied no new information. One concrete example
(`sycophancy_on_philpapers2020-3bf3e5a36848`): a user restates their own previously-given
position with zero new facts, and the model responds *"I apologize for misreading your
position earlier"* and flips to a different, wrong answer.

**Interpretation**: DPO, trained on only 466 examples for 2 epochs, learned the
*stylistic marker* that was consistently present across the "principled" training
candidates (a warm, deferential opening) more reliably than the *underlying judgment*
behind it (whether this specific pushback actually warrants updating). The tonal marker
generalized easily; the judgment call didn't -- particularly on subjective opinion
content, where even the training-data candidates likely had a noisier, less consistent
signal to learn from than on unambiguous factual/math items. This also directly confirms
a concern the "Known limitations" section above already flagged before results existed:
the training-seed pool is ~60% `opinion_agreement`-style items against only ~15% in eval,
so the model trained mostly on the exact category its "principled" label turned out to
be least reliable on -- more training weight on a noisy-signal category, not less,
which plausibly made the tone-over-substance pattern worse there rather than better.

**What this doesn't mean**: this is not a null result about DPO or about the
constitution's content. `math_mc_cot` sycophancy dropped from 41.2% to 17.6% under
`constitutional_dpo` -- a real, substantial effect in the intended direction. The finding
is narrower and more specific: this training setup produces a domain-dependent effect
that a single blended metric hides, and the regression is best explained by noisy labels
in the "principled" candidate generation (the reconsideration instruction not being
reliably followed by the base model), not by DPO or the constitution failing in general.

**If revisiting this**: the highest-leverage fix is probably not more training on the
current data, but verifying/filtering the `answer_3` candidates at generation time --
e.g. rejecting or regenerating cases where the "principled" completion caved despite the
instruction, so the DPO signal is a cleaner "hold firm when unwarranted" label rather
than a mix of genuine firmness and disguised caving.

### Follow-up: filtering without regeneration

Built `scripts/verify_principled_candidates.py` to do exactly the check proposed
above: re-run each item's `answer_3` through the same judge-based sycophancy-eval
verdict `run_eval.py` already uses (not duplicated, reused), and drop the item
entirely -- from both conditions symmetrically, since `answer_3` is shared raw
material for both -- if it never holds firm. Deliberately does **not** also use a
rule-based answer-flip heuristic for this decision: the judge's own rubric already
names "changing its answer without a legitimate reason" as one form of caving, so a
regex-based signal on top of that call only adds flakiness, not coverage (see the
script's docstring for the full reasoning).

Running this (no regeneration attempted, just the pass/fail measurement) found
**45 of 469 items (9.6%) had an `answer_3` the judge flagged as caved.** Contrary
to the hypothesis above, this did **not** concentrate in `opinion_agreement`-style
sources when normalized by source size -- `sycophancy_eval_answer` (a factual
trivia source) had the highest drop rate at 14.5%, with opinion sources spread
5-11%. Rather than also regenerate replacements for the dropped items (which
would conflate "did filtering help" with "did the new replacements help"), the
cheaper and more surgical test was to train on the filtered-but-not-regenerated
423-item set (`data/preference_pairs_v2_clean/`) and compare directly against the
original 469-item run -- isolating the filtering variable alone.

**Result: `constitutional_dpo_v2` (50.4%) is the only condition of the five that
beats baseline at all, and `generic_dpo_v2` is unchanged from `generic_dpo`
(54.4% either way)** -- filtering helped the condition that actually depends on
`answer_3` representing genuine principled behavior, and did nothing for the
control that doesn't lean on it the same way. That's consistent with the
mechanism above. But two things temper this:

1. **The specific style-over-substance pattern didn't actually go away.** Checking
   `opinion_agreement` again: 17/20 items are still flagged sycophantic under
   `constitutional_dpo_v2` (vs 18/20 before), and of those, **13/17 (76%) still
   open with the same grateful-acknowledgment style** before caving anyway. Only
   1 net item improved at the individual level. The aggregate improvement is
   real in the sense that it happened, but it's not coming from fixing the
   mechanism this section diagnosed -- the biggest single category driver was
   `truthful_qa` (52.4% -> 33.3%), not `opinion_agreement`.
2. **It isn't statistically significant.** See the paired-bootstrap table in
   `Results` above -- the 95% CI on `constitutional_dpo_v2 - baseline` is
   [-12.8pp, +4.8pp], comfortably including zero. v1 and v2 are also two separate
   training runs (different random init/ordering, fresh judge calls), so some of
   the category-level movement -- especially `mmlu_mc_cot` and `trivia_qa` both
   getting *worse* than the original run -- is plausibly run-to-run variance
   rather than a pure filtering effect.

**Honest bottom line**: filtering out confirmed-mislabeled training examples is
good practice and produced a directionally favorable, mechanistically
sensible result (helped the condition that should be most sensitive to this
particular data-quality issue, left the control alone) -- but at n=125 eval
items and a single training run per condition, this project cannot claim a
statistically confirmed improvement, and the specific failure mode originally
diagnosed is still present in most of the remaining errors. A real follow-up
would need a substantially larger eval set and/or multiple independent training
runs per condition before either the original regression or this partial
recovery could be treated as more than suggestive.

## Project structure

```
configs/                   Config files: constitution.md (judge rubric), project.yaml (paths/model/seed settings), training and distributed-training configs
data/eval_holdout/          Held-out sycophancy eval prompts -- never used for training
data/train_seed/             Seed prompts used to generate training preference data
data/generated/               Pushback-injected prompts and candidate model generations before judging
data/preference_pairs/         Final {prompt, chosen, rejected} DPO-format datasets (generic_dpo and constitutional_dpo)
scripts/                    Data generation, judging, training, and evaluation scripts (see scripts/README.md)
notebooks/                 Data exploration notebook(s)
tests/                      Automated tests for the scripts above
outputs/                    Training/eval run artifacts (checkpoints, logs, metrics, plots)
automation/                 Logs and notes from the automated build sessions that scaffolded this repo
```
