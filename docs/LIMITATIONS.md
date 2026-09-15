# Known limitations, and how the main confounds were controlled for

See [`README.md`](../README.md) for the project overview. This document has
two parts: the confound-control checks built into the pipeline (and what they
found), and the full list of known limitations/approximations that weren't
specifically controlled for.

## Confound controls

**Candidate order is randomized to isolate the constitution's effect.** The two
candidates a judge compares (`answer_2_sycophantic_candidate` and
`answer_3_principled_candidate`) aren't neutral, interchangeable samples.
`answer_3` is elicited with an extra "pause and reconsider" turn
(`generate_candidates.py`'s `RECONSIDER_PROMPT`) that `answer_2` never sees,
so the two differ systematically in *how they were produced*, not just in
content. If they were always shown to the judge in the same fixed
answer_2/answer_3 order, any position bias the judge has (a documented
LLM-judge failure mode) would be indistinguishable from a genuine
preference for one elicitation style, and generic_dpo, the whole point of
which is to be a clean, constitution-free control, could end up absorbing
the same confound instead of being a true null. `judge_common.py`'s
`assign_candidate_slots` fixes this: which slot ("answer_2" or "answer_3")
each candidate lands in is randomized independently per item and per
condition (deterministically, from the project seed), and
`judge_rank.py`'s `build_dpo_record` un-shuffles the judge's verdict back to
the correct candidate before writing the preference pair.
`scripts/check_judge_consistency.py` is built to spot-check that this is
actually working, by calling the judge twice per sampled item (real
ordering vs. deliberately flipped) and reporting how often the preferred
candidate changes purely because of position. **It has not yet been run
against the real dataset.** See "Known limitations" below and the
README's "Future work" section. Slot randomization is a sound design, but
its effectiveness in practice is currently an assumption, not a measured
result.

**A second, independent check asks a related question: do the two judge
rubrics even disagree in practice?** If the generic-quality judge and the
constitutional judge tend to prefer the same underlying candidate anyway,
generic_dpo would be training on data that isn't meaningfully different
from constitutional_dpo's, and the whole comparison between them would
show nothing regardless of whether the constitution works.
`scripts/check_condition_agreement.py` checks this directly on the
original 469-item dataset: for every item, it compares the actual text of
the response each condition's judge chose. The two judges picked the
identical response in only 170 of 469 cases, a **36.2% agreement rate**
(`outputs/eval/condition_agreement_check.json`). The two rubrics disagree
on nearly two-thirds of items, which is evidence that generic_dpo is
training on a genuinely different signal from constitutional_dpo, not a
relabeled copy of it, support for treating generic_dpo as a real
control rather than a confounded one.

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
  incident (see [`DEBUGGING.md`](DEBUGGING.md)). `scripts/check_judge_consistency.py`
  is built to cover one specific slice of this (position/order sensitivity,
  see "Confound controls" above), but as of this write-up it has not
  actually been run against the real dataset. It exists as tooling, not
  as a reported result, and even once run it would not be a substitute
  for a human spot-check of judge verdicts against hand labels (see the
  README's "Future work").
- **Eval rubric shares an author and some concepts with the constitution**:
  `run_eval.py`'s sycophancy-verdict rubric
  (`judge_common.SYCOPHANCY_EVAL_SYSTEM_PROMPT` /
  `build_sycophancy_eval_user_prompt`) never shows the judge
  `configs/constitution.md`, but both documents were written by the same
  person and lean on overlapping ideas ("unjustified pressure," "a
  legitimate reason to update"). That's intentional: the eval needs its
  own definition of sycophancy independent of training. But it means a
  constitutional_dpo win on this eval is not fully independent evidence
  from the constitution itself, and a reader should not treat the eval rubric
  as a neutral third party without having actually compared its wording
  against the constitution's.
- **Train/eval category-distribution mismatch**: per
  `data/train_seed/MANIFEST.md`, the training-seed pool is ~60%
  `opinion_agreement` items, while the eval-holdout set is only ~15%
  (round-robin category balancing caps any one category from dominating
  the smaller holdout set, see `split_eval_holdout.py`). So the model is
  trained mostly on stated-opinion-style pushback and evaluated mostly on
  factual-pushback-style items. This may or may not matter for
  generalization. It hasn't been checked, and the results write-up should
  say whether the per-category breakdown shows a gap between these two
  regimes.
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
  paired-bootstrap CIs in [`RESULTS.md`](RESULTS.md) show every condition-vs-baseline
  difference actually observed in this project, including the largest
  one, `constitutional_dpo_v2`'s 4-point improvement, has a 95% CI that
  includes zero. This isn't a theoretical caveat. It's a directly measured
  consequence of n=125. Any future rerun of this project should budget for
  a substantially larger eval set (or several independent training/eval
  runs per condition, to average out run-to-run training variance) before
  treating a point-estimate difference of this size as confirmed rather
  than suggestive.
