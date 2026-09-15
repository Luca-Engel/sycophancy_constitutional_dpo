# Known limitations, and how the main confounds were controlled for

See [`README.md`](../README.md) for the project overview. Two parts: the
confound-control checks built into the pipeline (and what they found),
and the limitations that weren't specifically controlled for.

## Confound controls

**Candidate order is randomized to isolate the constitution's effect.**
The two candidates a judge compares (`answer_2_sycophantic_candidate` and
`answer_3_principled_candidate`) aren't neutral, interchangeable samples:
`answer_3` is elicited with an extra "pause and reconsider" turn
(`generate_candidates.py`'s `RECONSIDER_PROMPT`) that `answer_2` never
sees. Shown to the judge in a fixed order, any judge position bias (a
documented LLM-judge failure mode) would be indistinguishable from a
genuine preference for one elicitation style, and generic_dpo, meant to
be a clean, constitution-free control, could absorb the same confound
instead of being a true null. `judge_common.py`'s
`assign_candidate_slots` fixes this by randomizing which slot each
candidate lands in, independently per item and condition (deterministic
from the project seed). `judge_rank.py`'s `build_dpo_record` un-shuffles
the verdict back to the correct candidate. `check_judge_consistency.py`
is built to spot-check this by calling the judge twice per sampled item
(real vs. flipped ordering), **but it hasn't been run against the real
dataset yet** (see "Known limitations" below). Slot randomization is
sound design, but its real-world effectiveness is currently an
assumption, not a measured result.

**A second check asks whether the two judge rubrics even disagree in
practice.** If the generic-quality judge and the constitutional judge
tend to prefer the same candidate anyway, generic_dpo would train on data
that isn't meaningfully different from constitutional_dpo's, and the
comparison would show nothing regardless of whether the constitution
works. `scripts/check_condition_agreement.py` checks this on the
original 469-item dataset by comparing the chosen response text under
each rubric. The two judges picked the identical response in only 170 of
469 cases, a **36.2% agreement rate**
(`outputs/eval/condition_agreement_check.json`), evidence that
generic_dpo trains on a genuinely different signal, not a relabeled copy,
supporting it as a real control.

## Known limitations and approximations

- **Rule-based answer-flip heuristic**: `run_eval.py`'s flip-rate metric
  compares an item's first answer against its post-pushback answer with a
  rule-based check (multiple-choice letter extraction, for instance).
  Cheap and fast, but will misfire on free-form answers that are
  semantically equivalent but lexically different, or the reverse, so
  it's a signal to read alongside the judge verdict, not ground truth.
- **Judge-based scoring noise**: sycophancy verdicts and rankings come
  from a single judge call per item, no self-consistency sampling or
  human validation. `check_judge_consistency.py` covers one slice of this
  (position sensitivity) but hasn't been run against the real dataset.
  Tooling, not a reported result, and even run it wouldn't substitute for
  a human spot-check against hand labels.
- **Eval rubric shares an author and concepts with the constitution**:
  `run_eval.py`'s sycophancy rubric never shows the judge
  `configs/constitution.md`, but both were written by the same person and
  share ideas ("unjustified pressure," "a legitimate reason to update").
  Intentional, since eval needs its own definition of sycophancy
  independent of training, but it means a constitutional_dpo win here
  isn't fully independent evidence.
- **Train/eval category mismatch**: the training-seed pool is ~60%
  `opinion_agreement` items, eval-holdout only ~15% (round-robin category
  balancing caps any category from dominating the smaller holdout set).
  So the model trains mostly on stated-opinion-style pushback and
  evaluates mostly on factual pushback. Whether this matters for
  generalization hasn't been checked.
- **Synthetic fallback data**: `fetch_sycophancy_data.py` falls back to a
  small bundled synthetic set only if every public source is unreachable.
  The actual run used real sources (MANIFEST files say "Fallback
  synthetic set used: False"), but any rerun in a restricted network
  should check that flag.
- **Small preference-dataset size**: target is 300-600 pairs per
  condition (`project.yaml`'s `target_preference_pairs: 400`), chosen to
  keep judge API cost low. Enough to see a training effect at this model
  scale, far smaller than a production dataset, so results should read as
  a directional demonstration, not a rigorously powered study.
- **125-item eval set is confirmed underpowered**: the bootstrap CIs in
  [`RESULTS.md`](RESULTS.md) show every observed difference, including
  the largest (`constitutional_dpo_v2`'s 4-point improvement), has a 95%
  CI that includes zero. A directly measured consequence of n=125, not a
  theoretical caveat. A future rerun should budget for a much larger eval
  set, or several independent runs per condition, before treating a
  difference this size as confirmed.
