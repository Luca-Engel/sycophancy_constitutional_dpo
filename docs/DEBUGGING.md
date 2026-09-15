# Debugging incident: why the headline number was flat, and what filtering did about it

See [`README.md`](../README.md) for the project overview and
[`RESULTS.md`](RESULTS.md) for the full numbers. This is the detailed
diagnosis of why the aggregate result looked broken, the fix, and an
honest account of how much it helped.

**Symptom**: the headline numbers show no aggregate improvement from
either DPO condition, and `constitutional_dpo` (55.2%) doesn't even beat
`generic_dpo` (54.4%). At first glance, indistinguishable from a broken
pipeline.

**Ruling out a pipeline bug**: the two trained LoRA adapters have distinct
weight hashes and different training curves (`constitutional_dpo`'s
`rewards/accuracies` reached 0.84), so training genuinely diverged per
condition. And 0 of 125 post-pushback answers are identical between
`baseline` and `constitutional_dpo`. The adapters are doing something
real, not silently failing to load.

**The mechanism**: each preference pair comes from two independently
sampled completions of the same post-pushback turn: a plain continuation
(`answer_2`), and one elicited with an extra instruction, roughly "only
change your answer if the pushback gave you a real new fact" (`answer_3`,
see `generate_candidates.py`'s `RECONSIDER_PROMPT`). That instruction
exists only at data-generation time. The DPO training and eval prompt is
just `[question, first answer, pushback]`, with no reconsideration
instruction attached. The model has to learn the *disposition* the
instruction was meant to produce, from reward alone, never seeing the
instruction itself at inference.

That would work if `answer_3` reliably held firm. Spot-checking
`constitutional_dpo`'s eval transcripts on `opinion_agreement` (the
category that regressed most, 85.0% -> 90.0%) found **18 of 18**
sycophancy-flagged responses open with near-identical grateful-
acknowledgment phrasing ("Thank you for the thoughtful clarification,
I appreciate your honesty...") **but then cave anyway** on pushback that
added no new information. One example: a user restates their own
previously-given position with zero new facts, and the model responds
*"I apologize for misreading your position earlier"* and flips to a
different, wrong answer.

**Interpretation**: DPO, trained on only 466 examples for 2 epochs,
learned the *stylistic marker* consistently present across "principled"
training candidates (a warm, deferential opening) more reliably than the
*judgment* behind it. The tone generalized easily, the judgment call
didn't, especially on subjective opinion content, where even the
training-data candidates likely had a noisier signal to learn from than
on unambiguous factual/math items. This confirms a concern already
flagged in [`LIMITATIONS.md`](LIMITATIONS.md) before results existed: the
training-seed pool is ~60% `opinion_agreement`-style items against only
~15% in eval, so the model trained mostly on the category where its
"principled" label was least reliable.

**What this doesn't mean**: not a null result about DPO or the
constitution's content. `math_mc_cot` sycophancy dropped from 41.2% to
17.6% under `constitutional_dpo`, a real effect in the intended
direction. The finding is narrower: this setup produces a
domain-dependent effect a single blended metric hides, best explained by
noisy labels in "principled" candidate generation, not by DPO or the
constitution failing in general.

**If revisiting this**: the highest-leverage fix is verifying and
filtering the `answer_3` candidates at generation time, rejecting or
regenerating cases where the "principled" completion caved despite the
instruction, for a cleaner "hold firm when unwarranted" signal.

## Follow-up: filtering without regeneration

Built `scripts/verify_principled_candidates.py` to do exactly that:
re-run each item's `answer_3` through the same judge-based sycophancy
verdict `run_eval.py` already uses, and drop the item from both
conditions symmetrically if it never holds firm (skips a rule-based
flip heuristic on top, since the judge's own rubric already covers
"changing its answer without a legitimate reason").

Result: **45 of 469 items (9.6%) had an `answer_3` the judge flagged as
caved.** Contrary to expectation, this did **not** concentrate in
`opinion_agreement`-style sources: `sycophancy_eval_answer` (factual
trivia) had the highest drop rate at 14.5%, opinion sources spread
5-11%. Rather than also regenerate replacements (which would conflate
"did filtering help" with "did new replacements help"), the more
surgical test was training on the filtered-but-not-regenerated 423-item
set (`data/preference_pairs_v2_clean/`) and comparing directly against
the original 469-item run. (Accounting: 469 minus 45 caved leaves 424,
what `verify_principled_candidates.py` writes. `filter_preference_pairs.py`
then drops one further repetition-loop item, leaving 423 trained on.)

**Result: `constitutional_dpo_v2` (50.4%) is the only condition of the
five that beats baseline, and `generic_dpo_v2` is unchanged from
`generic_dpo` (54.4% either way)**: filtering helped the condition that
depends on `answer_3` representing genuine principled behavior, and did
nothing for the control. Consistent with the mechanism above, but two
things temper this:

1. **The style-over-substance pattern didn't go away.** In
   `opinion_agreement`, 17/20 items are still flagged sycophantic under
   `constitutional_dpo_v2` (vs 18/20 before), and 13/17 (76%) of those
   still open with the same grateful-acknowledgment style before caving.
   Only 1 net item improved. The aggregate gain is real but isn't coming
   from fixing the diagnosed mechanism. The biggest category driver was
   `truthful_qa` (52.4% -> 33.3%), not `opinion_agreement`.
2. **It isn't statistically significant.** The 95% CI on
   `constitutional_dpo_v2 - baseline` is [-12.8pp, +4.8pp] (see
   [`RESULTS.md`](RESULTS.md)), comfortably including zero. v1 and v2 are
   also separate training runs, so some category-level movement
   (`mmlu_mc_cot`, `trivia_qa` both getting worse) is plausibly
   run-to-run variance rather than a pure filtering effect.

**Honest bottom line**: filtering out confirmed-mislabeled examples is
good practice and produced a directionally favorable, mechanistically
sensible result, but at n=125 and one training run per condition, this
project cannot claim a statistically confirmed improvement, and the
originally diagnosed failure mode is still present in most remaining
errors. A real follow-up needs a substantially larger eval set and/or
multiple independent training runs before treating either the original
regression or this partial recovery as more than suggestive.
