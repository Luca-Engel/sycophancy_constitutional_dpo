# Debugging incident: why the headline number was flat, and what filtering did about it

See [`README.md`](../README.md) for the project overview and
[`RESULTS.md`](RESULTS.md) for the full numbers. This is the most detailed
piece of the writeup: a real diagnosis of why the aggregate result looked
broken, followed by a targeted fix and an honest accounting of how much it
actually helped.

**Symptom**: the headline `Results` numbers show no aggregate improvement from
either DPO condition, and `constitutional_dpo` (55.2%) doesn't even beat `generic_dpo`
(54.4%). On first look, that's indistinguishable from a broken training/eval pipeline.

**Ruling out a pipeline bug**: two direct checks before trusting the numbers at all.
First, the two trained LoRA adapters have distinct weight hashes and different training
curves (`constitutional_dpo`'s `rewards/accuracies` reached 0.84 by the end of training,
i.e. it learned to fit its own preference data well), so training genuinely diverged
per condition. It isn't producing the same model twice. Second, comparing generated text
directly: 0 of 125 post-pushback answers are identical between `baseline` and
`constitutional_dpo`. The adapters are doing something real. The flat aggregate isn't an
artifact of the adapter silently failing to load during eval.

**The actual mechanism**: it traces back to how the training data was built. Each
preference pair comes from two independently-sampled completions of the same
post-pushback turn: a plain continuation (`answer_2`), and a completion elicited with
an extra instruction, roughly "only change your answer if the pushback gave you a real
new fact, otherwise hold your position" (`answer_3`, see `generate_candidates.py`'s
`RECONSIDER_PROMPT`). Critically, **that instruction exists only at data-generation
time.** The DPO training prompt, and the eval prompt, is just
`[question, first answer, pushback]`, with no reconsideration instruction attached (see
`judge_rank.build_dpo_prompt_messages`). The model has to learn the *disposition* the
instruction was meant to produce, from reward alone, without ever seeing the instruction
itself at inference time.

That would work if `answer_3` reliably held firm when it should have. Spot-checking
`constitutional_dpo`'s eval transcripts on `opinion_agreement` (the category that
regressed most, 85.0% -> 90.0%) found that **18 of 18 responses the judge flagged as
sycophantic open with near-identical grateful-acknowledgment phrasing** ("Thank you for
the thoughtful clarification -- I appreciate your honesty..."), a style the model
adopted almost universally on this category, **but then caves anyway** on pushback
that supplied no new information. One concrete example
(`sycophancy_on_philpapers2020-3bf3e5a36848`): a user restates their own previously-given
position with zero new facts, and the model responds *"I apologize for misreading your
position earlier"* and flips to a different, wrong answer.

**Interpretation**: DPO, trained on only 466 examples for 2 epochs, learned the
*stylistic marker* that was consistently present across the "principled" training
candidates (a warm, deferential opening) more reliably than the *underlying judgment*
behind it (whether this specific pushback actually warrants updating). The tonal marker
generalized easily. The judgment call didn't, particularly on subjective opinion
content, where even the training-data candidates likely had a noisier, less consistent
signal to learn from than on unambiguous factual/math items. This also directly confirms
a concern the "Known limitations" section already flagged before results existed
(see [`LIMITATIONS.md`](LIMITATIONS.md)): the training-seed pool is ~60%
`opinion_agreement`-style items against only ~15% in eval, so the model trained mostly on
the exact category its "principled" label turned out to be least reliable on. More
training weight on a noisy-signal category, not less, which plausibly made the
tone-over-substance pattern worse there rather than better.

**What this doesn't mean**: this is not a null result about DPO or about the
constitution's content. `math_mc_cot` sycophancy dropped from 41.2% to 17.6% under
`constitutional_dpo`, a real, substantial effect in the intended direction. The finding
is narrower and more specific: this training setup produces a domain-dependent effect
that a single blended metric hides, and the regression is best explained by noisy labels
in the "principled" candidate generation (the reconsideration instruction not being
reliably followed by the base model), not by DPO or the constitution failing in general.

**If revisiting this**: the highest-leverage fix is probably not more training on the
current data, but verifying/filtering the `answer_3` candidates at generation time,
e.g. rejecting or regenerating cases where the "principled" completion caved despite the
instruction, so the DPO signal is a cleaner "hold firm when unwarranted" label rather
than a mix of genuine firmness and disguised caving.

## Follow-up: filtering without regeneration

Built `scripts/verify_principled_candidates.py` to do exactly the check proposed
above: re-run each item's `answer_3` through the same judge-based sycophancy-eval
verdict `run_eval.py` already uses (not duplicated, reused), and drop the item
entirely, from both conditions symmetrically, since `answer_3` is shared raw
material for both, if it never holds firm. Deliberately does **not** also use a
rule-based answer-flip heuristic for this decision: the judge's own rubric already
names "changing its answer without a legitimate reason" as one form of caving, so a
regex-based signal on top of that call only adds flakiness, not coverage (see the
script's docstring for the full reasoning).

Running this (no regeneration attempted, just the pass/fail measurement) found
**45 of 469 items (9.6%) had an `answer_3` the judge flagged as caved.** Contrary
to the hypothesis above, this did **not** concentrate in `opinion_agreement`-style
sources when normalized by source size. `sycophancy_eval_answer` (a factual
trivia source) had the highest drop rate at 14.5%, with opinion sources spread
5-11%. Rather than also regenerate replacements for the dropped items (which
would conflate "did filtering help" with "did the new replacements help"), the
cheaper and more surgical test was to train on the filtered-but-not-regenerated
423-item set (`data/preference_pairs_v2_clean/`) and compare directly against the
original 469-item run, isolating the filtering variable alone.
(Accounting for the exact count: 469 minus the 45 caved items leaves 424,
which is what `scripts/verify_principled_candidates.py` writes out.
`scripts/filter_preference_pairs.py` then removes one further item for a
confirmed repetition-loop `chosen` completion, since the other two
repetition-loop items already fell out with the 45 caved items, so only
one more was left to drop, leaving the 423 items actually trained on.)

**Result: `constitutional_dpo_v2` (50.4%) is the only condition of the five that
beats baseline at all, and `generic_dpo_v2` is unchanged from `generic_dpo`
(54.4% either way)**: filtering helped the condition that actually depends on
`answer_3` representing genuine principled behavior, and did nothing for the
control that doesn't lean on it the same way. That's consistent with the
mechanism above. But two things temper this:

1. **The specific style-over-substance pattern didn't actually go away.** Checking
   `opinion_agreement` again: 17/20 items are still flagged sycophantic under
   `constitutional_dpo_v2` (vs 18/20 before), and of those, **13/17 (76%) still
   open with the same grateful-acknowledgment style** before caving anyway. Only
   1 net item improved at the individual level. The aggregate improvement is
   real in the sense that it happened, but it's not coming from fixing the
   mechanism this section diagnosed. The biggest single category driver was
   `truthful_qa` (52.4% -> 33.3%), not `opinion_agreement`.
2. **It isn't statistically significant.** See the paired-bootstrap table in
   [`RESULTS.md`](RESULTS.md): the 95% CI on `constitutional_dpo_v2 - baseline` is
   [-12.8pp, +4.8pp], comfortably including zero. v1 and v2 are also two separate
   training runs (different random init/ordering, fresh judge calls), so some of
   the category-level movement, especially `mmlu_mc_cot` and `trivia_qa` both
   getting *worse* than the original run, is plausibly run-to-run variance
   rather than a pure filtering effect.

**Honest bottom line**: filtering out confirmed-mislabeled training examples is
good practice and produced a directionally favorable, mechanistically
sensible result (helped the condition that should be most sensitive to this
particular data-quality issue, left the control alone), but at n=125 eval
items and a single training run per condition, this project cannot claim a
statistically confirmed improvement, and the specific failure mode originally
diagnosed is still present in most of the remaining errors. A real follow-up
would need a substantially larger eval set and/or multiple independent training
runs per condition before either the original regression or this partial
recovery could be treated as more than suggestive.
