# Verified principled-candidate provenance

Verified from `data/generated/candidates.jsonl` by
`scripts/verify_principled_candidates.py --max-regenerations 0` (no regeneration
attempted -- a pure measurement of how many `answer_3_principled_candidate`
values, as originally generated, the judge's sycophancy-eval verdict flags as
having caved). Items that failed are dropped entirely, not written to this
file, so they never reach `judge_rank.py` for either DPO condition -- see
README.md's "Debugging incident" section for why this check exists, and why
the exclusion needs to happen upstream of both conditions.

This is the direct-comparison variant discussed in the project's own
analysis: filter out items with a mislabeled "principled" candidate, but do
NOT regenerate replacements, so the effect of cleaning the labels can be
measured in isolation from the effect of adding newly-generated candidates
back in.

## Counts

- kept (held firm on the original generation): 424
- dropped (judge flagged the original answer_3 as caved): 45
- total: 469

## Dropped items by source

- `sycophancy_eval_answer`: 16 / 110 (14.5%)
- `sycophancy_on_nlp_survey`: 10 / 94 (10.6%)
- `sycophancy_eval_are_you_sure`: 7 / 79 (8.9%)
- `sycophancy_on_political_typology_quiz`: 7 / 92 (7.6%)
- `sycophancy_on_philpapers2020`: 5 / 94 (5.3%)

Full list of dropped ids: `data/generated/candidates_verified_dropped_ids.jsonl`.
