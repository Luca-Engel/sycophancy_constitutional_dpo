# Verified principled-candidate provenance

Verified from `data/generated/candidates.jsonl` by `scripts/verify_principled_candidates.py`.
Each item's `answer_3_principled_candidate` was checked against the item's own pushback (rule-based flip detection + a judge-based cave verdict, both reused from `run_eval.py`) and regenerated up to 0 time(s) if it caved. Items that never held firm are dropped entirely, not written to the output, so they never reach judge_rank.py for either DPO condition. See README.md's "Debugging incident" section for why.

This is the direct-comparison variant: filter out items with a mislabeled "principled" candidate but do NOT regenerate replacements, so the effect of cleaning the labels can be measured in isolation from the effect of adding new candidates.

## Counts (this run)

- kept (held firm, possibly after regeneration): 424
- dropped (never held firm after 0 regeneration(s)): 45
- total decided across all runs so far: 469 / 469

## Regenerations needed (this run)

- held firm on first try: 424 item(s)

## Dropped items by source (cumulative)

- `sycophancy_eval_answer`: 16 / 110 (14.5%)
- `sycophancy_on_nlp_survey`: 10 / 94 (10.6%)
- `sycophancy_eval_are_you_sure`: 7 / 79 (8.9%)
- `sycophancy_on_political_typology_quiz`: 7 / 92 (7.6%)
- `sycophancy_on_philpapers2020`: 5 / 94 (5.3%)

Full list of dropped ids: `data/generated/candidates_verified_dropped_ids.jsonl`.
