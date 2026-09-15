# Filtered DPO preference pairs: provenance

Filtered from `data/preference_pairs_v2` by `scripts/filter_preference_pairs.py`.
Raw judged data is untouched, this is a separate, derived copy.

## Exclusions

3 items excluded from **both** files (union, so the two conditions stay
aligned on the same item pool) because their `chosen` completion, in at
least one condition, is a confirmed, manually-verified catastrophic
repetition loop rather than a real answer. See
`scripts/filter_preference_pairs.py`'s module docstring for why this is a
hardcoded, human-reviewed list rather than an automated filter.

- `sycophancy_eval_answer-84175933d91b`: chosen completion is a ~70-repeat infinite loop asserting and negating 'Sunsilk' as a Swiss-mountain sunscreen brand, never resolves
- `sycophancy_eval_answer-877e6412496a`: chosen completion is a ~45-repeat loop asserting and negating 'Mia Farrow' as Jessica Rabbit's actress before eventually recovering
- `sycophancy_eval_answer-d3468517dbc2`: chosen completion is a ~90-repeat loop asserting and negating 'Jeffrey Combs' as the lead in *The Fly* (1986), never resolves, cut off by max_new_tokens

## Counts

- `constitutional_dpo.jsonl`: 424 total -> 423 kept (1 dropped)
- `generic_dpo.jsonl`: 424 total -> 423 kept (1 dropped)
