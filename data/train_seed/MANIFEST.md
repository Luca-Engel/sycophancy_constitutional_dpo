# Sycophancy prompt data: provenance and split

- Raw examples gathered (pre-dedup): 600
- After dedup by normalized prompt text: 594
- Eval holdout size: 125
- Train seed pool size: 469
- Split RNG seed: 42 (from configs/project.yaml)
- Fallback synthetic set used: False

## Counts by source

Eval holdout:
- sycophancy_eval_answer: 32
- sycophancy_eval_are_you_sure: 74
- sycophancy_on_nlp_survey: 6
- sycophancy_on_philpapers2020: 7
- sycophancy_on_political_typology_quiz: 6

Train seed:
- sycophancy_eval_answer: 112
- sycophancy_eval_are_you_sure: 76
- sycophancy_on_nlp_survey: 94
- sycophancy_on_philpapers2020: 93
- sycophancy_on_political_typology_quiz: 94

## Counts by category

Eval holdout:
- factual_qa/aqua_mc: 9
- factual_qa/math_mc_cot: 20
- factual_qa/mmlu_mc_cot: 20
- factual_qa/trivia_qa: 19
- factual_qa/truthful_qa: 19
- factual_qa/truthful_qa_mc: 19
- opinion_agreement: 19

Train seed:
- factual_qa/math_mc_cot: 23
- factual_qa/mmlu_mc_cot: 8
- factual_qa/trivia_qa: 93
- factual_qa/truthful_qa: 49
- factual_qa/truthful_qa_mc: 15
- opinion_agreement: 281

## Method

1. `scripts/fetch_sycophancy_data.py` gathers prompts from public sycophancy-eval sources (anthropics/evals GitHub repo and the meg-tong/sycophancy-eval Hugging Face mirror), normalizing every row to `{id, source, prompt, category}`. Falls back to a small bundled synthetic set only if every network source fails.
2. `scripts/split_eval_holdout.py` deduplicates by normalized (lowercased, whitespace-collapsed) prompt text, then splits: examples are grouped by category, shuffled within each category using the configured seed, and round-robined across categories into the eval holdout until it reaches its target size, so no single category dominates the held-out set. Everything left over is shuffled and capped to form the train-seed pool.
3. The eval holdout and train-seed pool are disjoint by construction (each example is drawn from the same deduplicated pool exactly once) and this is additionally asserted in code and covered by tests.
4. `data/eval_holdout/eval_holdout.jsonl` is never touched again until final evaluation. `data/train_seed/train_seed.jsonl` is the pool later stages turn into pushback-injected training prompts and, eventually, judge-labeled preference pairs.
