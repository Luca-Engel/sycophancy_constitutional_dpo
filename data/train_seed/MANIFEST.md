# Sycophancy prompt data: provenance and split

- Raw examples gathered (pre-dedup): 600
- After dedup by normalized prompt text: 594
- Eval holdout size: 125
- Train seed pool size: 469
- Split RNG seed: 42 (from configs/project.yaml)
- Fallback synthetic set used: False

## Counts by source

Eval holdout:
- sycophancy_eval_answer: 34
- sycophancy_eval_are_you_sure: 71
- sycophancy_on_nlp_survey: 6
- sycophancy_on_philpapers2020: 6
- sycophancy_on_political_typology_quiz: 8

Train seed:
- sycophancy_eval_answer: 110
- sycophancy_eval_are_you_sure: 79
- sycophancy_on_nlp_survey: 94
- sycophancy_on_philpapers2020: 94
- sycophancy_on_political_typology_quiz: 92

## Counts by category

Eval holdout:
- factual_qa/aqua_mc: 4
- factual_qa/math_mc_cot: 17
- factual_qa/mmlu_mc_cot: 21
- factual_qa/trivia_qa: 21
- factual_qa/truthful_qa: 21
- factual_qa/truthful_qa_mc: 21
- opinion_agreement: 20

Train seed:
- factual_qa/aqua_mc: 5
- factual_qa/math_mc_cot: 26
- factual_qa/mmlu_mc_cot: 7
- factual_qa/trivia_qa: 91
- factual_qa/truthful_qa: 47
- factual_qa/truthful_qa_mc: 13
- opinion_agreement: 280

## Method

1. `scripts/fetch_sycophancy_data.py` gathers prompts from public sycophancy-eval sources (anthropics/evals GitHub repo and the meg-tong/sycophancy-eval Hugging Face mirror), normalizing every row to `{id, source, prompt, category}`. Falls back to a small bundled synthetic set only if every network source fails.
2. `scripts/split_eval_holdout.py` deduplicates by normalized (lowercased, whitespace-collapsed) prompt text, then splits: examples are grouped by category, shuffled within each category using the configured seed, and given an equal round-robin eval quota per allocation group (aqua_mc is grouped with math_mc_cot so it isn't fully drained into eval; every other category is its own group), so no single category dominates the held-out set. Each group's quota is split back out across its member categories proportional to their share of the group's pool, so every category keeps its own label and a deterministic presence on both sides. Everything left over is shuffled and capped to form the train-seed pool.
3. The eval holdout and train-seed pool are disjoint by construction (each example is drawn from the same deduplicated pool exactly once) and this is additionally asserted in code and covered by tests.
4. `data/eval_holdout/eval_holdout.jsonl` is never touched again until final evaluation. `data/train_seed/train_seed.jsonl` is the pool later stages turn into pushback-injected training prompts and, eventually, judge-labeled preference pairs.
