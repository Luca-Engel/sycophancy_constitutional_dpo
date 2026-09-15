"""Deduplicate the combined raw prompt pool and split into eval_holdout / train_seed.

Reads the raw, normalized (but not yet deduplicated) pool written by
``scripts/fetch_sycophancy_data.py``, deduplicates by normalized prompt text,
then carves off a stratified held-out evaluation set and a capped
training-seed pool with zero prompt-text overlap between the two.

Usage:
    uv run scripts/split_eval_holdout.py [--input data/raw/raw_prompts.jsonl] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("split_eval_holdout")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Target size for the held-out eval set (project spec target: ~100-150).
EVAL_HOLDOUT_TARGET = 125
# Cap on the training-seed pool -- later stages call paid judge/policy APIs
# per example, so this deliberately stays well under "everything left over".
TRAIN_SEED_CAP = 500

# Categories pooled together when computing how many eval slots to hand out,
# so a very small category (aqua_mc, n=9) doesn't get entirely drained into
# eval before it has a chance to contribute anything to train_seed. Each
# key's examples keep their own `category` label in the output -- the merge
# only affects the eval/train *count* for that category, which is split back
# out proportional to its share of the merged pool (see _proportional_split).
ALLOCATION_GROUP = {
    "factual_qa/aqua_mc": "factual_qa/math_mc_cot",
}

# NOTE: opinion_agreement (a single templated persona-question format,
# confirmed near-perfectly separable by a TF-IDF classifier in the
# data-exploration notebook) ends up ~60% of train_seed. An earlier version
# of this script capped its train share, but that conflicts with
# target_preference_pairs=400 in configs/project.yaml: train_seed=469 was
# deliberately sized to leave slack above 400 for judge-call failures,
# and a cap tight enough to matter for diversity
# (e.g. 40%) drops train_seed to ~315, already short of 400 before any
# failures. If opinion_agreement's share turns out to hurt training in
# practice, address it when trimming 469 -> 400 during pair curation
# (scripts/judge_rank.py), which already has that slack budgeted in,
# rather than here where it costs the safety margin.


def normalize_for_dedup(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def dedup_examples(examples: list[dict]) -> list[dict]:
    """Drop examples whose normalized prompt text has already been seen."""
    seen: set[str] = set()
    out = []
    for ex in examples:
        key = normalize_for_dedup(ex["prompt"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def _round_robin_quota(pool_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Equal round-robin allocation of ``target`` slots across ``pool_sizes``.

    Cycles through keys in sorted order, taking one slot at a time from
    whichever keys still have pool left, until ``target`` is reached or every
    pool is exhausted. No key can be allocated more than its own pool size.
    """
    keys = sorted(pool_sizes.keys())
    remaining = dict(pool_sizes)
    quota = {k: 0 for k in keys}
    idx = 0
    taken = 0
    while taken < target and any(remaining.values()):
        k = keys[idx % len(keys)]
        if remaining[k] > 0:
            remaining[k] -= 1
            quota[k] += 1
            taken += 1
        idx += 1
    return quota


def _proportional_split(sub_pool_sizes: dict[str, int], quota: int) -> dict[str, int]:
    """Split ``quota`` across ``sub_pool_sizes`` proportional to each size.

    Uses largest-remainder rounding so the parts sum exactly to ``quota``,
    and never assigns a sub-category more than its own pool size.
    """
    total = sum(sub_pool_sizes.values())
    if total == 0:
        return {k: 0 for k in sub_pool_sizes}
    raw = {k: quota * n / total for k, n in sub_pool_sizes.items()}
    assigned = {k: min(int(v), sub_pool_sizes[k]) for k, v in raw.items()}
    remainder = quota - sum(assigned.values())
    by_fraction = sorted(sub_pool_sizes, key=lambda k: raw[k] - assigned[k], reverse=True)
    for k in by_fraction:
        if remainder <= 0:
            break
        if assigned[k] < sub_pool_sizes[k]:
            assigned[k] += 1
            remainder -= 1
    return assigned


def stratified_split(
    examples: list[dict],
    seed: int,
    eval_target: int = EVAL_HOLDOUT_TARGET,
    train_cap: int = TRAIN_SEED_CAP,
) -> tuple[list[dict], list[dict]]:
    """Split deduplicated examples into (eval_holdout, train_seed).

    Stratifies by category: shuffles within each category, then computes an
    equal round-robin eval quota per *allocation group* (categories merged
    via ``ALLOCATION_GROUP`` share one group so a tiny category isn't fully
    drained into eval -- see its docstring). Each group's quota is split back
    out across its member categories proportional to their share of the
    group's pool, so every category keeps its own label and gets a
    deterministic, non-zero presence on both sides when its pool allows.

    Everything not selected for eval becomes the train-seed pool, shuffled
    and capped at ``train_cap``.
    """
    rng = random.Random(seed)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_category[ex["category"]].append(ex)

    for cat in by_category:
        rng.shuffle(by_category[cat])

    group_of = {cat: ALLOCATION_GROUP.get(cat, cat) for cat in by_category}
    members: dict[str, list[str]] = defaultdict(list)
    for cat, grp in group_of.items():
        members[grp].append(cat)

    group_pool_sizes = {grp: sum(len(by_category[c]) for c in cats) for grp, cats in members.items()}
    group_quota = _round_robin_quota(group_pool_sizes, eval_target)

    eval_holdout: list[dict] = []
    train_pool: list[dict] = []
    for grp, cats in members.items():
        sub_sizes = {c: len(by_category[c]) for c in cats}
        sub_quota = _proportional_split(sub_sizes, group_quota[grp])
        for c in cats:
            n = sub_quota[c]
            eval_holdout.extend(by_category[c][:n])
            train_pool.extend(by_category[c][n:])

    rng.shuffle(eval_holdout)
    rng.shuffle(train_pool)

    if len(train_pool) > train_cap:
        train_pool = train_pool[:train_cap]

    return eval_holdout, train_pool


def _counts_by(examples: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex[field]] += 1
    return dict(sorted(counts.items()))


def build_manifest(
    raw_count: int,
    deduped_count: int,
    eval_holdout: list[dict],
    train_seed: list[dict],
    seed: int,
    used_fallback: bool,
) -> str:
    lines = [
        "# Sycophancy prompt data: provenance and split",
        "",
        f"- Raw examples gathered (pre-dedup): {raw_count}",
        f"- After dedup by normalized prompt text: {deduped_count}",
        f"- Eval holdout size: {len(eval_holdout)}",
        f"- Train seed pool size: {len(train_seed)}",
        f"- Split RNG seed: {seed} (from configs/project.yaml)",
        f"- Fallback synthetic set used: {used_fallback}",
        "",
        "## Counts by source",
        "",
        "Eval holdout:",
    ]
    for src, n in _counts_by(eval_holdout, "source").items():
        lines.append(f"- {src}: {n}")
    lines.append("")
    lines.append("Train seed:")
    for src, n in _counts_by(train_seed, "source").items():
        lines.append(f"- {src}: {n}")
    lines.append("")
    lines.append("## Counts by category")
    lines.append("")
    lines.append("Eval holdout:")
    for cat, n in _counts_by(eval_holdout, "category").items():
        lines.append(f"- {cat}: {n}")
    lines.append("")
    lines.append("Train seed:")
    for cat, n in _counts_by(train_seed, "category").items():
        lines.append(f"- {cat}: {n}")
    lines.extend(
        [
            "",
            "## Method",
            "",
            (
                "1. `scripts/fetch_sycophancy_data.py` gathers prompts from public "
                "sycophancy-eval sources (anthropics/evals GitHub repo and the "
                "meg-tong/sycophancy-eval Hugging Face mirror), normalizing every "
                "row to `{id, source, prompt, category}`. Falls back to a small "
                "bundled synthetic set only if every network source fails."
            ),
            (
                "2. `scripts/split_eval_holdout.py` deduplicates by normalized "
                "(lowercased, whitespace-collapsed) prompt text, then splits: "
                "examples are grouped by category, shuffled within each category "
                "using the configured seed, and given an equal round-robin eval "
                "quota per allocation group (aqua_mc is grouped with "
                "math_mc_cot so it isn't fully drained into eval; every other "
                "category is its own group), so no single category dominates "
                "the held-out set. Each group's quota is split back out across "
                "its member categories proportional to their share of the "
                "group's pool, so every category keeps its own label and a "
                "deterministic presence on both sides. Everything left over "
                "is shuffled and capped to form the train-seed pool."
            ),
            (
                "3. The eval holdout and train-seed pool are disjoint by "
                "construction (each example is drawn from the same deduplicated "
                "pool exactly once) and this is additionally asserted in code and "
                "covered by tests."
            ),
            (
                "4. `data/eval_holdout/eval_holdout.jsonl` is never touched again "
                "until final evaluation. `data/train_seed/train_seed.jsonl` is the "
                "pool later stages turn into pushback-injected training prompts "
                "and, eventually, judge-labeled preference pairs."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _load_seed_from_config() -> int:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "raw" / "raw_prompts.jsonl"))
    parser.add_argument("--eval-output", default=str(REPO_ROOT / "data" / "eval_holdout" / "eval_holdout.jsonl"))
    parser.add_argument("--train-output", default=str(REPO_ROOT / "data" / "train_seed" / "train_seed.jsonl"))
    parser.add_argument("--seed", type=int, default=None, help="Override configs/project.yaml's seed.")
    parser.add_argument("--eval-target", type=int, default=EVAL_HOLDOUT_TARGET)
    parser.add_argument("--train-cap", type=int, default=TRAIN_SEED_CAP)
    parser.add_argument(
        "--used-fallback",
        action="store_true",
        help="Record in the manifest that the raw pool came from the fallback set (informational only).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    seed = args.seed if args.seed is not None else _load_seed_from_config()

    raw_examples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_examples.append(json.loads(line))

    deduped = dedup_examples(raw_examples)
    logger.info("loaded %d raw examples, %d after dedup", len(raw_examples), len(deduped))

    eval_holdout, train_seed = stratified_split(
        deduped, seed=seed, eval_target=args.eval_target, train_cap=args.train_cap
    )

    eval_prompts = {normalize_for_dedup(ex["prompt"]) for ex in eval_holdout}
    train_prompts = {normalize_for_dedup(ex["prompt"]) for ex in train_seed}
    overlap = eval_prompts & train_prompts
    assert not overlap, f"eval_holdout and train_seed overlap on {len(overlap)} prompts"

    eval_path = Path(args.eval_output)
    train_path = Path(args.train_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with eval_path.open("w", encoding="utf-8") as f:
        for ex in eval_holdout:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with train_path.open("w", encoding="utf-8") as f:
        for ex in train_seed:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    manifest = build_manifest(
        raw_count=len(raw_examples),
        deduped_count=len(deduped),
        eval_holdout=eval_holdout,
        train_seed=train_seed,
        seed=seed,
        used_fallback=args.used_fallback,
    )
    (eval_path.parent / "MANIFEST.md").write_text(manifest, encoding="utf-8")
    (train_path.parent / "MANIFEST.md").write_text(manifest, encoding="utf-8")

    logger.info("wrote %d eval_holdout examples to %s", len(eval_holdout), eval_path)
    logger.info("wrote %d train_seed examples to %s", len(train_seed), train_path)


if __name__ == "__main__":
    main()
