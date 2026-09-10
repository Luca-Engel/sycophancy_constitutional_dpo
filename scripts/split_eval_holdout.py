"""Deduplicate the combined raw prompt pool and split into eval_holdout / train_seed.

Reads the raw, normalized (but not yet deduplicated) pool written by
``scripts/fetch_sycophancy_data.py``, deduplicates by normalized prompt text,
then carves off a stratified held-out evaluation set and a capped
training-seed pool with zero prompt-text overlap between the two.

Usage:
    uv run scripts/split_eval_holdout.py [--input data/raw_prompts.jsonl] [--seed 42]
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

# Target size for the held-out eval set (PROJECT_PLAN / subtask spec: ~100-150).
EVAL_HOLDOUT_TARGET = 125
# Cap on the training-seed pool -- later stages call paid judge/policy APIs
# per example, so this deliberately stays well under "everything left over".
TRAIN_SEED_CAP = 500


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


def stratified_split(
    examples: list[dict],
    seed: int,
    eval_target: int = EVAL_HOLDOUT_TARGET,
    train_cap: int = TRAIN_SEED_CAP,
) -> tuple[list[dict], list[dict]]:
    """Split deduplicated examples into (eval_holdout, train_seed).

    Stratifies by category: shuffles within each category, then round-robins
    across categories when filling the eval set so no single category
    dominates it. Everything not selected for eval becomes the train-seed
    pool, shuffled and capped at ``train_cap``.
    """
    rng = random.Random(seed)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_category[ex["category"]].append(ex)

    categories = sorted(by_category.keys())
    for cat in categories:
        rng.shuffle(by_category[cat])

    remaining = {cat: list(rows) for cat, rows in by_category.items()}
    eval_holdout: list[dict] = []
    idx = 0
    while len(eval_holdout) < eval_target and any(remaining.values()):
        cat = categories[idx % len(categories)]
        if remaining[cat]:
            eval_holdout.append(remaining[cat].pop())
        idx += 1

    train_pool = [ex for rows in remaining.values() for ex in rows]
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
                "using the configured seed, and round-robined across categories "
                "into the eval holdout until it reaches its target size, so no "
                "single category dominates the held-out set. Everything left over "
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
