"""Judge self-consistency / position-bias check.

``scripts/judge_rank.py`` already randomizes, per item and condition, which
physical prompt slot ("answer_2" or "answer_3") each candidate lands in
(``judge_common.assign_candidate_slots``), specifically so judge position
bias can't be silently confounded with which candidate is which (see
``judge_common.py``'s module docstring, "Slot randomization"). This script
is the direct check that the fix is actually doing something: for a sample
of ``candidates.jsonl`` items, it calls the judge *twice* per item/condition
-- once under the slot assignment ``judge_rank.py`` would really use, and
once under the deliberately opposite assignment -- and checks whether the
judge still prefers the same underlying candidate (sycophantic-style vs
principled/reconsideration-style) both times.

If the judge is reasoning about content, the preferred *role* should match
across both calls (only the reported slot label should flip). If the judge
has a position bias, some items will flip which role "wins" purely because
of which slot it was shown in -- that's the failure mode slot randomization
protects the dataset from, and this script quantifies how often it would
have happened.

This makes 2x the judge calls of a normal judge_rank.py run for the sampled
items, so keep --sample-size small for a real (non-mocked) run -- this is a
spot-check, not something meant to run over the whole dataset.

Mock dry run (no API key, no network):
    uv run scripts/check_judge_consistency.py --mock --sample-size 5

Real run (small, cheap sample):
    uv run scripts/check_judge_consistency.py --sample-size 20 --condition constitutional
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import judge_common as jc
import judge_rank as jr

logger = logging.getLogger("check_judge_consistency")

REPO_ROOT = Path(__file__).resolve().parent.parent


def opposite_slots(slots: dict[str, str]) -> dict[str, str]:
    """The other of the two possible slot assignments."""
    return {role: ("answer_3" if slot == "answer_2" else "answer_2") for role, slot in slots.items()}


def sample_items(items: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Deterministic sample of up to ``sample_size`` items, seeded
    independently of assign_candidate_slots's per-item RNGs so the sample
    selection itself doesn't correlate with slot assignment."""
    rng = random.Random(f"{seed}:consistency-sample")
    if len(items) <= sample_size:
        return list(items)
    return rng.sample(items, sample_size)


def _resolve_role(verdict_chosen_slot: str, slots: dict[str, str]) -> str:
    role_by_slot = {slot: role for role, slot in slots.items()}
    return role_by_slot[verdict_chosen_slot]


def check_item_consistency(
    item: dict,
    condition: str,
    seed: int,
    *,
    mock: bool,
    client=None,
    model: str | None = None,
    constitution_text: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Call the judge twice for one item/condition: once under the real
    (assign_candidate_slots) ordering, once under the deliberately flipped
    ordering. Returns a report dict; raises judge_common.JudgeError if
    either real call fails after all retries (mock never raises)."""
    actual_slots = jc.assign_candidate_slots(item["id"], seed, condition)
    flipped_slots = opposite_slots(actual_slots)

    actual_verdict = jr.get_verdict(
        item,
        condition,
        mock=mock,
        slots=actual_slots,
        client=client,
        model=model,
        constitution_text=constitution_text,
        max_retries=max_retries,
    )
    flipped_verdict = jr.get_verdict(
        item,
        condition,
        mock=mock,
        slots=flipped_slots,
        client=client,
        model=model,
        constitution_text=constitution_text,
        max_retries=max_retries,
    )

    actual_role = _resolve_role(actual_verdict["chosen"], actual_slots)
    flipped_role = _resolve_role(flipped_verdict["chosen"], flipped_slots)

    return {
        "id": item["id"],
        "condition": condition,
        "actual_slots": actual_slots,
        "actual_chosen_role": actual_role,
        "flipped_slots": flipped_slots,
        "flipped_chosen_role": flipped_role,
        "consistent": actual_role == flipped_role,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate consistency rate overall and per condition."""

    def _rate(rows: list[dict]) -> float | None:
        return sum(1 for r in rows if r["consistent"]) / len(rows) if rows else None

    by_condition: dict[str, dict] = {}
    for condition in sorted({r["condition"] for r in results}):
        rows = [r for r in results if r["condition"] == condition]
        by_condition[condition] = {
            "n": len(rows),
            "consistency_rate": _rate(rows),
            "n_inconsistent": sum(1 for r in rows if not r["consistent"]),
        }

    return {
        "n_checks": len(results),
        "consistency_rate": _rate(results),
        "n_inconsistent": sum(1 for r in results if not r["consistent"]),
        "by_condition": by_condition,
    }


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--candidates-file",
        default=None,
        help="Path to candidates.jsonl. Defaults to <generated_dir>/candidates.jsonl from configs/project.yaml.",
    )
    parser.add_argument(
        "--condition",
        choices=("constitutional", "generic", "both"),
        default="both",
        help="Which judge rubric(s) to check.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of candidates.jsonl items to sample (each costs 2 judge calls per condition checked).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Hard cap on total judge calls (real or mocked) made in this invocation. Safety net for API spend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override configs/project.yaml's seed (used for sampling and must match the seed judge_rank.py "
        "was/will be run with for 'actual_slots' to reflect the real dataset's assignment).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the full per-item report JSON. Defaults to outputs/eval/judge_consistency_check.json.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic fake judge instead of calling the real API. No network access.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_config()
    seed = args.seed if args.seed is not None else cfg["seed"]

    candidates_path = (
        Path(args.candidates_file)
        if args.candidates_file
        else REPO_ROOT / cfg["paths"]["generated_dir"] / "candidates.jsonl"
    )
    items = load_jsonl(candidates_path)
    sample = sample_items(items, args.sample_size, seed)
    conditions = ["constitutional", "generic"] if args.condition == "both" else [args.condition]

    logger.info(
        "%d total candidates, sampled %d, checking condition(s): %s (2 judge calls per item per condition)",
        len(items),
        len(sample),
        conditions,
    )

    client = None
    model = None
    constitution_text = None
    judge_cfg = cfg.get("judge", {})
    max_retries = judge_cfg.get("max_retries", 3)

    if not args.mock:
        try:
            api_key = jc.get_api_key()
        except jc.MissingAPIKeyError as exc:
            logger.error(str(exc))
            raise SystemExit(1)
        client = jc.build_anthropic_client(api_key)
        model = judge_cfg["model"]
        constitution_text = jc.load_constitution(REPO_ROOT / cfg["paths"]["constitution_file"])

    results = []
    call_count = 0
    stopped_early = False
    for item in sample:
        if stopped_early:
            break
        for condition in conditions:
            if args.max_calls is not None and call_count + 2 > args.max_calls:
                logger.info("stopped early: reached --max-calls=%d", args.max_calls)
                stopped_early = True
                break
            try:
                result = check_item_consistency(
                    item,
                    condition,
                    seed,
                    mock=args.mock,
                    client=client,
                    model=model,
                    constitution_text=constitution_text,
                    max_retries=max_retries,
                )
            except jc.JudgeError as exc:
                logger.error("skipping id=%s condition=%s: %s", item["id"], condition, exc)
                continue
            call_count += 2
            results.append(result)
            logger.info(
                "id=%s condition=%s actual=%s flipped=%s consistent=%s",
                item["id"],
                condition,
                result["actual_chosen_role"],
                result["flipped_chosen_role"],
                result["consistent"],
            )

    summary = summarize(results)

    out_path = (
        Path(args.out) if args.out else REPO_ROOT / cfg["paths"]["outputs_dir"] / "eval" / "judge_consistency_check.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    logger.info("wrote report to %s", out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
