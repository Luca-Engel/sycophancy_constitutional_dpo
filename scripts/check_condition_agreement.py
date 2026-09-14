"""Diagnostic: how often do generic_dpo and constitutional_dpo already pick
the *same* underlying candidate for the same item?

Both conditions currently choose between the same two candidates
(``answer_2_sycophantic_candidate`` vs ``answer_3_principled_candidate``,
see ``scripts/generate_candidates.py``) for a given item -- only the judge
rubric differs (``judge_common.py``'s constitutional vs generic prompts).
``answer_3`` is elicited with an extra instruction that tells the model to
hold its position when right and correct itself when wrong -- traits a
*generic* quality judge is also likely to reward, independent of the
constitution. If that's happening in practice, generic_dpo (meant to be a
constitution-free control) ends up training on largely the same preference
signal as constitutional_dpo, which would understate how much the
constitution itself contributes on the held-out eval.

This script doesn't fix that (see README.md's "Future work" discussion of
regenerating the candidate pool); it just measures how big the effect is on
whatever preference-pairs data you already have. A high agreement rate is
evidence the confound matters in practice; a low one means it's mostly
theoretical for this dataset.

Since both output files carry the actual chosen response *text* (not just
a role label), and a given item's answer_2/answer_3 pair is fixed,
"generic_dpo picked the same underlying candidate as constitutional_dpo for
item X" is just "the two files' `chosen` strings are identical for id X" --
no need to re-derive roles or re-read candidates.jsonl.

Usage (after running judge_rank.py, real or --mock, for both conditions):
    uv run scripts/check_condition_agreement.py
    uv run scripts/check_condition_agreement.py \\
        --generic data/preference_pairs/generic_dpo.jsonl \\
        --constitutional data/preference_pairs/constitutional_dpo.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("check_condition_agreement")

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_agreement(generic_rows: list[dict], constitutional_rows: list[dict]) -> dict:
    """Compare chosen-candidate text between the two conditions for every id
    present in both.

    Returns ``{"n_common": int, "n_agree": int, "agreement_rate": float | None,
    "agreeing_ids": [str, ...], "disagreeing_ids": [str, ...]}``.
    ``agreement_rate`` is ``None`` if there are no ids in common (e.g. one
    file is empty) rather than raising a division error.
    """
    generic_by_id = {r["id"]: r for r in generic_rows}
    constitutional_by_id = {r["id"]: r for r in constitutional_rows}
    common_ids = sorted(set(generic_by_id) & set(constitutional_by_id))

    agreeing_ids = [i for i in common_ids if generic_by_id[i]["chosen"] == constitutional_by_id[i]["chosen"]]
    disagreeing_ids = [i for i in common_ids if i not in agreeing_ids]

    n = len(common_ids)
    return {
        "n_common": n,
        "n_agree": len(agreeing_ids),
        "agreement_rate": (len(agreeing_ids) / n) if n else None,
        "agreeing_ids": agreeing_ids,
        "disagreeing_ids": disagreeing_ids,
    }


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generic",
        default=None,
        help="Path to generic_dpo.jsonl. Defaults to <preference_pairs_dir>/generic_dpo.jsonl "
        "from configs/project.yaml.",
    )
    parser.add_argument(
        "--constitutional",
        default=None,
        help="Path to constitutional_dpo.jsonl. Defaults to "
        "<preference_pairs_dir>/constitutional_dpo.jsonl from configs/project.yaml.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the full report JSON. Defaults to "
        "<outputs_dir>/eval/condition_agreement_check.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_config()

    generic_path = (
        Path(args.generic) if args.generic else REPO_ROOT / cfg["paths"]["preference_pairs_dir"] / "generic_dpo.jsonl"
    )
    constitutional_path = (
        Path(args.constitutional)
        if args.constitutional
        else REPO_ROOT / cfg["paths"]["preference_pairs_dir"] / "constitutional_dpo.jsonl"
    )
    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / cfg["paths"]["outputs_dir"] / "eval" / "condition_agreement_check.json"
    )

    generic_rows = load_jsonl(generic_path)
    constitutional_rows = load_jsonl(constitutional_path)

    report = compute_agreement(generic_rows, constitutional_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "%d ids in common, %d agree (%.1f%%) -- see %s for the full id list",
        report["n_common"],
        report["n_agree"],
        (report["agreement_rate"] or 0.0) * 100,
        out_path,
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("agreeing_ids", "disagreeing_ids")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
