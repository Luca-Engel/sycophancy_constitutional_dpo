"""Remove a small, human-verified set of catastrophically degenerate items
from the DPO preference-pair files before training.

Background: the Day-2 candidate-generation run (Qwen3-4B-Instruct, 469
pushback items) occasionally produced a specific policy-model failure mode
under the "are you sure?" / factual-recall pushback prompts -- the model
asserts a wrong fact, immediately negates it ("... **X** -- no."), restates
it, negates it again, and never resolves, burning the entire
``max_new_tokens`` budget on the loop. ``scripts/judge_rank.py``'s real
judge correctly recognizes when this happens (its ``judge_reasoning`` says
so explicitly) but is still forced to pick a "chosen" side, so a handful of
DPO pairs end up with a ``chosen`` field that is itself incoherent
looping text rather than a real answer.

This matters specifically for DPO: training directly increases the
likelihood of the ``chosen`` completion's exact tokens, so a pair whose
``chosen`` text is a repetition loop would teach the model to be *more*
likely to produce that kind of breakdown -- the opposite of the intended
effect. A pair where only ``rejected`` is degenerate is fine to keep (it's
useful "don't do this" signal, since only its likelihood gets decreased).

Why this is a hardcoded id list rather than an automated regex/heuristic:
several were tried (n-gram repetition counting, dash+negation-word
markers) and none had usable precision on this data -- both flagged
legitimate step-by-step elimination reasoning and careful double-checking
(which also says "-- no" or uses repeated checkmarks while making genuine
forward progress) as often as they flagged genuine loops. All items any
heuristic flagged were read in full by hand; the ones below are the only
confirmed genuine loops out of that review. At this scale (3 of 469 items,
found via review of a candidate list no larger than ~10), a documented
manual exclusion list is more precise and more auditable than trusting an
imprecise automated filter.

The exclusion is applied identically to both constitutional_dpo.jsonl and
generic_dpo.jsonl (a union of ids, even though not every id's *chosen* text
was degenerate in both conditions) so the two training sets stay aligned on
the same underlying item pool -- filtering each file independently by its
own chosen-degeneracy would let them drift to different sizes/compositions,
which would itself confound a constitutional-vs-generic comparison.

Usage:
    uv run scripts/filter_preference_pairs.py

Reads <preference_pairs_dir>/{constitutional_dpo,generic_dpo}.jsonl (from
configs/project.yaml) and writes the filtered copies plus a MANIFEST.md to
--out-dir (default: data/preference_pairs_clean/), never touching the raw
judged input.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("filter_preference_pairs")

REPO_ROOT = Path(__file__).resolve().parent.parent

# id -> one-line reason, confirmed by full manual reading (see module
# docstring). Add to this list only after reading the full chosen text
# yourself -- see the docstring for why automated detection isn't trusted
# here.
EXCLUDED_IDS: dict[str, str] = {
    "sycophancy_eval_answer-84175933d91b": (
        "chosen completion is a ~70-repeat infinite loop asserting and negating "
        "'Sunsilk' as a Swiss-mountain sunscreen brand, never resolves"
    ),
    "sycophancy_eval_answer-877e6412496a": (
        "chosen completion is a ~45-repeat loop asserting and negating 'Mia Farrow' "
        "as Jessica Rabbit's actress before eventually recovering"
    ),
    "sycophancy_eval_answer-d3468517dbc2": (
        "chosen completion is a ~90-repeat loop asserting and negating 'Jeffrey Combs' "
        "as the lead in *The Fly* (1986), never resolves, cut off by max_new_tokens"
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--in-dir",
        default=None,
        help="Directory containing constitutional_dpo.jsonl/generic_dpo.jsonl. "
        "Defaults to configs/project.yaml's preference_pairs_dir.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the filtered files + MANIFEST.md into. "
        "Defaults to <preference_pairs_dir>_clean, a sibling of --in-dir.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_config()
    in_dir = Path(args.in_dir) if args.in_dir else REPO_ROOT / cfg["paths"]["preference_pairs_dir"]
    out_dir = Path(args.out_dir) if args.out_dir else in_dir.parent / f"{in_dir.name}_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        in_dir_display = in_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        in_dir_display = in_dir.as_posix()

    filenames = ["constitutional_dpo.jsonl", "generic_dpo.jsonl"]
    manifest_lines = [
        "# Filtered DPO preference pairs: provenance",
        "",
        f"Filtered from `{in_dir_display}` by `scripts/filter_preference_pairs.py`.",
        "Raw judged data is untouched; this is a separate, derived copy.",
        "",
        "## Exclusions",
        "",
        f"{len(EXCLUDED_IDS)} item(s) excluded from **both** files (union, so the two "
        "conditions stay aligned on the same item pool) because their `chosen` "
        "completion -- in at least one condition -- is a confirmed, manually-verified "
        "catastrophic repetition loop rather than a real answer. See "
        "`scripts/filter_preference_pairs.py`'s module docstring for why this is a "
        "hardcoded, human-reviewed list rather than an automated filter.",
        "",
    ]
    for item_id, reason in EXCLUDED_IDS.items():
        manifest_lines.append(f"- `{item_id}`: {reason}")
    manifest_lines.append("")
    manifest_lines.append("## Counts")
    manifest_lines.append("")

    for filename in filenames:
        records = load_jsonl(in_dir / filename)
        kept = [r for r in records if r["id"] not in EXCLUDED_IDS]
        dropped = len(records) - len(kept)
        write_jsonl(kept, out_dir / filename)
        logger.info(
            "%s: %d total, %d dropped, %d kept -> %s",
            filename, len(records), dropped, len(kept), out_dir / filename,
        )
        manifest_lines.append(f"- `{filename}`: {len(records)} total -> {len(kept)} kept ({dropped} dropped)")

    (out_dir / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_dir / "MANIFEST.md")


if __name__ == "__main__":
    main()
