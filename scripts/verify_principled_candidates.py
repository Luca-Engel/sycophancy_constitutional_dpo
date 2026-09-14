"""Verify that each item's answer_3_principled_candidate actually held firm
against unjustified pushback, and regenerate/drop it if it didn't.

Background: generate_candidates.py elicits answer_3_principled_candidate by
appending an extra instruction (RECONSIDER_PROMPT) telling the policy model
"only change your answer if the pushback gave you a real new fact --
otherwise hold your position." Nothing checks whether the model actually
complied. A Day-2 real run found it often doesn't: the model produces a
warm, deferential opening ("Thank you for the thoughtful clarification --
I appreciate your honesty...") and then caves anyway, especially on
opinion_agreement items. Since answer_3 is trusted as-is by judge_rank.py
and can end up as the DPO "chosen" example, a candidate that only performs
politeness without the substance teaches the model exactly the wrong
lesson -- see README.md's "Debugging incident" section for the full
analysis this script exists to fix.

This script re-checks answer_3 for every candidate using the same
judge-based cave verdict run_eval.py already uses (get_sycophancy_verdict,
never duplicated here, just reused -- the same sycophancy-eval rubric
run_eval.py uses to score real eval items). Its rubric already explicitly
names "changing its answer without a legitimate reason" as one form of
caving, with full semantic context a rule-based check doesn't have, so
that's the sole gate (see check_held_firm's docstring for why a rule-based
answer-flip check was tried and dropped, not just deprioritized). If the
judge says the candidate caved, it's regenerated (a fresh,
independently-sampled completion of the same pushback turn) and
re-checked, up to --max-regenerations times. An item
whose answer_3 never holds firm is dropped entirely -- not just for one
DPO condition, but from the output altogether, since answer_3 is shared
raw material for both conditions in judge_rank.py: dropping it upstream,
before judge_rank.py ever sees it, is what keeps constitutional_dpo and
generic_dpo trained on the exact same item pool (see README.md's
Debugging incident section for why asymmetric cleaning would confound the
condition comparison).

Usage:
    uv run scripts/verify_principled_candidates.py

Mock/dry-run (no GPU, no network, exercises full control flow):
    uv run scripts/verify_principled_candidates.py --dry-run --mock --limit 5

Real run, same backend flags as generate_candidates.py --endpoint-url,
plus judge_rank.py's judge setup (no --mock):
    uv run scripts/verify_principled_candidates.py \\
        --endpoint-url https://... --endpoint-api text-generation

Writes a filtered candidates jsonl (default:
<generated_dir>/candidates_verified.jsonl) with answer_3_principled_candidate
replaced by whichever version actually held firm, plus a MANIFEST.md next
to it documenting pass/regenerate/drop counts. Resumable: ids already
decided (kept or dropped) in a prior run are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import threading
from pathlib import Path
from typing import Any

from tqdm import tqdm

import generate_candidates as gc
import judge_common as jc
import judge_rank as jr
import run_eval as re_mod

logger = logging.getLogger("verify_principled_candidates")

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_ids(path: Path) -> set[str]:
    """Ids already present in a jsonl file (empty set if it doesn't exist)."""
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def dropped_ids_path(out_path: Path) -> Path:
    """Companion file recording ids dropped in a prior run, so a resumed run
    doesn't re-attempt (and re-pay judge/generation calls for) an item
    already known to never hold firm."""
    return out_path.with_name(out_path.stem + "_dropped_ids.jsonl")


def check_held_firm(
    item: dict,
    candidate_text: str,
    *,
    mock: bool,
    judge_client=None,
    judge_model: str | None = None,
    judge_max_retries: int = 3,
) -> dict:
    """Check whether ``candidate_text`` (an answer_3 candidate) actually
    held firm against ``item``'s pushback.

    Gating is on the judge's own ``sycophantic`` verdict alone. Deliberately
    does NOT also use run_eval.detect_answer_flip's rule-based MC-letter/
    capitulation-phrase check: the judge's rubric
    (judge_common.build_sycophancy_eval_user_prompt) already explicitly
    names "changing its answer... without a legitimate reason" as one of
    the things it's asked to catch, with full semantic context a regex
    doesn't have, and its free-text ``reasoning`` already surfaces whether
    the answer changed when that's relevant -- so the rule-based check adds
    no coverage the judge doesn't already have, only a flakier second vote
    (an earlier version of this function OR'd the two together; see git
    history if you want to compare).

    Returns ``{"caved": bool, "verdict": <get_sycophancy_verdict result>}``.
    Raises judge_common.JudgeError if the real judge call fails after retries.
    """
    prediction = {
        "prompt": item["prompt"],
        "answer_1": item["answer_1"],
        "pushback_text": item["pushback_text"],
        "answer_2_post_pushback": candidate_text,
    }
    verdict = re_mod.get_sycophancy_verdict(
        prediction,
        mock=mock,
        client=judge_client,
        model=judge_model,
        max_retries=judge_max_retries,
    )
    caved = bool(verdict["sycophantic"])
    return {"caved": caved, "verdict": verdict}


def regenerate_answer_3(item: dict, generate_fn, gen_cfg: dict) -> str:
    """Produce a fresh, independently-sampled answer_3 for ``item``, reusing
    the exact same turn-construction judge_rank.py already relies on
    (build_dpo_prompt_messages) plus generate_candidates.py's
    RECONSIDER_PROMPT/strip_meta_commentary -- not duplicated here."""
    turn_2 = jr.build_dpo_prompt_messages(item)
    turn_3 = turn_2 + [{"role": "user", "content": gc.RECONSIDER_PROMPT}]
    return gc.strip_meta_commentary(generate_fn(turn_3))


def verify_item(
    item: dict,
    generate_fn,
    gen_cfg: dict,
    *,
    max_regenerations: int,
    mock: bool,
    judge_client=None,
    judge_model: str | None = None,
    judge_max_retries: int = 3,
) -> dict[str, Any]:
    """Run the check -> regenerate -> re-check loop for one item's answer_3.

    Returns ``{"kept": bool, "regenerations_used": int, "item": <updated item or None>}``.
    ``item`` in the result has ``answer_3_principled_candidate`` replaced by
    whichever version actually held firm; ``None`` if it never did.
    """
    candidate = item["answer_3_principled_candidate"]
    for attempt in range(max_regenerations + 1):
        result = check_held_firm(
            item,
            candidate,
            mock=mock,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_max_retries=judge_max_retries,
        )
        if not result["caved"]:
            updated = dict(item)
            updated["answer_3_principled_candidate"] = candidate
            return {"kept": True, "regenerations_used": attempt, "item": updated}
        if attempt < max_regenerations:
            candidate = regenerate_answer_3(item, generate_fn, gen_cfg)
    return {"kept": False, "regenerations_used": max_regenerations, "item": None}


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default=None,
        help="Path to candidates.jsonl. Defaults to <generated_dir>/candidates.jsonl from configs/project.yaml.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        default=None,
        help="Output path. Defaults to <generated_dir>/candidates_verified.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N items (after resumability filtering).",
    )
    parser.add_argument(
        "--max-regenerations",
        type=int,
        default=2,
        help="Max times to regenerate answer_3 for one item before giving up and dropping it (default 2, "
        "so up to 3 total attempts).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic stub generator instead of a real policy-model backend, for regeneration.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Generate regenerations via an OpenAI-chat-compatible HTTP endpoint. Same as "
        "generate_candidates.py's --endpoint-url -- see its help for details.",
    )
    parser.add_argument(
        "--endpoint-model",
        default=None,
        help="Model id for --endpoint-url (HF serverless router only). See generate_candidates.py's help.",
    )
    parser.add_argument(
        "--endpoint-api",
        choices=("chat", "text-generation"),
        default="chat",
        help="API shape --endpoint-url exposes. See generate_candidates.py's help.",
    )
    parser.add_argument(
        "--endpoint-max-retries",
        type=int,
        default=None,
        help="Retries for --endpoint-url on a 503/network error. See generate_candidates.py's help.",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=float,
        default=None,
        help="Seconds to wait for one --endpoint-url generation request. See generate_candidates.py's help.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override configs/project.yaml's policy_model for local generation (ignored with --dry-run "
        "or --endpoint-url).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic fake judge instead of calling the real API. No network access.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of items to verify in parallel. Only meaningful with --endpoint-url (a local --model "
        "shares one GPU/model instance and is forced to concurrency=1, same as generate_candidates.py); "
        "the judge side is always safe to parallelize regardless.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_config()

    input_path = Path(args.input) if args.input else REPO_ROOT / cfg["paths"]["generated_dir"] / "candidates.jsonl"
    output_path = (
        Path(args.output) if args.output else REPO_ROOT / cfg["paths"]["generated_dir"] / "candidates_verified.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path = dropped_ids_path(output_path)

    records = load_jsonl(input_path)
    done_ids = load_ids(output_path) | load_ids(dropped_path)
    todo = [r for r in records if r["id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]

    logger.info(
        "%d total items, %d already decided (kept or dropped), %d to verify",
        len(records),
        len(done_ids),
        len(todo),
    )

    gen_cfg = cfg.get("candidate_generation", {})

    if args.dry_run:
        generate_fn: gc.GenerateFn = gc.dry_run_generate
    elif args.endpoint_url:
        from policy_model_common import generate_via_http_endpoint, get_hf_token

        token = get_hf_token()
        endpoint_kwargs = {"api": args.endpoint_api}
        if args.endpoint_max_retries is not None:
            endpoint_kwargs["max_retries"] = args.endpoint_max_retries
        if args.endpoint_timeout is not None:
            endpoint_kwargs["timeout"] = args.endpoint_timeout

        def generate_fn(messages: list[dict]) -> str:
            return generate_via_http_endpoint(
                messages, args.endpoint_url, token, gen_cfg, model_name=args.endpoint_model, **endpoint_kwargs
            )
    else:
        from policy_model_common import generate_reply, load_policy_model

        model_name = args.model or cfg["policy_model"]
        tokenizer, model = load_policy_model(model_name)

        def generate_fn(messages: list[dict]) -> str:
            return generate_reply(tokenizer, model, messages, gen_cfg)

    judge_client = None
    judge_model = None
    judge_cfg = cfg.get("judge", {})
    judge_max_retries = judge_cfg.get("max_retries", 3)
    if not args.mock:
        try:
            api_key = jc.get_api_key()
        except jc.MissingAPIKeyError as exc:
            logger.error(str(exc))
            raise SystemExit(1)
        judge_client = jc.build_anthropic_client(api_key)
        judge_model = judge_cfg["model"]

    concurrency = args.concurrency
    if args.dry_run or not args.endpoint_url:
        concurrency = 1

    def run_one(item: dict) -> dict:
        return verify_item(
            item,
            generate_fn,
            gen_cfg,
            max_regenerations=args.max_regenerations,
            mock=args.mock,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_max_retries=judge_max_retries,
        )

    write_lock = threading.Lock()
    kept = 0
    dropped = 0
    regen_counts: dict[int, int] = {}

    def handle_result(item_id: str, result: dict, out_f, dropped_f) -> None:
        nonlocal kept, dropped
        regen_counts[result["regenerations_used"]] = regen_counts.get(result["regenerations_used"], 0) + 1
        with write_lock:
            if result["kept"]:
                out_f.write(json.dumps(result["item"], ensure_ascii=False) + "\n")
                out_f.flush()
                kept += 1
            else:
                dropped_f.write(json.dumps({"id": item_id}, ensure_ascii=False) + "\n")
                dropped_f.flush()
                dropped += 1

    with output_path.open("a", encoding="utf-8") as out_f, dropped_path.open("a", encoding="utf-8") as dropped_f:
        if concurrency <= 1:
            for item in tqdm(todo, desc="verifying answer_3", unit="item"):
                result = run_one(item)
                handle_result(item["id"], result, out_f, dropped_f)
        else:
            logger.info("verifying with %d concurrent workers", concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(run_one, item): item for item in todo}
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="verifying answer_3", unit="item"
                ):
                    item = futures[future]
                    result = future.result()
                    handle_result(item["id"], result, out_f, dropped_f)

    total_decided = len(load_ids(output_path)) + len(load_ids(dropped_path))
    manifest_lines = [
        "# Verified principled-candidate provenance",
        "",
        f"Verified from `{input_path}` by `scripts/verify_principled_candidates.py`.",
        "Each item's `answer_3_principled_candidate` was checked against the item's own pushback "
        "(rule-based flip detection + a judge-based cave verdict, both reused from `run_eval.py`) "
        f"and regenerated up to {args.max_regenerations} time(s) if it caved. Items that never held "
        "firm are dropped entirely -- not written to the output -- so they never reach judge_rank.py "
        "for either DPO condition. See README.md's \"Debugging incident\" section for why.",
        "",
        "## Counts (this run)",
        "",
        f"- kept (held firm, possibly after regeneration): {kept}",
        f"- dropped (never held firm after {args.max_regenerations} regeneration(s)): {dropped}",
        f"- total decided across all runs so far: {total_decided} / {len(records)}",
        "",
        "## Regenerations needed (this run)",
        "",
    ]
    for n in sorted(regen_counts):
        label = "held firm on first try" if n == 0 else f"needed {n} regeneration(s)"
        manifest_lines.append(f"- {label}: {regen_counts[n]} item(s)")

    (output_path.parent / f"{output_path.stem}_MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    logger.info(
        "done. kept %d, dropped %d (this run) -> %s (%d total kept so far)",
        kept,
        dropped,
        output_path,
        len(load_ids(output_path)),
    )


if __name__ == "__main__":
    main()
