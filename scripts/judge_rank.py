"""AI-feedback judge: critique-and-rank candidate pairs into DPO preference data.

For every ``{id, prompt, pushback_text, answer_1, answer_2_sycophantic_candidate,
answer_3_principled_candidate}`` record in ``data/generated/candidates.jsonl``
(produced by ``scripts/generate_candidates.py``), this asks a judge model
(``configs/project.yaml``'s ``judge.model``, via the Anthropic SDK) to pick the
better of ``answer_2`` vs ``answer_3`` under two independent rubrics:

- **constitutional_dpo**: shown the full text of ``configs/constitution.md``
  and asked which response better honors those principles under user
  pushback -> ``data/preference_pairs/constitutional_dpo.jsonl``.
- **generic_dpo (the control)**: asked generically which response is more
  helpful, correct, and high quality -- no mention of the constitution,
  pushback, sycophancy, or consistency -> written to
  ``data/preference_pairs/generic_dpo.jsonl``.

The actual "call the judge model and parse a JSON verdict" logic lives in
``scripts/judge_common.py`` so a later eval-harness script can reuse it
without duplicating prompt-building or retry logic.

Each output record is DPO-format:
``{"id", "prompt": <3-turn chat-format context: original prompt, answer_1,
pushback>, "chosen": <winning answer text>, "rejected": <losing answer
text>, "judge_reasoning": <judge's brief reasoning>}``.

Writing is append-and-flush per item and resumable per condition file
independently: on restart, ids already present in a given output file are
skipped for that condition, so an interrupted run (or one stopped early by
--max-calls) can be re-invoked with the same --out and continue.

Mock dry run (no API key, no network, exercises the full control flow with a
deterministic fake judge):
    uv run scripts/judge_rank.py --mock --limit 5

Real run (requires ANTHROPIC_API_KEY, see .env.example):
    uv run scripts/judge_rank.py

Safety-net cap on real API calls in one invocation:
    uv run scripts/judge_rank.py --max-calls 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import judge_common as jc

logger = logging.getLogger("judge_rank")

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_ids(path: Path) -> set[str]:
    """Ids already written to ``path`` (empty set if it doesn't exist yet)."""
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def build_dpo_prompt_messages(item: dict) -> list[dict]:
    """The 3-turn chat-format context the policy model saw before producing
    answer_2/answer_3: original prompt, its first answer, then the pushback.
    Matches generate_candidates.py's ``turn_2`` exactly, so a training script
    applying the policy model's chat template reproduces the same context
    the candidates were actually generated from."""
    return [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["answer_1"]},
        {"role": "user", "content": item["pushback_text"]},
    ]


def build_dpo_record(item: dict, verdict: dict) -> dict:
    """Turn one candidates.jsonl item + a judge verdict into a DPO pair."""
    if verdict["chosen"] == "answer_2":
        chosen_key, rejected_key = (
            "answer_2_sycophantic_candidate",
            "answer_3_principled_candidate",
        )
    else:
        chosen_key, rejected_key = (
            "answer_3_principled_candidate",
            "answer_2_sycophantic_candidate",
        )
    return {
        "id": item["id"],
        "prompt": build_dpo_prompt_messages(item),
        "chosen": item[chosen_key],
        "rejected": item[rejected_key],
        "judge_reasoning": verdict.get("reasoning", ""),
    }


def get_verdict(
    item: dict,
    condition: str,
    *,
    mock: bool,
    client=None,
    model: str | None = None,
    constitution_text: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Dispatch to the mock or real judge for one item/condition pair.

    ``condition`` is ``"constitutional"`` (constitutional_dpo) or
    ``"generic"`` (generic_dpo). Raises judge_common.JudgeError if a real
    call fails after all retries -- never raises for --mock (it's
    deterministic and offline).
    """
    if mock:
        if condition == "constitutional":
            return jc.mock_verdict_constitutional(item)
        return jc.mock_verdict_generic(item)

    if condition == "constitutional":
        system_prompt = jc.CONSTITUTIONAL_SYSTEM_PROMPT
        user_prompt = jc.build_constitutional_user_prompt(item, constitution_text or "")
    else:
        system_prompt = jc.GENERIC_SYSTEM_PROMPT
        user_prompt = jc.build_generic_user_prompt(item)

    return jc.call_judge(client, model, system_prompt, user_prompt, max_retries=max_retries)


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        "--input",
        dest="input",
        default=None,
        help="Path to candidates.jsonl. Defaults to <generated_dir>/candidates.jsonl from configs/project.yaml.",
    )
    parser.add_argument(
        "--out",
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Directory to write generic_dpo.jsonl/constitutional_dpo.jsonl into. "
        "Defaults to configs/project.yaml's preference_pairs_dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N input items (after resumability filtering).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Hard cap on the number of judge calls (real or mocked) made in this invocation. "
        "Each item can consume up to 2 calls (one per condition). Safety net for API spend.",
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

    input_path = (
        Path(args.input) if args.input else REPO_ROOT / cfg["paths"]["generated_dir"] / "candidates.jsonl"
    )
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["paths"]["preference_pairs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_constitutional_path = out_dir / "constitutional_dpo.jsonl"
    out_generic_path = out_dir / "generic_dpo.jsonl"

    items = load_jsonl(input_path)
    done_constitutional = load_existing_ids(out_constitutional_path)
    done_generic = load_existing_ids(out_generic_path)

    todo = [it for it in items if it["id"] not in done_constitutional or it["id"] not in done_generic]
    if args.limit is not None:
        todo = todo[: args.limit]

    logger.info(
        "%d total items, %d to consider (already fully done: %d)",
        len(items),
        len(todo),
        len(items) - len(todo) if args.limit is None else -1,
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

    call_count = 0
    stopped_early = False

    def budget_ok() -> bool:
        return args.max_calls is None or call_count < args.max_calls

    with (
        out_constitutional_path.open("a", encoding="utf-8") as f_constitutional,
        out_generic_path.open("a", encoding="utf-8") as f_generic,
    ):
        for item in todo:
            if stopped_early:
                break

            if item["id"] not in done_constitutional:
                if not budget_ok():
                    stopped_early = True
                    break
                call_count += 1
                try:
                    verdict = get_verdict(
                        item,
                        "constitutional",
                        mock=args.mock,
                        client=client,
                        model=model,
                        constitution_text=constitution_text,
                        max_retries=max_retries,
                    )
                except jc.JudgeError as exc:
                    logger.error("skipping id=%s condition=constitutional_dpo: %s", item["id"], exc)
                else:
                    record = build_dpo_record(item, verdict)
                    f_constitutional.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_constitutional.flush()
                    done_constitutional.add(item["id"])
                    logger.info("judged id=%s condition=constitutional_dpo -> %s", item["id"], verdict["chosen"])

            if item["id"] not in done_generic:
                if not budget_ok():
                    stopped_early = True
                    break
                call_count += 1
                try:
                    verdict = get_verdict(
                        item,
                        "generic",
                        mock=args.mock,
                        client=client,
                        model=model,
                        max_retries=max_retries,
                    )
                except jc.JudgeError as exc:
                    logger.error("skipping id=%s condition=generic_dpo: %s", item["id"], exc)
                else:
                    record = build_dpo_record(item, verdict)
                    f_generic.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_generic.flush()
                    done_generic.add(item["id"])
                    logger.info("judged id=%s condition=generic_dpo -> %s", item["id"], verdict["chosen"])

    if stopped_early:
        logger.info("stopped early: reached --max-calls=%d (%d calls made)", args.max_calls, call_count)
    logger.info(
        "done. constitutional_dpo: %d total pairs at %s, generic_dpo: %d total pairs at %s",
        len(done_constitutional),
        out_constitutional_path,
        len(done_generic),
        out_generic_path,
    )


if __name__ == "__main__":
    main()
