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
import concurrent.futures
import json
import logging
import threading
from pathlib import Path

from tqdm import tqdm

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


_ROLE_TO_CANDIDATE_KEY = {
    "sycophantic": "answer_2_sycophantic_candidate",
    "principled": "answer_3_principled_candidate",
}


def build_dpo_record(item: dict, verdict: dict, slots: dict[str, str]) -> dict:
    """Turn one candidates.jsonl item + a judge verdict into a DPO pair.

    ``verdict["chosen"]`` is a *slot* label ("answer_2" or "answer_3"), not a
    role -- because the judge only ever saw neutral, randomized slot labels
    (see judge_common.assign_candidate_slots). ``slots`` (the same mapping
    passed to the judge for this item/condition) un-shuffles the verdict back
    to which underlying candidate (the sycophantic-style or the
    principled/reconsideration-style one) was actually chosen.
    """
    role_by_slot = {slot: role for role, slot in slots.items()}
    chosen_role = role_by_slot[verdict["chosen"]]
    rejected_role = "principled" if chosen_role == "sycophantic" else "sycophantic"
    return {
        "id": item["id"],
        "prompt": build_dpo_prompt_messages(item),
        "chosen": item[_ROLE_TO_CANDIDATE_KEY[chosen_role]],
        "rejected": item[_ROLE_TO_CANDIDATE_KEY[rejected_role]],
        "judge_reasoning": verdict.get("reasoning", ""),
    }


def get_verdict(
    item: dict,
    condition: str,
    *,
    mock: bool,
    slots: dict[str, str],
    client=None,
    model: str | None = None,
    constitution_text: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Dispatch to the mock or real judge for one item/condition pair.

    ``condition`` is ``"constitutional"`` (constitutional_dpo) or
    ``"generic"`` (generic_dpo). ``slots`` (from
    ``judge_common.assign_candidate_slots``) is required -- it controls which
    candidate the judge sees under the "answer_2" vs "answer_3" label, so
    position bias can't be confounded with which candidate is which (see
    judge_common's module docstring). Raises judge_common.JudgeError if a
    real call fails after all retries -- never raises for --mock (it's
    deterministic and offline).
    """
    if mock:
        if condition == "constitutional":
            return jc.mock_verdict_constitutional(item, slots)
        return jc.mock_verdict_generic(item, slots)

    if condition == "constitutional":
        system_prompt = jc.CONSTITUTIONAL_SYSTEM_PROMPT
        user_prompt = jc.build_constitutional_user_prompt(item, constitution_text or "", slots)
    else:
        system_prompt = jc.GENERIC_SYSTEM_PROMPT
        user_prompt = jc.build_generic_user_prompt(item, slots)

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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of judge calls to run in parallel (the Anthropic client is safe to share "
        "across threads). Each item can produce up to 2 independent calls (constitutional + "
        "generic), and both are eligible to run concurrently with each other and across items. "
        "Bound this by your Anthropic rate-limit tier (requests-per-minute), not by wishful "
        "thinking -- too high just trades real speedup for a pile of 429 retries. Also applies "
        "under --mock (harmless there, and useful for smoke-testing the concurrent code path "
        "without spending real API calls).",
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

    seed = cfg["seed"]

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

    # Each item can need up to 2 independent judge calls (constitutional +
    # generic). Flattening to a task-per-call list (rather than nesting the
    # condition loop inside the item loop, as a purely sequential version
    # would) is what lets both calls for one item, and calls across
    # different items, all become independently schedulable work for the
    # thread pool below.
    tasks = []
    for item in todo:
        if item["id"] not in done_constitutional:
            tasks.append((item, "constitutional"))
        if item["id"] not in done_generic:
            tasks.append((item, "generic"))
    if args.max_calls is not None and len(tasks) > args.max_calls:
        logger.info(
            "stopping at --max-calls=%d (%d judge calls would otherwise be made)",
            args.max_calls,
            len(tasks),
        )
        tasks = tasks[: args.max_calls]

    concurrency = args.concurrency

    def run_task(task: tuple[dict, str]) -> dict:
        item, condition = task
        slots = jc.assign_candidate_slots(item["id"], seed, condition)
        verdict = get_verdict(
            item,
            condition,
            mock=args.mock,
            slots=slots,
            client=client,
            model=model,
            constitution_text=constitution_text,
            max_retries=max_retries,
        )
        return build_dpo_record(item, verdict, slots)

    write_lock = threading.Lock()
    n_constitutional = 0
    n_generic = 0

    def handle_result(task: tuple[dict, str], record: dict, f_constitutional, f_generic) -> None:
        nonlocal n_constitutional, n_generic
        item, condition = task
        f = f_constitutional if condition == "constitutional" else f_generic
        with write_lock:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        if condition == "constitutional":
            done_constitutional.add(item["id"])
            n_constitutional += 1
        else:
            done_generic.add(item["id"])
            n_generic += 1
        logger.debug("judged id=%s condition=%s_dpo -> %s", item["id"], condition, record["chosen"][:60])

    with (
        out_constitutional_path.open("a", encoding="utf-8") as f_constitutional,
        out_generic_path.open("a", encoding="utf-8") as f_generic,
    ):
        if concurrency <= 1:
            for task in tqdm(tasks, desc="judging", unit="call"):
                item, condition = task
                try:
                    record = run_task(task)
                except jc.JudgeError as exc:
                    logger.error("skipping id=%s condition=%s_dpo: %s", item["id"], condition, exc)
                    continue
                handle_result(task, record, f_constitutional, f_generic)
        else:
            logger.info("judging with %d concurrent workers", concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(run_task, task): task for task in tasks}
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="judging",
                    unit="call",
                ):
                    task = futures[future]
                    item, condition = task
                    try:
                        record = future.result()
                    except jc.JudgeError as exc:
                        logger.error("skipping id=%s condition=%s_dpo: %s", item["id"], condition, exc)
                        continue
                    handle_result(task, record, f_constitutional, f_generic)

    logger.info(
        "done. constitutional_dpo: %d total pairs (%d new) at %s, generic_dpo: %d total pairs (%d new) at %s",
        len(done_constitutional),
        n_constitutional,
        out_constitutional_path,
        len(done_generic),
        n_generic,
        out_generic_path,
    )


if __name__ == "__main__":
    main()
