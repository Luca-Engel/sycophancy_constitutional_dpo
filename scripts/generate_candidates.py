"""Sample the policy model's three candidate responses for each pushback item.

For every ``{id, source, original_prompt, pushback_template_id, pushback_text}``
record in ``data/generated/pushback_prompts.jsonl`` (produced by
``scripts/inject_pushback.py``), this generates three assistant replies from
the policy model (``configs/project.yaml``'s ``policy_model``, by default
``Qwen/Qwen3-4B-Instruct-2507``):

1. ``answer_1`` -- the model's reply to ``original_prompt`` alone.
2. ``answer_2_sycophantic_candidate`` -- the model's reply after being shown
   ``answer_1`` and then the pushback message, i.e. a completion of the
   3-turn conversation ``[user: original_prompt, assistant: answer_1, user:
   pushback_text]``. This is the candidate most likely to cave to pressure.
3. ``answer_3_principled_candidate`` -- a *second, independent* completion of
   that same 3-turn conversation, but with one extra short user turn
   (``RECONSIDER_PROMPT`` below) that asks the model to pause and only
   change its answer if the pushback actually supplied a new fact or
   argument -- informed by (not quoting) ``configs/constitution.md``'s
   "don't flip on pressure alone" / "actually update when the user is
   right" principles. answer_2 and answer_3 are alternate continuations of
   the *same* context, not chained off each other, so a later judge step can
   compare them as a genuine preference pair.

   Because only answer_3 gets this extra instruction, the model can (and
   often will) narrate that it was asked to pause and reconsider ("after
   thinking this over...", "let me reconsider..."). That's a content-level
   tell that reveals which candidate is which independent of
   ``judge_common.assign_candidate_slots``'s slot randomization -- a judge
   could end up preferring answer_3 for sounding more deliberate rather than
   for the substance of its position, which would contaminate even the
   generic_dpo (no-constitution) control. Two mitigations: ``RECONSIDER_PROMPT``
   itself now explicitly asks the model not to narrate the reconsideration,
   and ``strip_meta_commentary`` strips a leading self-referential clause as
   a safety net if the model does it anyway, before the text is ever written
   to ``candidates.jsonl`` -- i.e. before it can reach a judge or training
   data.

Output: ``data/generated/candidates.jsonl``, one JSON object per line:
``{"id", "prompt", "pushback_text", "answer_1",
"answer_2_sycophantic_candidate", "answer_3_principled_candidate"}``.

Writing is append-and-flush per item and resumable: on restart, ids already
present in the output file are skipped, so a run interrupted partway through
(e.g. a rented GPU pod dying) can just be re-invoked with the same
``--out`` and continue where it left off without re-doing or duplicating work.

Real run (Day 2, on a rented GPU box, after ``uv sync --extra train``):
    uv run scripts/generate_candidates.py

Local dry run (no GPU, no heavy deps, no network -- exercises the full
control flow with a deterministic stub generator):
    uv run scripts/generate_candidates.py --dry-run --limit 5

CLI overrides:
    uv run scripts/generate_candidates.py --model Qwen/Qwen3-1.7B \\
        --limit 20 --out data/generated/candidates_smoke.jsonl

Small exploratory subset via a hosted endpoint instead of a local/rented GPU
(no torch/transformers install needed -- see --endpoint-url's help). Add
--concurrency > 1 to run multiple records at once -- with a dedicated
Inference Endpoint that only matters if its autoscaling max replicas is also
raised above 1 (otherwise there's still only one replica to queue on):
    uv run scripts/generate_candidates.py --limit 25 --concurrency 4 \\
        --endpoint-url https://router.huggingface.co \\
        --endpoint-model Qwen/Qwen3-4B-Instruct-2507 \\
        --out data/generated/candidates_diagnostic.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from policy_model_common import (
    dry_run_generate,
    generate_reply,
    load_policy_model,
    looks_truncated,
)

logger = logging.getLogger("generate_candidates")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Extra turn appended after the pushback in the 3-turn conversation to elicit
# answer_3_principled_candidate. Deliberately short, and phrased in our own
# words rather than quoting configs/constitution.md directly -- it nudges the
# model toward principles 1 and 3 there (don't flip on pressure alone, do
# update when the user is actually right) without a rubric dump mid-chat.
#
# The final sentence explicitly tells the model not to narrate that it
# paused/reconsidered -- see the module docstring above for why: without
# this, answer_3 tends to contain self-referential tells ("after thinking
# this over...") that a judge could latch onto instead of judging the
# substance, which would leak into generic_dpo as well as constitutional_dpo.
RECONSIDER_PROMPT = (
    "Before you answer, pause and think it through carefully: only change "
    "your previous answer if the message above actually gave you a new "
    "fact, argument, or consideration you had missed -- not simply because "
    "the user expressed doubt or disagreement. If your original answer was "
    "correct and well-supported, hold your position clearly and briefly "
    "explain why. If the pushback did reveal a real error, say so plainly "
    "and correct it. Now give your answer directly, exactly as you would if "
    "this were your first reply to the question -- do not mention that you "
    "paused, reconsidered, or were asked to think it over."
)

# Leading self-referential clauses the model sometimes produces despite
# RECONSIDER_PROMPT's instruction not to narrate the reconsideration (e.g.
# "After thinking this over, ...", "Let me reconsider -- ..."). Each pattern
# is anchored at the very start of the text so a legitimate, substantive use
# of a similar phrase later in the answer is never touched.
_META_COMMENTARY_LEAD_INS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        (
            r"^after\s+(?:careful(?:ly)?\s+|further\s+|much\s+|some\s+)?"
            r"(?:reconsidering|reconsideration|thought|thinking\s+(?:it|this)\s+(?:over|through))[,:]?\s*"
        ),
        r"^(?:upon|on)\s+(?:further\s+)?reflection[,:]?\s*",
        r"^having\s+(?:reconsidered|thought\s+(?:it|this)\s+(?:over|through))[,:]?\s*",
        r"^let\s+me\s+reconsider(?:\s+(?:this|that|my\s+answer))?[,:.]?\s*(?:[\-–—]+\s*)?",
        r"^let\s+me\s+think\s+(?:this|it)\s+(?:through|over)[,:.]?\s*(?:[\-–—]+\s*)?",
        r"^(?:pausing|taking\s+a\s+moment)\s+to\s+reconsider[,:]?\s*",
        r"^okay,?\s+(?:let\s+me|i'll)\s+(?:pause\s+and\s+)?reconsider[,:.]?\s*",
    ]
]


def strip_meta_commentary(text: str) -> str:
    """Strip a leading self-referential "I paused/reconsidered" clause from
    a generated reply, if present -- the safety-net mitigation described in
    the module docstring. Only ever matches at the very start of the
    (whitespace-stripped) text, so it can't accidentally remove a genuine,
    substantive use of similar wording elsewhere in the answer. Returns the
    text unchanged if no lead-in matches; otherwise returns the remainder
    with its first letter re-capitalized so it still reads as a normal
    reply."""
    stripped = text.lstrip()
    for pattern in _META_COMMENTARY_LEAD_INS:
        match = pattern.match(stripped)
        if match:
            remainder = stripped[match.end() :].lstrip()
            if remainder:
                remainder = remainder[0].upper() + remainder[1:]
            return remainder
    return text

# Type alias: a generation backend is any callable taking a chat-format
# message list and returning the assistant's next reply as a string. Both
# the real policy-model backend and --dry-run's stub satisfy this, so
# build_candidate() never needs to know which one it's calling.
GenerateFn = Callable[[list[dict]], str]


def _warn_if_truncated(item_id: str, field_name: str, text: str) -> None:
    """Log a warning if ``text`` looks cut off before a natural stopping
    point (see policy_model_common.looks_truncated) -- a real diagnostic
    run found candidate_generation.max_new_tokens too low was silently
    truncating a majority of one candidate type mid-sentence, corrupting
    that data without any visible signal. Never alters or drops ``text``;
    this only makes the problem visible in the run's logs so a raised
    max_new_tokens (configs/project.yaml) can be verified, or a recurrence
    caught early, instead of requiring a manual inspection to notice."""
    if looks_truncated(text):
        logger.warning(
            "id=%s: %s looks truncated (%d chars, no sentence-ending punctuation) -- "
            "consider raising candidate_generation.max_new_tokens in configs/project.yaml",
            item_id,
            field_name,
            len(text),
        )


def build_candidate(record: dict, generate_fn: GenerateFn) -> dict:
    """Run the 3-generation control flow for one pushback record.

    ``record`` is one line of pushback_prompts.jsonl:
    ``{id, source, original_prompt, pushback_template_id, pushback_text}``.
    """
    prompt = record["original_prompt"]
    pushback_text = record["pushback_text"]

    turn_1 = [{"role": "user", "content": prompt}]
    answer_1 = generate_fn(turn_1)
    _warn_if_truncated(record["id"], "answer_1", answer_1)

    turn_2 = turn_1 + [
        {"role": "assistant", "content": answer_1},
        {"role": "user", "content": pushback_text},
    ]
    answer_2 = generate_fn(turn_2)
    _warn_if_truncated(record["id"], "answer_2_sycophantic_candidate", answer_2)

    turn_3 = turn_2 + [{"role": "user", "content": RECONSIDER_PROMPT}]
    answer_3 = strip_meta_commentary(generate_fn(turn_3))
    _warn_if_truncated(record["id"], "answer_3_principled_candidate", answer_3)

    return {
        "id": record["id"],
        "prompt": prompt,
        "pushback_text": pushback_text,
        "answer_1": answer_1,
        "answer_2_sycophantic_candidate": answer_2,
        "answer_3_principled_candidate": answer_3,
    }


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


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "generated" / "pushback_prompts.jsonl"),
        help="Path to pushback_prompts.jsonl.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        default=None,
        help="Output path. Defaults to <generated_dir>/candidates.jsonl from configs/project.yaml.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override configs/project.yaml's policy_model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N items (after resumability filtering is applied to the full input).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic stub generator instead of loading the real policy model.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Generate via an OpenAI-chat-compatible HTTP endpoint instead of loading the model "
        "locally (no torch/transformers needed) -- either Hugging Face's serverless router "
        "(https://router.huggingface.co, pass --endpoint-model too) or a dedicated HF Inference "
        "Endpoint URL you deployed yourself. Requires HF_TOKEN (see .env.example). Ignored if "
        "--dry-run is set.",
    )
    parser.add_argument(
        "--endpoint-model",
        default=None,
        help="Model id to send in the request body (e.g. Qwen/Qwen3-4B-Instruct-2507). Required for "
        "Hugging Face's serverless router (it serves many models); omit for a dedicated Inference "
        "Endpoint, which is already bound to one model.",
    )
    parser.add_argument(
        "--endpoint-api",
        choices=("chat", "text-generation"),
        default="chat",
        help="API shape --endpoint-url exposes. 'chat' (default): OpenAI-style Messages API at "
        "<url>/v1/chat/completions -- what HF's serverless router exposes, and what a dedicated "
        "Inference Endpoint exposes only if deployed with Messages API support. "
        "'text-generation': the older/default TGI shape at <url>/ ({\"inputs\": ...} in, raw "
        "generated text out) -- what a dedicated Inference Endpoint exposes if it was NOT "
        "deployed with Messages API support (a common default-deploy outcome). A 404 (not 503) "
        "on 'chat' is the tell you need 'text-generation' instead.",
    )
    parser.add_argument(
        "--endpoint-max-retries",
        type=int,
        default=None,
        help="Retries for --endpoint-url on a 503 (endpoint still warming up) or transient network "
        "error, with exponential backoff up to 60s between attempts. A freshly-deployed or "
        "recently-idle dedicated endpoint can take several minutes to finish loading, so bump this "
        "well above the default (5, ~2 minutes total) for a cold first call -- e.g. 12 (~10 "
        "minutes). Ignored if --dry-run is set or --endpoint-url is not given.",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=float,
        default=None,
        help="Seconds to wait for one generation request to --endpoint-url before treating it as "
        "failed (default 300). Separate from --endpoint-max-retries: a request that times out here "
        "because generation itself is slow will NOT be fixed by retrying, since each retry starts a "
        "fresh generation and hits the same wall again -- raise this instead. Scale it up if you "
        "raise candidate_generation.max_new_tokens in configs/project.yaml. Ignored if --dry-run is "
        "set or --endpoint-url is not given.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of records to generate in parallel via --endpoint-url (each record's 3 "
        "generations still run sequentially, since answer_2/answer_3 depend on answer_1). Only "
        "meaningful with --endpoint-url: with the default of 1, there is never more than one "
        "request in flight, so a multi-replica Inference Endpoint's autoscaler has no concurrent "
        "load to scale up on and stays pinned at 1 replica. Ignored (forced to 1) for --dry-run and "
        "for local --model generation, which share a single model instance on one GPU.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_config()

    output_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / cfg["paths"]["generated_dir"] / "candidates.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    done_ids = load_existing_ids(output_path)
    todo = [r for r in records if r["id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]

    logger.info(
        "%d total items, %d already done, %d to generate", len(records), len(done_ids), len(todo)
    )

    gen_cfg = cfg.get("candidate_generation", {})

    if args.dry_run:
        generate_fn: GenerateFn = dry_run_generate
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
        model_name = args.model or cfg["policy_model"]
        tokenizer, model = load_policy_model(model_name)

        def generate_fn(messages: list[dict]) -> str:
            return generate_reply(tokenizer, model, messages, gen_cfg)

    concurrency = args.concurrency
    if args.dry_run or not args.endpoint_url:
        concurrency = 1

    total = len(todo)
    done = 0
    write_lock = threading.Lock()

    with output_path.open("a", encoding="utf-8") as f:
        if concurrency <= 1:
            for record in tqdm(todo, desc="generating candidates", unit="item"):
                candidate = build_candidate(record, generate_fn)
                f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                logger.debug("generated candidate for id=%s (%d/%d)", record["id"], done, total)
        else:
            logger.info("generating with %d concurrent workers", concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(build_candidate, record, generate_fn): record for record in todo
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="generating candidates",
                    unit="item",
                ):
                    record = futures[future]
                    candidate = future.result()
                    with write_lock:
                        f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                        f.flush()
                    done += 1
                    logger.debug("generated candidate for id=%s (%d/%d)", record["id"], done, total)

    logger.info("wrote %d new candidates to %s", done, output_path)


if __name__ == "__main__":
    main()
