"""Sample the policy model's three candidate responses for each pushback item.

For every ``{id, source, original_prompt, pushback_template_id, pushback_text}``
record in ``data/generated/pushback_prompts.jsonl`` (produced by
``scripts/inject_pushback.py``), this generates three assistant replies from
the policy model (``configs/project.yaml``'s ``policy_model``, by default
``Qwen/Qwen2.5-3B-Instruct``):

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
    uv run scripts/generate_candidates.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --limit 20 --out data/generated/candidates_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("generate_candidates")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Extra turn appended after the pushback in the 3-turn conversation to elicit
# answer_3_principled_candidate. Deliberately short, and phrased in our own
# words rather than quoting configs/constitution.md directly -- it nudges the
# model toward principles 1 and 3 there (don't flip on pressure alone, do
# update when the user is actually right) without a rubric dump mid-chat.
RECONSIDER_PROMPT = (
    "Before you answer, pause and think it through carefully: only change "
    "your previous answer if the message above actually gave you a new "
    "fact, argument, or consideration you had missed -- not simply because "
    "the user expressed doubt or disagreement. If your original answer was "
    "correct and well-supported, hold your position clearly and briefly "
    "explain why. If the pushback did reveal a real error, say so plainly "
    "and correct it. Now give your honest, carefully reconsidered answer."
)

# Type alias: a generation backend is any callable taking a chat-format
# message list and returning the assistant's next reply as a string. Both
# the real policy-model backend and --dry-run's stub satisfy this, so
# build_candidate() never needs to know which one it's calling.
GenerateFn = Callable[[list[dict]], str]


def dry_run_generate(messages: list[dict]) -> str:
    """Deterministic placeholder generator used by --dry-run.

    No model, no network: a trivial stub so the full 3-generation control
    flow, output schema, and resumability logic can be exercised without
    any heavy ML dependency installed. Deterministic in the message
    history so repeated runs on the same input produce identical output.
    """
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    return f"[stub reply, turn {len(messages)}] {last_user[:80]}"


def load_policy_model(model_name: str):
    """Load the policy model + tokenizer for real generation.

    Imports torch/transformers lazily so this module can be imported and
    its CLI parsed without those (heavy, GPU-run-only) packages installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def generate_reply(tokenizer, model, messages: list[dict], gen_cfg: dict) -> str:
    """Generate one assistant reply for a chat-format message list.

    Uses the tokenizer's chat template so this works for whatever policy
    model is configured, without hardcoding a prompt format.
    """
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=gen_cfg.get("max_new_tokens", 512),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_candidate(record: dict, generate_fn: GenerateFn) -> dict:
    """Run the 3-generation control flow for one pushback record.

    ``record`` is one line of pushback_prompts.jsonl:
    ``{id, source, original_prompt, pushback_template_id, pushback_text}``.
    """
    prompt = record["original_prompt"]
    pushback_text = record["pushback_text"]

    turn_1 = [{"role": "user", "content": prompt}]
    answer_1 = generate_fn(turn_1)

    turn_2 = turn_1 + [
        {"role": "assistant", "content": answer_1},
        {"role": "user", "content": pushback_text},
    ]
    answer_2 = generate_fn(turn_2)

    turn_3 = turn_2 + [{"role": "user", "content": RECONSIDER_PROMPT}]
    answer_3 = generate_fn(turn_3)

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

    if args.dry_run:
        generate_fn: GenerateFn = dry_run_generate
    else:
        model_name = args.model or cfg["policy_model"]
        gen_cfg = cfg.get("candidate_generation", {})
        tokenizer, model = load_policy_model(model_name)

        def generate_fn(messages: list[dict]) -> str:
            return generate_reply(tokenizer, model, messages, gen_cfg)

    with output_path.open("a", encoding="utf-8") as f:
        for record in todo:
            candidate = build_candidate(record, generate_fn)
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            f.flush()
            logger.info("generated candidate for id=%s", record["id"])

    logger.info("wrote %d new candidates to %s", len(todo), output_path)


if __name__ == "__main__":
    main()
