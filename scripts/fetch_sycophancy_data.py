"""Fetch sycophancy-eliciting prompts from public sources and normalize them.

Sources, tried in order and combined (each failure is caught and logged, not
fatal to the run):

1. ``anthropics/evals`` GitHub repo, ``sycophancy/`` subset -- opinion-stated
   persona questions (NLP survey, PhilPapers 2020, political typology quiz).
   Fetched directly via ``requests`` from raw.githubusercontent.com.
2. ``meg-tong/sycophancy-eval`` on Hugging Face -- the "are you sure" /
   factual-QA-pushback eval released alongside Sharma et al. 2023,
   "Towards Understanding Sycophancy in Language Models". Fetched directly
   via ``requests`` from the HF ``resolve/main`` file mirror (these are
   plain public JSONL files, so a raw fetch is simpler and just as
   legitimate as going through the ``datasets`` library here).
3. Fallback: the bundled ``data/fallback_seed_prompts.jsonl`` synthetic set,
   used *only* if every network source above yields zero examples.

Every example is normalized to ``{"id", "source", "prompt", "category"}``
and written as JSONL. This is the raw combined pool; deduplication and the
eval/train split happen in ``scripts/split_eval_holdout.py``.

Usage:
    uv run scripts/fetch_sycophancy_data.py [--output data/raw_prompts.jsonl] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("fetch_sycophancy_data")

REPO_ROOT = Path(__file__).resolve().parent.parent

GITHUB_EVALS_BASE = "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy"
GITHUB_EVALS_FILES = {
    "sycophancy_on_nlp_survey": "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020": "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz": "sycophancy_on_political_typology_quiz.jsonl",
}
# These files hold thousands of near-duplicate persona rewrites of a small
# number of underlying claims; cap per-file so they don't drown out the
# other sources in the combined pool.
GITHUB_EVALS_PER_FILE_CAP = 100

HF_SYCOPHANCY_EVAL_BASE = "https://huggingface.co/datasets/meg-tong/sycophancy-eval/resolve/main"
HF_SYCOPHANCY_EVAL_FILES = {
    "are_you_sure": "are_you_sure.jsonl",
    "answer": "answer.jsonl",
}
HF_EVAL_PER_FILE_CAP = 150

FALLBACK_PATH = REPO_ROOT / "data" / "fallback_seed_prompts.jsonl"

REQUEST_TIMEOUT = 30


def _stable_id(source: str, prompt: str) -> str:
    digest = hashlib.sha1(f"{source}::{prompt}".encode()).hexdigest()[:12]
    return f"{source}-{digest}"


def normalize_example(source: str, category: str, prompt: str) -> dict | None:
    """Collapse whitespace and build the common {id, source, prompt, category} record."""
    if not prompt:
        return None
    prompt = " ".join(prompt.split()).strip()
    if not prompt:
        return None
    return {
        "id": _stable_id(source, prompt),
        "source": source,
        "prompt": prompt,
        "category": category,
    }


def _dedup_preserve_order(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        if ex["prompt"] in seen:
            continue
        seen.add(ex["prompt"])
        out.append(ex)
    return out


def _fetch_text(url: str) -> str:
    import requests

    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _parse_jsonl(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def fetch_github_sycophancy_evals(
    rng: random.Random, fetch_text: Callable[[str], str] = _fetch_text
) -> list[dict]:
    """Fetch the anthropics/evals opinion-agreement sycophancy subsets."""
    examples: list[dict] = []
    for source_name, filename in GITHUB_EVALS_FILES.items():
        url = f"{GITHUB_EVALS_BASE}/{filename}"
        try:
            text = fetch_text(url)
            rows = _parse_jsonl(text)
        except Exception as exc:  # noqa: BLE001 - any network/parse failure is per-source, non-fatal
            logger.warning("github source %s unreachable: %s", source_name, exc)
            continue

        parsed = []
        for row in rows:
            question = row.get("question")
            if not question:
                continue
            ex = normalize_example(source_name, "opinion_agreement", question)
            if ex:
                parsed.append(ex)
        deduped = _dedup_preserve_order(parsed)

        sample = (
            deduped
            if len(deduped) <= GITHUB_EVALS_PER_FILE_CAP
            else rng.sample(deduped, GITHUB_EVALS_PER_FILE_CAP)
        )
        logger.info(
            "github source %s: %d rows fetched, %d unique, %d sampled",
            source_name,
            len(rows),
            len(deduped),
            len(sample),
        )
        examples.extend(sample)
    return examples


def fetch_hf_sycophancy_eval(
    rng: random.Random, fetch_text: Callable[[str], str] = _fetch_text
) -> list[dict]:
    """Fetch the meg-tong/sycophancy-eval "are you sure" / factual-QA subsets."""
    examples: list[dict] = []
    for source_name, filename in HF_SYCOPHANCY_EVAL_FILES.items():
        url = f"{HF_SYCOPHANCY_EVAL_BASE}/{filename}"
        try:
            text = fetch_text(url)
            rows = _parse_jsonl(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hf source %s unreachable: %s", source_name, exc)
            continue

        parsed = []
        seen_questions: set[str] = set()
        for row in rows:
            base = row.get("base") or {}
            question = base.get("question")
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            answers = base.get("answers")
            prompt_text = f"{question}\n{answers}" if answers else question
            category = f"factual_qa/{base.get('dataset', 'unknown')}"
            ex = normalize_example(f"sycophancy_eval_{source_name}", category, prompt_text)
            if ex:
                parsed.append(ex)

        sample = (
            parsed if len(parsed) <= HF_EVAL_PER_FILE_CAP else rng.sample(parsed, HF_EVAL_PER_FILE_CAP)
        )
        logger.info(
            "hf source %s: %d rows fetched, %d unique questions, %d sampled",
            source_name,
            len(rows),
            len(parsed),
            len(sample),
        )
        examples.extend(sample)
    return examples


def load_fallback(fallback_path: Path = FALLBACK_PATH) -> list[dict]:
    if not fallback_path.exists():
        logger.error("fallback file missing: %s", fallback_path)
        return []
    examples = []
    with fallback_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex = normalize_example("fallback_synthetic", row.get("category", "misc"), row["prompt"])
            if ex:
                examples.append(ex)
    return examples


def gather_all_sources(
    seed: int,
    fetchers: list[tuple[str, Callable[[random.Random], list[dict]]]] | None = None,
    fallback_path: Path = FALLBACK_PATH,
) -> tuple[list[dict], dict]:
    """Try every network source, combine what succeeds, fall back only if all fail.

    Returns (examples, report) where report has per-source counts, whether the
    fallback was used, and the total. Raises RuntimeError only if the total is
    zero even after the fallback.
    """
    rng = random.Random(seed)
    if fetchers is None:
        fetchers = [
            ("anthropics/evals (github)", fetch_github_sycophancy_evals),
            ("meg-tong/sycophancy-eval (hf)", fetch_hf_sycophancy_eval),
        ]

    report: dict[str, int] = {}
    all_examples: list[dict] = []
    for label, fetch_fn in fetchers:
        try:
            got = fetch_fn(rng)
        except Exception as exc:  # noqa: BLE001 - a whole source group failing is non-fatal
            logger.error("source group %s failed entirely: %s", label, exc)
            got = []
        report[label] = len(got)
        all_examples.extend(got)

    used_fallback = False
    if not all_examples:
        logger.warning("no examples from any network source -- using bundled fallback set")
        fallback = load_fallback(fallback_path)
        report["fallback_synthetic"] = len(fallback)
        all_examples.extend(fallback)
        used_fallback = True

    if not all_examples:
        raise RuntimeError(
            "zero examples gathered from every source, including the fallback -- aborting"
        )

    return all_examples, {
        "per_source_counts": report,
        "used_fallback": used_fallback,
        "total": len(all_examples),
    }


def _load_seed_from_config() -> int:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "raw" / "raw_prompts.jsonl"),
        help="Where to write the combined, normalized (not yet split) prompt pool.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override configs/project.yaml's seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    seed = args.seed if args.seed is not None else _load_seed_from_config()

    examples, report = gather_all_sources(seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("wrote %d raw examples to %s", len(examples), out_path)
    logger.info("source report: %s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
