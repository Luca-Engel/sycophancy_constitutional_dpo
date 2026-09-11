"""End-to-end smoke test for the full data-generation + eval pipeline.

Runs the chain ``inject_pushback.py -> generate_candidates.py --dry-run ->
judge_rank.py --mock (conditions B and C) -> run_eval.py --mock`` on a tiny
10-item fixture (``tests/fixtures/smoke/``), and asserts every stage produces
its expected output file with the expected schema. Everything routes through
the dry-run/mock code paths already built into those scripts, so this needs
no GPU, no real API key, and no network access -- it runs in seconds and is
meant to be the thing CI runs to catch a schema mismatch between one script's
output and the next script's input before it reaches a real (paid) run.

Does *not* touch scripts/train_dpo.py: that script's own ``--smoke-test`` flag
already exercises the model-load -> LoRA -> DPOTrainer -> save loop in
isolation, and doing so here would pull in torch/transformers/peft/trl, which
this pipeline smoke test is specifically meant to avoid requiring.

Usage:
    uv run python scripts/smoke_test_pipeline.py

Exits non-zero (via the assertion's traceback) on any stage failure or schema
mismatch, so it works as a CI step.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "smoke"

LIMIT = 10


def log(msg: str) -> None:
    print(f"[smoke_test_pipeline] {msg}")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_step(args: list[str]) -> None:
    log("running: " + " ".join(args))
    result = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"smoke test failed: step exited with code {result.returncode}: {' '.join(args)}"
        )


def assert_records(records: list[dict], required_keys: set[str], stage: str) -> None:
    assert records, f"{stage}: expected at least one record, got none"
    for i, rec in enumerate(records):
        missing = required_keys - rec.keys()
        assert not missing, f"{stage}: record {i} missing keys {missing}: {rec}"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="smoke_pipeline_") as tmp:
        tmp_dir = Path(tmp)

        train_seed_fixture = FIXTURES_DIR / "train_seed_fixture.jsonl"
        eval_holdout_fixture = FIXTURES_DIR / "eval_holdout_fixture.jsonl"
        assert train_seed_fixture.exists(), f"missing fixture: {train_seed_fixture}"
        assert eval_holdout_fixture.exists(), f"missing fixture: {eval_holdout_fixture}"

        pushback_out = tmp_dir / "pushback_prompts.jsonl"
        candidates_out = tmp_dir / "candidates.jsonl"
        preference_pairs_dir = tmp_dir / "preference_pairs"
        eval_out_dir = tmp_dir / "eval" / "smoke"

        # --- 1. inject_pushback.py ---------------------------------------
        run_step(
            [
                str(SCRIPTS_DIR / "inject_pushback.py"),
                "--input",
                str(train_seed_fixture),
                "--output",
                str(pushback_out),
                "--seed",
                "42",
            ]
        )
        assert pushback_out.exists(), f"expected output missing: {pushback_out}"
        pushback_records = load_jsonl(pushback_out)
        assert_records(
            pushback_records,
            {"id", "source", "original_prompt", "pushback_template_id", "pushback_text"},
            "inject_pushback",
        )
        log(f"inject_pushback: {len(pushback_records)} records OK")

        # --- 2. generate_candidates.py --dry-run -------------------------
        run_step(
            [
                str(SCRIPTS_DIR / "generate_candidates.py"),
                "--input",
                str(pushback_out),
                "--out",
                str(candidates_out),
                "--dry-run",
                "--limit",
                str(LIMIT),
            ]
        )
        assert candidates_out.exists(), f"expected output missing: {candidates_out}"
        candidate_records = load_jsonl(candidates_out)
        assert_records(
            candidate_records,
            {
                "id",
                "prompt",
                "pushback_text",
                "answer_1",
                "answer_2_sycophantic_candidate",
                "answer_3_principled_candidate",
            },
            "generate_candidates",
        )
        log(f"generate_candidates: {len(candidate_records)} records OK")

        # --- 3. judge_rank.py --mock (both conditions B and C) -----------
        run_step(
            [
                str(SCRIPTS_DIR / "judge_rank.py"),
                "--in",
                str(candidates_out),
                "--out",
                str(preference_pairs_dir),
                "--mock",
                "--limit",
                str(LIMIT),
            ]
        )
        condition_c_path = preference_pairs_dir / "condition_c.jsonl"
        condition_b_path = preference_pairs_dir / "condition_b.jsonl"
        assert condition_c_path.exists(), f"expected output missing: {condition_c_path}"
        assert condition_b_path.exists(), f"expected output missing: {condition_b_path}"

        dpo_required_keys = {"id", "prompt", "chosen", "rejected", "judge_reasoning"}
        for label, path in (("condition_c", condition_c_path), ("condition_b", condition_b_path)):
            records = load_jsonl(path)
            assert_records(records, dpo_required_keys, f"judge_rank/{label}")
            for rec in records:
                assert isinstance(rec["prompt"], list), (
                    f"judge_rank/{label}: 'prompt' must be a chat-format message list, "
                    f"got {type(rec['prompt'])}"
                )
                for turn in rec["prompt"]:
                    assert {"role", "content"} <= turn.keys(), (
                        f"judge_rank/{label}: prompt turn missing role/content: {turn}"
                    )
            log(f"judge_rank/{label}: {len(records)} records OK")

        # --- 4. run_eval.py --mock --dry-run against the eval fixture ----
        run_step(
            [
                str(SCRIPTS_DIR / "run_eval.py"),
                "--eval-file",
                str(eval_holdout_fixture),
                "--out-dir",
                str(eval_out_dir),
                "--condition-name",
                "smoke",
                "--dry-run",
                "--mock",
                "--limit",
                str(LIMIT),
            ]
        )
        metrics_path = eval_out_dir / "metrics.csv"
        summary_path = eval_out_dir / "summary.json"
        assert metrics_path.exists(), f"expected output missing: {metrics_path}"
        assert summary_path.exists(), f"expected output missing: {summary_path}"

        import pandas as pd

        metrics_df = pd.read_csv(metrics_path)
        expected_metrics_cols = {
            "id",
            "source",
            "category",
            "prompt",
            "pushback_template_id",
            "pushback_text",
            "answer_1",
            "answer_2_post_pushback",
            "flip_detected",
            "flip_method",
            "flip_confidence",
            "judge_sycophantic",
            "judge_score",
            "judge_reasoning",
        }
        missing_cols = expected_metrics_cols - set(metrics_df.columns)
        assert not missing_cols, f"run_eval metrics.csv missing columns: {missing_cols}"
        eval_items = load_jsonl(eval_holdout_fixture)
        assert len(metrics_df) == len(eval_items), (
            f"run_eval: expected {len(eval_items)} rows, got {len(metrics_df)}"
        )

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        expected_summary_keys = {
            "condition_name",
            "model",
            "adapter",
            "n_items",
            "n_judged",
            "sycophancy_rate",
            "avg_judge_score",
            "flip_rate",
            "by_source",
            "by_category",
        }
        missing_summary_keys = expected_summary_keys - summary.keys()
        assert not missing_summary_keys, f"run_eval summary.json missing keys: {missing_summary_keys}"
        assert summary["condition_name"] == "smoke"
        assert summary["n_items"] == len(eval_items)
        log(f"run_eval: {len(metrics_df)} rows, summary OK")

    log("all stages passed")


if __name__ == "__main__":
    main()
