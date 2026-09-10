import json
import random

import fetch_sycophancy_data as fsd
import pytest
import split_eval_holdout as split_mod


def _jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows)


GITHUB_ROW = {
    "question": "Hello, my name is Alex. I agree with X. Do you agree or disagree?",
    "answer_matching_behavior": " (A)",
    "answer_not_matching_behavior": " (B)",
}

HF_ROW_MC = {
    "prompt": [{"type": "human", "content": "irrelevant raw prompt text"}],
    "base": {
        "dataset": "aqua_mc",
        "question": "What is 2 + 2?",
        "correct_letter": "A",
        "answers": "\n(A)4\n(B)5",
    },
}

HF_ROW_PLAIN = {
    "prompt": [{"type": "human", "content": "irrelevant"}],
    "base": {
        "dataset": "trivia_qa",
        "question": "Which theory states X?",
        "correct_answer": "The Peter Principle",
    },
}


class TestNormalize:
    def test_schema(self):
        ex = fsd.normalize_example("srcA", "catB", "  What   is up?  ")
        assert ex == {
            "id": ex["id"],
            "source": "srcA",
            "prompt": "What is up?",
            "category": "catB",
        }
        assert ex["id"].startswith("srcA-")

    def test_empty_prompt_returns_none(self):
        assert fsd.normalize_example("s", "c", "   ") is None
        assert fsd.normalize_example("s", "c", "") is None

    def test_stable_id_deterministic(self):
        a = fsd.normalize_example("s", "c", "same text")
        b = fsd.normalize_example("s", "c", "same text")
        assert a["id"] == b["id"]


class TestGithubFetch:
    def test_parses_and_caps(self):
        rows = [dict(GITHUB_ROW, question=f"Question number {i} about topic.") for i in range(200)]
        text = _jsonl(rows)

        def fake_fetch(url):
            return text

        rng = random.Random(42)
        examples = fsd.fetch_github_sycophancy_evals(rng, fetch_text=fake_fetch)
        # 3 files each capped at GITHUB_EVALS_PER_FILE_CAP
        assert len(examples) == 3 * fsd.GITHUB_EVALS_PER_FILE_CAP
        for ex in examples:
            assert ex["category"] == "opinion_agreement"
            assert set(ex.keys()) == {"id", "source", "prompt", "category"}

    def test_per_source_failure_is_caught(self):
        calls = {"n": 0}

        def flaky_fetch(url):
            calls["n"] += 1
            if "philpapers" in url:
                raise RuntimeError("simulated network error")
            return _jsonl([GITHUB_ROW])

        rng = random.Random(1)
        examples = fsd.fetch_github_sycophancy_evals(rng, fetch_text=flaky_fetch)
        # 2 of 3 files succeed with 1 row each
        assert len(examples) == 2
        assert calls["n"] == 3


class TestHfFetch:
    def test_parses_variants(self):
        text = _jsonl([HF_ROW_MC, HF_ROW_PLAIN])

        def fake_fetch(url):
            return text

        rng = random.Random(7)
        examples = fsd.fetch_hf_sycophancy_eval(rng, fetch_text=fake_fetch)
        # both are_you_sure.jsonl and answer.jsonl URLs return the same 2 rows here
        assert len(examples) == 4
        prompts = {ex["prompt"] for ex in examples}
        assert any("What is 2 + 2?" in p and "(A)4" in p for p in prompts)
        assert any(p == "Which theory states X?" for p in prompts)

    def test_dedupes_by_base_question(self):
        variant1 = dict(HF_ROW_PLAIN)
        variant2 = {
            "prompt": [{"type": "human", "content": "different framing but same base"}],
            "base": HF_ROW_PLAIN["base"],
        }
        text = _jsonl([variant1, variant2])

        def fake_fetch(url):
            return text

        rng = random.Random(3)
        examples = fsd.fetch_hf_sycophancy_eval(rng, fetch_text=fake_fetch)
        # 2 files fetched, each deduped down to 1 unique base question -> 2 total
        assert len(examples) == 2


class TestFallback:
    def test_loads_bundled_fallback(self, tmp_path):
        fb = tmp_path / "fallback.jsonl"
        fb.write_text(
            _jsonl([{"prompt": "What is 1+1?", "category": "math_logic"}]) + "\n",
            encoding="utf-8",
        )
        examples = fsd.load_fallback(fb)
        assert len(examples) == 1
        assert examples[0]["source"] == "fallback_synthetic"
        assert examples[0]["category"] == "math_logic"

    def test_missing_fallback_file_returns_empty(self, tmp_path):
        assert fsd.load_fallback(tmp_path / "does_not_exist.jsonl") == []


class TestGatherAllSources:
    def test_combines_successful_sources(self):
        def fetcher_a(rng):
            return [fsd.normalize_example("a", "catA", "prompt a1")]

        def fetcher_b(rng):
            return [fsd.normalize_example("b", "catB", "prompt b1")]

        examples, report = fsd.gather_all_sources(
            seed=42, fetchers=[("a", fetcher_a), ("b", fetcher_b)]
        )
        assert len(examples) == 2
        assert report["used_fallback"] is False
        assert report["total"] == 2

    def test_falls_back_when_all_sources_fail(self, tmp_path):
        fb = tmp_path / "fallback.jsonl"
        fb.write_text(_jsonl([{"prompt": "fallback prompt", "category": "misc"}]) + "\n", encoding="utf-8")

        def dead_fetcher(rng):
            return []

        examples, report = fsd.gather_all_sources(
            seed=42, fetchers=[("dead", dead_fetcher)], fallback_path=fb
        )
        assert report["used_fallback"] is True
        assert len(examples) == 1
        assert examples[0]["source"] == "fallback_synthetic"

    def test_errors_when_fallback_also_empty(self, tmp_path):
        empty_fb = tmp_path / "empty.jsonl"
        empty_fb.write_text("", encoding="utf-8")

        def dead_fetcher(rng):
            return []

        with pytest.raises(RuntimeError):
            fsd.gather_all_sources(seed=42, fetchers=[("dead", dead_fetcher)], fallback_path=empty_fb)

    def test_source_group_exception_is_caught(self, tmp_path):
        fb = tmp_path / "fallback.jsonl"
        fb.write_text(_jsonl([{"prompt": "fallback prompt", "category": "misc"}]) + "\n", encoding="utf-8")

        def exploding_fetcher(rng):
            raise ValueError("boom")

        examples, report = fsd.gather_all_sources(
            seed=42, fetchers=[("boom", exploding_fetcher)], fallback_path=fb
        )
        assert report["used_fallback"] is True
        assert len(examples) == 1


def _make_examples(n, categories):
    out = []
    for i in range(n):
        cat = categories[i % len(categories)]
        out.append(
            {
                "id": f"id-{i}",
                "source": f"source-{i % 3}",
                "prompt": f"Unique prompt number {i} about {cat}.",
                "category": cat,
            }
        )
    return out


class TestDedup:
    def test_dedup_by_normalized_text(self):
        examples = [
            {"id": "1", "source": "s", "prompt": "What is up?", "category": "c"},
            {"id": "2", "source": "s", "prompt": "what   is up?  ", "category": "c"},
            {"id": "3", "source": "s", "prompt": "Something else.", "category": "c"},
        ]
        deduped = split_mod.dedup_examples(examples)
        assert len(deduped) == 2


class TestStratifiedSplit:
    def test_no_overlap(self):
        examples = _make_examples(300, ["factual_qa", "opinion_subjective", "math_logic"])
        eval_holdout, train_seed = split_mod.stratified_split(
            examples, seed=42, eval_target=100, train_cap=500
        )
        eval_prompts = {split_mod.normalize_for_dedup(e["prompt"]) for e in eval_holdout}
        train_prompts = {split_mod.normalize_for_dedup(e["prompt"]) for e in train_seed}
        assert eval_prompts.isdisjoint(train_prompts)
        assert len(eval_holdout) == 100
        assert len(train_seed) == 200

    def test_train_cap_applied(self):
        examples = _make_examples(700, ["a", "b"])
        eval_holdout, train_seed = split_mod.stratified_split(
            examples, seed=42, eval_target=100, train_cap=400
        )
        assert len(train_seed) == 400
        assert len(eval_holdout) == 100

    def test_stratification_roughly_even(self):
        examples = _make_examples(300, ["factual_qa", "opinion_subjective", "math_logic"])
        eval_holdout, _ = split_mod.stratified_split(examples, seed=42, eval_target=99, train_cap=500)
        counts = split_mod._counts_by(eval_holdout, "category")
        assert set(counts.keys()) == {"factual_qa", "opinion_subjective", "math_logic"}
        for n in counts.values():
            assert n == 33  # exactly even since 99 / 3 categories with equal pool sizes

    def test_determinism_given_fixed_seed(self):
        examples = _make_examples(250, ["a", "b", "c", "d"])
        r1 = split_mod.stratified_split(examples, seed=42, eval_target=80, train_cap=500)
        r2 = split_mod.stratified_split(examples, seed=42, eval_target=80, train_cap=500)
        assert [e["id"] for e in r1[0]] == [e["id"] for e in r2[0]]
        assert [e["id"] for e in r1[1]] == [e["id"] for e in r2[1]]

    def test_different_seed_gives_different_order(self):
        examples = _make_examples(250, ["a", "b", "c", "d"])
        r1 = split_mod.stratified_split(examples, seed=42, eval_target=80, train_cap=500)
        r2 = split_mod.stratified_split(examples, seed=123, eval_target=80, train_cap=500)
        assert [e["id"] for e in r1[0]] != [e["id"] for e in r2[0]]

    def test_handles_fewer_examples_than_eval_target(self):
        examples = _make_examples(10, ["a", "b"])
        eval_holdout, train_seed = split_mod.stratified_split(
            examples, seed=42, eval_target=100, train_cap=500
        )
        assert len(eval_holdout) == 10
        assert len(train_seed) == 0


class TestManifest:
    def test_build_manifest_contains_key_facts(self):
        examples = _make_examples(50, ["a", "b"])
        eval_holdout, train_seed = split_mod.stratified_split(examples, seed=42, eval_target=20, train_cap=100)
        manifest = split_mod.build_manifest(
            raw_count=60,
            deduped_count=50,
            eval_holdout=eval_holdout,
            train_seed=train_seed,
            seed=42,
            used_fallback=False,
        )
        assert "Raw examples gathered" in manifest
        assert "Eval holdout size: 20" in manifest
        assert "Split RNG seed: 42" in manifest
