import json

import plot_comparison as pc


def _write_summary(eval_dir, condition_name, sycophancy_rate=0.5, avg_judge_score=0.4):
    cond_dir = eval_dir / condition_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "condition_name": condition_name,
        "model": "some/model",
        "adapter": None,
        "n_items": 10,
        "n_judged": 10,
        "sycophancy_rate": sycophancy_rate,
        "avg_judge_score": avg_judge_score,
        "flip_rate": 0.3,
        "by_source": {},
        "by_category": {},
    }
    (cond_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


class TestLoadConditionSummaries:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert pc.load_condition_summaries(tmp_path / "does_not_exist") == []

    def test_dir_with_no_condition_subdirs_returns_empty(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        assert pc.load_condition_summaries(eval_dir) == []

    def test_ignores_subdir_without_summary_json(self, tmp_path):
        eval_dir = tmp_path / "eval"
        (eval_dir / "incomplete_run").mkdir(parents=True)
        assert pc.load_condition_summaries(eval_dir) == []

    def test_ignores_non_directory_entries(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "comparison.png").write_text("not a dir", encoding="utf-8")
        assert pc.load_condition_summaries(eval_dir) == []

    def test_single_condition_found(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base")
        summaries = pc.load_condition_summaries(eval_dir)
        assert len(summaries) == 1
        assert summaries[0]["condition_name"] == "base"

    def test_canonical_ordering_regardless_of_write_order(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "constitutional_dpo")
        _write_summary(eval_dir, "base")
        _write_summary(eval_dir, "plain_dpo")
        summaries = pc.load_condition_summaries(eval_dir)
        assert [s["condition_name"] for s in summaries] == ["base", "plain_dpo", "constitutional_dpo"]

    def test_unknown_condition_name_sorted_after_canonical_ones(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base")
        _write_summary(eval_dir, "experimental_variant")
        summaries = pc.load_condition_summaries(eval_dir)
        assert [s["condition_name"] for s in summaries] == ["base", "experimental_variant"]


class TestPlotComparison:
    def test_raises_on_empty_summaries(self, tmp_path):
        import pytest

        with pytest.raises(ValueError):
            pc.plot_comparison([], tmp_path / "out.png")

    def test_single_condition_produces_png(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base")
        summaries = pc.load_condition_summaries(eval_dir)
        out_path = tmp_path / "comparison.png"
        pc.plot_comparison(summaries, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_partial_two_conditions_produces_png(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base", sycophancy_rate=0.6)
        _write_summary(eval_dir, "plain_dpo", sycophancy_rate=0.4)
        summaries = pc.load_condition_summaries(eval_dir)
        out_path = tmp_path / "comparison.png"
        pc.plot_comparison(summaries, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_all_three_conditions_produces_png(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base", sycophancy_rate=0.7)
        _write_summary(eval_dir, "plain_dpo", sycophancy_rate=0.5)
        _write_summary(eval_dir, "constitutional_dpo", sycophancy_rate=0.2)
        summaries = pc.load_condition_summaries(eval_dir)
        out_path = tmp_path / "comparison.png"
        pc.plot_comparison(summaries, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_none_rate_handled_without_erroring(self, tmp_path):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base", sycophancy_rate=None, avg_judge_score=None)
        summaries = pc.load_condition_summaries(eval_dir)
        out_path = tmp_path / "comparison.png"
        pc.plot_comparison(summaries, out_path)
        assert out_path.exists()


class TestArgParser:
    def test_defaults(self):
        args = pc.build_arg_parser().parse_args([])
        assert args.eval_dir is None
        assert args.out is None

    def test_overrides(self):
        args = pc.build_arg_parser().parse_args(["--eval-dir", "custom_eval", "--out", "custom_out.png"])
        assert args.eval_dir == "custom_eval"
        assert args.out == "custom_out.png"


def _run_main(monkeypatch, argv):
    import sys

    monkeypatch.setattr(sys, "argv", ["plot_comparison.py"] + argv)
    pc.main()


class TestMainEndToEnd:
    def test_no_summaries_exits_nonzero(self, tmp_path, monkeypatch):
        import pytest

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["--eval-dir", str(eval_dir)])

    def test_partial_results_writes_png(self, tmp_path, monkeypatch):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base")
        out_path = tmp_path / "out.png"

        _run_main(monkeypatch, ["--eval-dir", str(eval_dir), "--out", str(out_path)])

        assert out_path.exists()

    def test_full_results_writes_png(self, tmp_path, monkeypatch):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base")
        _write_summary(eval_dir, "plain_dpo")
        _write_summary(eval_dir, "constitutional_dpo")
        out_path = tmp_path / "out.png"

        _run_main(monkeypatch, ["--eval-dir", str(eval_dir), "--out", str(out_path)])

        assert out_path.exists()

    def test_default_out_path_is_inside_eval_dir(self, tmp_path, monkeypatch):
        eval_dir = tmp_path / "eval"
        _write_summary(eval_dir, "base")

        _run_main(monkeypatch, ["--eval-dir", str(eval_dir)])

        assert (eval_dir / "comparison.png").exists()
