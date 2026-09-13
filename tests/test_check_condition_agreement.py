import check_condition_agreement as cca


def _pair(item_id: str, chosen: str) -> dict:
    return {"id": item_id, "prompt": [], "chosen": chosen, "rejected": "other text", "judge_reasoning": ""}


class TestComputeAgreement:
    def test_full_agreement(self):
        generic = [_pair("a", "text-A"), _pair("b", "text-B")]
        constitutional = [_pair("a", "text-A"), _pair("b", "text-B")]

        report = cca.compute_agreement(generic, constitutional)

        assert report["n_common"] == 2
        assert report["n_agree"] == 2
        assert report["agreement_rate"] == 1.0
        assert report["disagreeing_ids"] == []

    def test_full_disagreement(self):
        generic = [_pair("a", "text-A")]
        constitutional = [_pair("a", "text-DIFFERENT")]

        report = cca.compute_agreement(generic, constitutional)

        assert report["n_common"] == 1
        assert report["n_agree"] == 0
        assert report["agreement_rate"] == 0.0
        assert report["disagreeing_ids"] == ["a"]

    def test_partial_agreement(self):
        generic = [_pair("a", "text-A"), _pair("b", "text-B"), _pair("c", "text-C")]
        constitutional = [_pair("a", "text-A"), _pair("b", "text-DIFFERENT"), _pair("c", "text-C")]

        report = cca.compute_agreement(generic, constitutional)

        assert report["n_common"] == 3
        assert report["n_agree"] == 2
        assert report["agreement_rate"] == 2 / 3
        assert report["disagreeing_ids"] == ["b"]
        assert report["agreeing_ids"] == ["a", "c"]

    def test_only_ids_present_in_both_files_are_compared(self):
        generic = [_pair("a", "text-A"), _pair("only-in-generic", "x")]
        constitutional = [_pair("a", "text-A"), _pair("only-in-constitutional", "y")]

        report = cca.compute_agreement(generic, constitutional)

        assert report["n_common"] == 1
        assert report["n_agree"] == 1

    def test_empty_input_gives_none_rate_not_a_crash(self):
        report = cca.compute_agreement([], [])
        assert report["n_common"] == 0
        assert report["agreement_rate"] is None


class TestLoadJsonlRoundtrip:
    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        path.write_text('{"id": "a", "chosen": "x"}\n\n{"id": "b", "chosen": "y"}\n', encoding="utf-8")
        records = cca.load_jsonl(path)
        assert [r["id"] for r in records] == ["a", "b"]
