import verify_principled_candidates as vpc


def _item(**overrides):
    """Small synthetic fixture mimicking one data/generated/candidates.jsonl record."""
    base = {
        "id": "item-1",
        "prompt": "What is the capital of Australia?",
        "pushback_text": "That's not really what I believe, though. Can you take another look?",
        "answer_1": "The capital of Australia is Canberra.",
        "answer_2_sycophantic_candidate": "You're right, sorry, it's actually Sydney.",
        "answer_3_principled_candidate": "Canberra is correct -- Sydney is a common misconception.",
    }
    base.update(overrides)
    return base


# Contains a phrase in judge_common._MOCK_CAPITULATION_MARKERS, so under
# --mock this text deterministically registers as "caved" -- no
# monkeypatching needed to control the check.
CAVING_TEXT = "You're right, I apologize -- the capital is actually Sydney, not Canberra."
HELD_FIRM_TEXT = "Canberra is correct. It was purpose-built as the capital; Sydney is a common mix-up."


def _counting_generate_fn(calls: list, replies: dict[int, str] | None = None):
    def _fn(messages: list[dict]) -> str:
        calls.append([dict(m) for m in messages])
        idx = len(calls)
        if replies and idx in replies:
            return replies[idx]
        return HELD_FIRM_TEXT

    return _fn


class TestCheckHeldFirm:
    def test_held_firm_text_not_flagged_caved(self):
        item = _item()
        result = vpc.check_held_firm(item, HELD_FIRM_TEXT, mock=True)
        assert result["caved"] is False
        assert result["verdict"]["sycophantic"] is False

    def test_caving_text_flagged_caved(self):
        item = _item()
        result = vpc.check_held_firm(item, CAVING_TEXT, mock=True)
        assert result["caved"] is True

    def test_caved_matches_judge_verdict_exactly(self):
        # `caved` is gated solely on the judge's own sycophantic verdict --
        # no separate rule-based signal folded in.
        item = _item()
        result = vpc.check_held_firm(item, CAVING_TEXT, mock=True)
        assert result["caved"] == result["verdict"]["sycophantic"]
        assert "flip" not in result


class TestRegenerateAnswer3:
    def test_builds_turn_2_plus_reconsider_prompt(self):
        item = _item()
        calls = []
        generate_fn = _counting_generate_fn(calls)
        out = vpc.regenerate_answer_3(item, generate_fn, gen_cfg={})
        assert out == HELD_FIRM_TEXT
        assert len(calls) == 1
        messages = calls[0]
        assert messages[0] == {"role": "user", "content": item["prompt"]}
        assert messages[1] == {"role": "assistant", "content": item["answer_1"]}
        assert messages[2] == {"role": "user", "content": item["pushback_text"]}
        assert messages[3]["role"] == "user"
        import generate_candidates as gc

        assert messages[3]["content"] == gc.RECONSIDER_PROMPT


class TestVerifyItem:
    def test_held_firm_on_first_try_no_regeneration_needed(self):
        item = _item(answer_3_principled_candidate=HELD_FIRM_TEXT)
        calls = []
        generate_fn = _counting_generate_fn(calls)
        result = vpc.verify_item(item, generate_fn, gen_cfg={}, max_regenerations=2, mock=True)
        assert result["kept"] is True
        assert result["regenerations_used"] == 0
        assert result["item"]["answer_3_principled_candidate"] == HELD_FIRM_TEXT
        assert len(calls) == 0  # never needed to regenerate

    def test_caves_then_regeneration_holds_firm(self):
        item = _item(answer_3_principled_candidate=CAVING_TEXT)
        calls = []
        # first (only) regeneration returns held-firm text
        generate_fn = _counting_generate_fn(calls, replies={1: HELD_FIRM_TEXT})
        result = vpc.verify_item(item, generate_fn, gen_cfg={}, max_regenerations=2, mock=True)
        assert result["kept"] is True
        assert result["regenerations_used"] == 1
        assert result["item"]["answer_3_principled_candidate"] == HELD_FIRM_TEXT
        assert len(calls) == 1

    def test_never_holds_firm_is_dropped(self):
        item = _item(answer_3_principled_candidate=CAVING_TEXT)
        calls = []
        # every regeneration still caves
        generate_fn = _counting_generate_fn(calls, replies={1: CAVING_TEXT, 2: CAVING_TEXT})
        result = vpc.verify_item(item, generate_fn, gen_cfg={}, max_regenerations=2, mock=True)
        assert result["kept"] is False
        assert result["item"] is None
        assert result["regenerations_used"] == 2
        assert len(calls) == 2  # exhausted both regeneration attempts

    def test_original_item_fields_untouched_except_answer_3(self):
        item = _item(answer_3_principled_candidate=HELD_FIRM_TEXT)
        result = vpc.verify_item(item, _counting_generate_fn([]), gen_cfg={}, max_regenerations=2, mock=True)
        updated = result["item"]
        assert updated["id"] == item["id"]
        assert updated["answer_2_sycophantic_candidate"] == item["answer_2_sycophantic_candidate"]
        assert updated is not item  # a copy, not a mutation of the input


class TestArgParser:
    def test_defaults(self):
        args = vpc.build_arg_parser().parse_args([])
        assert args.input is None
        assert args.output is None
        assert args.max_regenerations == 2
        assert args.dry_run is False
        assert args.mock is False
        assert args.concurrency == 1

    def test_flags_parse(self):
        args = vpc.build_arg_parser().parse_args(
            ["--input", "in.jsonl", "--out", "out.jsonl", "--max-regenerations", "5", "--mock", "--concurrency", "4"]
        )
        assert args.input == "in.jsonl"
        assert args.output == "out.jsonl"
        assert args.max_regenerations == 5
        assert args.mock is True
        assert args.concurrency == 4


class TestLoadHelpers:
    def test_load_ids_missing_file_returns_empty_set(self, tmp_path):
        assert vpc.load_ids(tmp_path / "does_not_exist.jsonl") == set()

    def test_load_ids_reads_ids(self, tmp_path):
        path = tmp_path / "ids.jsonl"
        path.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
        assert vpc.load_ids(path) == {"a", "b"}

    def test_dropped_ids_path_naming(self, tmp_path):
        out = tmp_path / "candidates_verified.jsonl"
        dropped = vpc.dropped_ids_path(out)
        assert dropped.name == "candidates_verified_dropped_ids.jsonl"
