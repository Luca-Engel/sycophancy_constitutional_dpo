import json

import judge_common as jc
import judge_rank as jr


def _candidate_items():
    """Small synthetic fixture mimicking data/generated/candidates.jsonl."""
    return [
        {
            "id": "item-1",
            "prompt": "What is the capital of Australia?",
            "pushback_text": "Hmm, are you sure? That doesn't sound right to me.",
            "answer_1": "The capital of Australia is Canberra.",
            "answer_2_sycophantic_candidate": "You're right, sorry, it's actually Sydney.",
            "answer_3_principled_candidate": "It is Canberra -- a common misconception is Sydney, "
            "but Canberra was purpose-built as the capital.",
        },
        {
            "id": "item-2",
            "prompt": "Who wrote the play Hamlet?",
            "pushback_text": "Actually, I'm pretty sure that's wrong.",
            "answer_1": "William Shakespeare wrote Hamlet.",
            "answer_2_sycophantic_candidate": "You may be right, I'm not fully certain.",
            "answer_3_principled_candidate": "Hamlet was written by William Shakespeare, generally "
            "dated to around 1600-1601.",
        },
        {
            "id": "item-3",
            "prompt": "What is 9 * 7?",
            "pushback_text": "I don't think that's right, can you double check?",
            "answer_1": "9 * 7 = 63.",
            "answer_2_sycophantic_candidate": "Let me reconsider -- it might be 61.",
            "answer_3_principled_candidate": "9 * 7 = 63. I've double-checked and that's correct.",
        },
    ]


class FakeAnthropicClient:
    """Fake Anthropic SDK client: .messages.create(...) returns queued responses."""

    def __init__(self, texts_or_exceptions):
        self._queue = list(texts_or_exceptions)
        self.calls = []

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                item = self._outer._queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return _FakeResponse(item)

        self.messages = _Messages(self)


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeBlock:
    def __init__(self, text):
        self.text = text


CONSTITUTION_TEXT = "1. Don't flip on pressure alone.\n2. Actually update when the user is right."


class TestPromptConstruction:
    def test_constitutional_prompt_includes_full_constitution_text(self):
        item = _candidate_items()[0]
        prompt = jc.build_constitutional_user_prompt(item, CONSTITUTION_TEXT)
        assert CONSTITUTION_TEXT in prompt

    def test_constitutional_prompt_frames_pushback_and_constitution(self):
        item = _candidate_items()[0]
        prompt = jc.build_constitutional_user_prompt(item, CONSTITUTION_TEXT).lower()
        assert "constitution" in prompt
        assert "pushback" in prompt

    def test_constitutional_prompt_includes_both_candidates(self):
        item = _candidate_items()[0]
        prompt = jc.build_constitutional_user_prompt(item, CONSTITUTION_TEXT)
        assert item["answer_2_sycophantic_candidate"] in prompt
        assert item["answer_3_principled_candidate"] in prompt

    def test_plain_prompt_excludes_forbidden_framing_words(self):
        item = _candidate_items()[0]
        prompt = jc.build_plain_user_prompt(item).lower()
        for forbidden in ("constitution", "pushback", "sycophan", "consisten"):
            assert forbidden not in prompt

    def test_plain_prompt_is_generic_quality_framing(self):
        item = _candidate_items()[0]
        prompt = jc.build_plain_user_prompt(item).lower()
        assert "helpful" in prompt
        assert "quality" in prompt

    def test_plain_prompt_includes_both_candidates(self):
        item = _candidate_items()[0]
        prompt = jc.build_plain_user_prompt(item)
        assert item["answer_2_sycophantic_candidate"] in prompt
        assert item["answer_3_principled_candidate"] in prompt

    def test_both_prompts_request_json_shape(self):
        item = _candidate_items()[0]
        for prompt in (
            jc.build_constitutional_user_prompt(item, CONSTITUTION_TEXT),
            jc.build_plain_user_prompt(item),
        ):
            assert "answer_2" in prompt and "answer_3" in prompt
            assert "json" in prompt.lower()


class TestParseVerdict:
    def test_parse_clean_json(self):
        verdict = jc.parse_verdict('{"chosen": "answer_2", "reasoning": "shorter and clearer"}')
        assert verdict == {"chosen": "answer_2", "reasoning": "shorter and clearer"}

    def test_parse_json_in_markdown_fence(self):
        text = '```json\n{"chosen": "answer_3", "reasoning": "held its ground"}\n```'
        verdict = jc.parse_verdict(text)
        assert verdict["chosen"] == "answer_3"

    def test_parse_json_with_surrounding_prose(self):
        text = 'Sure, here is my verdict:\n{"chosen": "answer_2", "reasoning": "ok"}\nHope that helps!'
        verdict = jc.parse_verdict(text)
        assert verdict["chosen"] == "answer_2"

    def test_parse_missing_reasoning_defaults_to_empty_string(self):
        verdict = jc.parse_verdict('{"chosen": "answer_2"}')
        assert verdict["reasoning"] == ""

    def test_parse_invalid_chosen_value_raises(self):
        try:
            jc.parse_verdict('{"chosen": "answer_1", "reasoning": "x"}')
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_parse_non_json_raises(self):
        try:
            jc.parse_verdict("not json at all")
            assert False, "expected an exception"
        except ValueError:
            pass


class TestCallJudge:
    def test_success_on_first_try(self, monkeypatch):
        monkeypatch.setattr(jc.time, "sleep", lambda *_: None)
        client = FakeAnthropicClient(['{"chosen": "answer_2", "reasoning": "fine"}'])
        verdict = jc.call_judge(client, "fake-model", "sys", "user", max_retries=3)
        assert verdict == {"chosen": "answer_2", "reasoning": "fine"}
        assert len(client.calls) == 1

    def test_retries_on_malformed_json_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(jc.time, "sleep", lambda *_: None)
        client = FakeAnthropicClient(
            ["this is not json", '{"chosen": "answer_3", "reasoning": "recovered"}']
        )
        verdict = jc.call_judge(client, "fake-model", "sys", "user", max_retries=3)
        assert verdict["chosen"] == "answer_3"
        assert len(client.calls) == 2

    def test_retries_on_transient_api_error_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(jc.time, "sleep", lambda *_: None)
        client = FakeAnthropicClient(
            [RuntimeError("connection reset"), '{"chosen": "answer_2", "reasoning": "ok"}']
        )
        verdict = jc.call_judge(client, "fake-model", "sys", "user", max_retries=3)
        assert verdict["chosen"] == "answer_2"
        assert len(client.calls) == 2

    def test_raises_judge_error_after_exhausting_retries(self, monkeypatch):
        monkeypatch.setattr(jc.time, "sleep", lambda *_: None)
        client = FakeAnthropicClient(["nope", "still nope", "nope again"])
        try:
            jc.call_judge(client, "fake-model", "sys", "user", max_retries=3)
            assert False, "expected JudgeError"
        except jc.JudgeError:
            pass
        assert len(client.calls) == 3


class TestMockVerdicts:
    def test_constitutional_mock_prefers_principled_candidate(self):
        item = _candidate_items()[0]
        verdict = jc.mock_verdict_constitutional(item)
        assert verdict["chosen"] == "answer_3"

    def test_plain_mock_prefers_longer_answer_2(self):
        item = {
            "answer_2_sycophantic_candidate": "a" * 100,
            "answer_3_principled_candidate": "b" * 10,
        }
        assert jc.mock_verdict_plain(item)["chosen"] == "answer_2"

    def test_plain_mock_prefers_longer_answer_3(self):
        item = {
            "answer_2_sycophantic_candidate": "a" * 10,
            "answer_3_principled_candidate": "b" * 100,
        }
        assert jc.mock_verdict_plain(item)["chosen"] == "answer_3"


class TestGetApiKey:
    def test_missing_key_raises_actionable_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(jc, "_REPO_ROOT", tmp_path)
        try:
            jc.get_api_key()
            assert False, "expected MissingAPIKeyError"
        except jc.MissingAPIKeyError as exc:
            assert "--mock" in str(exc)

    def test_present_key_is_returned(self, monkeypatch, tmp_path):
        monkeypatch.setattr(jc, "_REPO_ROOT", tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-not-a-real-key")
        assert jc.get_api_key() == "sk-test-fake-not-a-real-key"


class TestBuildDpoRecord:
    def test_prompt_messages_shape(self):
        item = _candidate_items()[0]
        messages = jr.build_dpo_prompt_messages(item)
        assert messages == [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["answer_1"]},
            {"role": "user", "content": item["pushback_text"]},
        ]

    def test_record_schema_chosen_answer_2(self):
        item = _candidate_items()[0]
        record = jr.build_dpo_record(item, {"chosen": "answer_2", "reasoning": "r"})
        assert record["chosen"] == item["answer_2_sycophantic_candidate"]
        assert record["rejected"] == item["answer_3_principled_candidate"]
        assert record["id"] == item["id"]
        assert record["judge_reasoning"] == "r"
        assert set(record.keys()) == {"id", "prompt", "chosen", "rejected", "judge_reasoning"}

    def test_record_schema_chosen_answer_3(self):
        item = _candidate_items()[0]
        record = jr.build_dpo_record(item, {"chosen": "answer_3", "reasoning": "r"})
        assert record["chosen"] == item["answer_3_principled_candidate"]
        assert record["rejected"] == item["answer_2_sycophantic_candidate"]


class TestGetVerdictDispatch:
    def test_mock_constitutional(self):
        item = _candidate_items()[0]
        verdict = jr.get_verdict(item, "constitutional", mock=True)
        assert verdict == jc.mock_verdict_constitutional(item)

    def test_mock_plain(self):
        item = _candidate_items()[0]
        verdict = jr.get_verdict(item, "plain", mock=True)
        assert verdict == jc.mock_verdict_plain(item)

    def test_real_constitutional_calls_call_judge_with_constitutional_prompts(self, monkeypatch):
        captured = {}

        def fake_call_judge(client, model, system_prompt, user_prompt, max_retries=3):
            captured.update(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=max_retries,
            )
            return {"chosen": "answer_2", "reasoning": "x"}

        monkeypatch.setattr(jc, "call_judge", fake_call_judge)
        item = _candidate_items()[0]
        verdict = jr.get_verdict(
            item,
            "constitutional",
            mock=False,
            client="fake-client",
            model="fake-model",
            constitution_text=CONSTITUTION_TEXT,
            max_retries=5,
        )
        assert verdict["chosen"] == "answer_2"
        assert captured["system_prompt"] == jc.CONSTITUTIONAL_SYSTEM_PROMPT
        assert CONSTITUTION_TEXT in captured["user_prompt"]
        assert captured["max_retries"] == 5

    def test_real_plain_calls_call_judge_with_plain_prompts(self, monkeypatch):
        captured = {}

        def fake_call_judge(client, model, system_prompt, user_prompt, max_retries=3):
            captured.update(system_prompt=system_prompt, user_prompt=user_prompt)
            return {"chosen": "answer_3", "reasoning": "x"}

        monkeypatch.setattr(jc, "call_judge", fake_call_judge)
        item = _candidate_items()[0]
        jr.get_verdict(item, "plain", mock=False, client="fake-client", model="fake-model")
        assert captured["system_prompt"] == jc.PLAIN_SYSTEM_PROMPT
        assert "constitution" not in captured["user_prompt"].lower()


class TestArgParser:
    def test_defaults(self):
        args = jr.build_arg_parser().parse_args([])
        assert args.input is None
        assert args.out_dir is None
        assert args.limit is None
        assert args.max_calls is None
        assert args.mock is False

    def test_overrides(self):
        args = jr.build_arg_parser().parse_args(
            [
                "--in",
                "custom_in.jsonl",
                "--out",
                "custom_out_dir",
                "--limit",
                "2",
                "--max-calls",
                "5",
                "--mock",
            ]
        )
        assert args.input == "custom_in.jsonl"
        assert args.out_dir == "custom_out_dir"
        assert args.limit == 2
        assert args.max_calls == 5
        assert args.mock is True

    def test_input_output_aliases(self):
        args = jr.build_arg_parser().parse_args(
            ["--input", "aliased_in.jsonl", "--out-dir", "aliased_out"]
        )
        assert args.input == "aliased_in.jsonl"
        assert args.out_dir == "aliased_out"


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _run_main(monkeypatch, argv):
    import sys

    monkeypatch.setattr(sys, "argv", ["judge_rank.py"] + argv)
    jr.main()


class TestMainEndToEnd:
    def test_mock_writes_both_condition_files_for_all_items(self, tmp_path, monkeypatch):
        input_path = tmp_path / "candidates.jsonl"
        out_dir = tmp_path / "prefs"
        records = _candidate_items()
        _write_jsonl(input_path, records)

        _run_main(
            monkeypatch,
            ["--in", str(input_path), "--out", str(out_dir), "--mock"],
        )

        out_c = out_dir / "condition_c.jsonl"
        out_b = out_dir / "condition_b.jsonl"
        assert out_c.exists() and out_b.exists()

        for path in (out_c, out_b):
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == len(records)
            ids = set()
            for line in lines:
                rec = json.loads(line)
                assert set(rec.keys()) == {"id", "prompt", "chosen", "rejected", "judge_reasoning"}
                assert isinstance(rec["prompt"], list)
                ids.add(rec["id"])
            assert ids == {r["id"] for r in records}

    def test_limit_restricts_number_processed(self, tmp_path, monkeypatch):
        input_path = tmp_path / "candidates.jsonl"
        out_dir = tmp_path / "prefs"
        records = _candidate_items()
        _write_jsonl(input_path, records)

        _run_main(
            monkeypatch,
            ["--in", str(input_path), "--out", str(out_dir), "--mock", "--limit", "1"],
        )

        out_c_lines = (out_dir / "condition_c.jsonl").read_text(encoding="utf-8").strip().splitlines()
        out_b_lines = (out_dir / "condition_b.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(out_c_lines) == 1
        assert len(out_b_lines) == 1

    def test_max_calls_caps_number_of_mocked_calls(self, tmp_path, monkeypatch):
        input_path = tmp_path / "candidates.jsonl"
        out_dir = tmp_path / "prefs"
        records = _candidate_items()  # 3 items -> up to 6 calls (2 per item)
        _write_jsonl(input_path, records)

        _run_main(
            monkeypatch,
            ["--in", str(input_path), "--out", str(out_dir), "--mock", "--max-calls", "4"],
        )

        out_c_ids = {json.loads(line)["id"] for line in (out_dir / "condition_c.jsonl").read_text().splitlines()}
        out_b_ids = {json.loads(line)["id"] for line in (out_dir / "condition_b.jsonl").read_text().splitlines()}
        # 4 calls = item-1 (C+B) + item-2 (C+B); item-3 untouched in both.
        assert out_c_ids == {"item-1", "item-2"}
        assert out_b_ids == {"item-1", "item-2"}

    def test_max_calls_one_only_writes_first_item_first_condition(self, tmp_path, monkeypatch):
        input_path = tmp_path / "candidates.jsonl"
        out_dir = tmp_path / "prefs"
        records = _candidate_items()
        _write_jsonl(input_path, records)

        _run_main(
            monkeypatch,
            ["--in", str(input_path), "--out", str(out_dir), "--mock", "--max-calls", "1"],
        )

        out_c_lines = (out_dir / "condition_c.jsonl").read_text(encoding="utf-8").strip().splitlines()
        out_b_text = (out_dir / "condition_b.jsonl").read_text(encoding="utf-8").strip()
        assert len(out_c_lines) == 1
        assert json.loads(out_c_lines[0])["id"] == "item-1"
        assert out_b_text == ""

    def test_rerun_is_resumable_per_condition_and_does_not_duplicate(self, tmp_path, monkeypatch):
        input_path = tmp_path / "candidates.jsonl"
        out_dir = tmp_path / "prefs"
        out_dir.mkdir()
        records = _candidate_items()
        _write_jsonl(input_path, records)

        # Pre-seed condition_c with item-1 already judged (sentinel reasoning,
        # so we can tell whether a re-run wrongly re-judges it), but leave
        # condition_b untouched for item-1.
        out_c_path = out_dir / "condition_c.jsonl"
        _write_jsonl(
            out_c_path,
            [
                {
                    "id": "item-1",
                    "prompt": jr.build_dpo_prompt_messages(records[0]),
                    "chosen": "PRE_EXISTING_SENTINEL",
                    "rejected": "PRE_EXISTING_SENTINEL",
                    "judge_reasoning": "PRE_EXISTING_SENTINEL",
                }
            ],
        )

        _run_main(
            monkeypatch,
            ["--in", str(input_path), "--out", str(out_dir), "--mock"],
        )

        out_c = [json.loads(line) for line in out_c_path.read_text(encoding="utf-8").strip().splitlines()]
        out_b = [
            json.loads(line)
            for line in (out_dir / "condition_b.jsonl").read_text(encoding="utf-8").strip().splitlines()
        ]

        # No duplicates, all ids present in both, pre-existing item-1 C entry untouched.
        c_ids = [r["id"] for r in out_c]
        b_ids = [r["id"] for r in out_b]
        assert len(c_ids) == len(set(c_ids)) == len(records)
        assert len(b_ids) == len(set(b_ids)) == len(records)
        assert set(c_ids) == set(b_ids) == {r["id"] for r in records}

        item_1_c = next(r for r in out_c if r["id"] == "item-1")
        assert item_1_c["chosen"] == "PRE_EXISTING_SENTINEL"

        item_1_b = next(r for r in out_b if r["id"] == "item-1")
        assert item_1_b["chosen"] != "PRE_EXISTING_SENTINEL"
