import policy_model_common as pmc
import pytest


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_body


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestGetHfToken:
    def test_raises_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(pmc, "_REPO_ROOT", tmp_path)  # no .env to load here
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            pmc.get_hf_token()

    def test_reads_from_environment(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pmc, "_REPO_ROOT", tmp_path)
        monkeypatch.setenv("HF_TOKEN", "test-token-123")
        assert pmc.get_hf_token() == "test-token-123"


class TestGenerateViaHttpEndpoint:
    def test_success_strips_whitespace(self, monkeypatch):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse(200, _chat_response("  Canberra is the capital.  "))

        monkeypatch.setattr("requests.post", fake_post)

        result = pmc.generate_via_http_endpoint(
            [{"role": "user", "content": "What is the capital of Australia?"}],
            "https://router.huggingface.co",
            "hf-token",
            {"max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9},
            model_name="Qwen/Qwen3-4B-Instruct-2507",
        )

        assert result == "Canberra is the capital."
        assert len(calls) == 1
        assert calls[0]["url"] == "https://router.huggingface.co/v1/chat/completions"
        assert calls[0]["headers"] == {"Authorization": "Bearer hf-token"}
        assert calls[0]["json"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert calls[0]["json"]["max_tokens"] == 128
        assert calls[0]["timeout"] == 300  # default, separate from max_retries/backoff

    def test_custom_timeout_is_passed_through(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["timeout"] = timeout
            return FakeResponse(200, _chat_response("ok"))

        monkeypatch.setattr("requests.post", fake_post)

        pmc.generate_via_http_endpoint([{"role": "user", "content": "hi"}], "https://x", "tok", {}, timeout=600)
        assert captured["timeout"] == 600

    def test_read_timeout_is_retried_not_raised_immediately(self, monkeypatch):
        """A read timeout (generation slower than `timeout`) is a
        retryable condition like a 503, not a fatal error -- distinct from
        the 404 fast-fail path, even though retrying with the same timeout
        won't actually help (see the real incident this documents: raising
        max_new_tokens without raising timeout to match)."""
        import requests as requests_module

        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append(1)
            raise requests_module.exceptions.ReadTimeout("Read timed out.")

        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="failed after 3 attempt"):
            pmc.generate_via_http_endpoint([{"role": "user", "content": "hi"}], "https://x", "tok", {}, max_retries=3)

        assert len(calls) == 3

    def test_omits_model_field_when_not_given(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["json"] = json
            return FakeResponse(200, _chat_response("reply"))

        monkeypatch.setattr("requests.post", fake_post)

        pmc.generate_via_http_endpoint(
            [{"role": "user", "content": "hi"}], "https://my-dedicated-endpoint.example", "tok", {}
        )

        assert "model" not in captured["json"]

    def test_retries_on_503_then_succeeds(self, monkeypatch):
        responses = [FakeResponse(503, text="loading"), FakeResponse(200, _chat_response("ready now"))]

        def fake_post(url, headers, json, timeout):
            return responses.pop(0)

        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = pmc.generate_via_http_endpoint([{"role": "user", "content": "hi"}], "https://x", "tok", {})
        assert result == "ready now"

    def test_raises_after_exhausting_retries(self, monkeypatch):
        def fake_post(url, headers, json, timeout):
            return FakeResponse(503, text="still loading")

        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="failed after 3 attempt"):
            pmc.generate_via_http_endpoint([{"role": "user", "content": "hi"}], "https://x", "tok", {}, max_retries=3)

    def test_strips_trailing_slash_from_endpoint_url(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            return FakeResponse(200, _chat_response("ok"))

        monkeypatch.setattr("requests.post", fake_post)

        pmc.generate_via_http_endpoint([{"role": "user", "content": "hi"}], "https://x/", "tok", {})
        assert captured["url"] == "https://x/v1/chat/completions"

    def test_404_on_chat_api_fails_fast_without_retrying(self, monkeypatch):
        """A 404 on the chat route means the wrong api was picked for this
        endpoint -- retrying can never fix that, so it must not consume the
        retry loop (see the real incident this guards against: a dedicated
        endpoint deployed without Messages API support, which 404s on every
        attempt and would otherwise burn the full backoff schedule, several
        minutes, before giving up)."""
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append(url)
            return FakeResponse(404, text="Not Found")

        monkeypatch.setattr("requests.post", fake_post)
        sleep_calls = []
        monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

        with pytest.raises(RuntimeError, match="text-generation"):
            pmc.generate_via_http_endpoint(
                [{"role": "user", "content": "hi"}], "https://x", "tok", {}, max_retries=14
            )

        assert len(calls) == 1  # not retried
        assert sleep_calls == []  # no backoff wait either

    def test_text_generation_api_success(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse(200, [{"generated_text": "  Canberra.  "}])

        monkeypatch.setattr("requests.post", fake_post)

        result = pmc.generate_via_http_endpoint(
            [{"role": "user", "content": "What is the capital of Australia?"}],
            "https://my-dedicated-endpoint.example",
            "tok",
            {"max_new_tokens": 64, "temperature": 0.7, "top_p": 0.9},
            api="text-generation",
        )

        assert result == "Canberra."
        assert captured["url"] == "https://my-dedicated-endpoint.example/"
        assert captured["json"]["inputs"] == (
            "<|im_start|>user\nWhat is the capital of Australia?<|im_end|>\n<|im_start|>assistant\n"
        )
        assert captured["json"]["parameters"]["max_new_tokens"] == 64
        assert captured["json"]["parameters"]["return_full_text"] is False

    def test_invalid_api_value_raises(self):
        with pytest.raises(ValueError, match="api must be"):
            pmc.generate_via_http_endpoint([{"role": "user", "content": "hi"}], "https://x", "tok", {}, api="bogus")


class TestLooksTruncated:
    def test_complete_sentence_not_flagged(self):
        assert pmc.looks_truncated("Canberra is the capital of Australia.") is False

    def test_complete_question_not_flagged(self):
        assert pmc.looks_truncated("Are you sure about that?") is False

    def test_closing_markdown_bold_not_flagged(self):
        assert pmc.looks_truncated("The answer is **(A)**") is False

    def test_closing_latex_math_delimiter_not_flagged(self):
        """A real false positive hit in practice: a complete math answer
        ending on a LaTeX '$' delimiter (e.g. '...\\boxed{10\\sqrt{3}}$')
        was flagged as truncated before '$' was added to the accepted set."""
        assert pmc.looks_truncated(r"Final answer: $\boxed{10\sqrt{3}}$") is False

    def test_trailing_whitespace_ignored(self):
        assert pmc.looks_truncated("Complete sentence.   \n\n") is False

    def test_trailing_emoji_not_flagged(self):
        """Real false positive hit in practice: this model closes many
        enthusiastic/sycophantic-style replies with an emoji, which the
        punctuation-based accepted set didn't recognize as complete."""
        assert pmc.looks_truncated("Thanks for sharing your thoughts! \U0001f60a") is False

    def test_trailing_multiple_emoji_not_flagged(self):
        assert pmc.looks_truncated("Keep exploring! \U0001f3b9\U0001f33f\U0001f916") is False

    def test_empty_string_flagged(self):
        assert pmc.looks_truncated("") is True

    def test_cut_off_mid_word_flagged(self):
        assert pmc.looks_truncated("Answer: **(A") is True

    def test_cut_off_mid_formula_flagged(self):
        assert pmc.looks_truncated("Characteristic equation:\n\ndet(M -") is True

    def test_cut_off_mid_sentence_flagged(self):
        assert pmc.looks_truncated("the field has largely correctly identified the right direction, not") is True


class TestRenderChatmlPrompt:
    def test_single_user_turn(self):
        prompt = pmc._render_chatml_prompt([{"role": "user", "content": "Hello"}])
        assert prompt == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "Are you sure?"},
        ]
        prompt = pmc._render_chatml_prompt(messages)
        assert prompt == (
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n4.<|im_end|>\n"
            "<|im_start|>user\nAre you sure?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
