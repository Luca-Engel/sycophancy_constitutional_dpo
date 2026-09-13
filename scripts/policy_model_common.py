"""Shared "load the policy model and generate one chat reply" helpers.

Used by both ``scripts/generate_candidates.py`` (candidate generation for
judge ranking) and ``scripts/run_eval.py`` (the held-out eval harness), so
the generation backend -- real HF model, an HTTP-hosted endpoint, or the
offline ``--dry-run`` stub -- isn't duplicated and can't silently drift
between the two call sites.

``torch``/``transformers``/``peft`` are heavy, GPU-run-only dependencies,
imported lazily inside the functions that need them, so this module (and
anything that imports it) stays importable and its CLI parseable on a plain
dev machine with only the core dependency group installed.

``generate_via_http_endpoint`` below is a second real (non-stub) backend
that needs none of those heavy deps -- only ``requests`` (a core
dependency) -- because it delegates the actual model forward pass to an
OpenAI-chat-compatible HTTP endpoint instead of loading weights locally.
This is what lets a small exploratory run (e.g. a subset diagnostic) use
real policy-model generations from a plain laptop, via Hugging Face's
serverless Inference Providers router or a dedicated HF Inference Endpoint,
without installing torch/transformers or renting a GPU box.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("policy_model_common")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def dry_run_generate(messages: list[dict]) -> str:
    """Deterministic placeholder generator used by --dry-run.

    No model, no network: a trivial stub so control flow, output schema,
    and resumability logic can be exercised without any heavy ML
    dependency installed. Deterministic in the message history so repeated
    runs on the same input produce identical output.
    """
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    return f"[stub reply, turn {len(messages)}] {last_user[:80]}"


def load_policy_model(model_name: str, adapter_path: str | None = None):
    """Load the policy model + tokenizer for real generation, optionally
    wrapped with a LoRA adapter directory (as saved by
    ``scripts/train_dpo.py``'s ``trainer.save_model``).

    ``adapter_path`` is unused by ``generate_candidates.py`` (the base
    policy model has no adapter yet at that stage) and used by
    ``run_eval.py`` to evaluate a trained condition.
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
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
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
            max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Characters a genuinely finished reply plausibly ends on: sentence
# punctuation, a closing quote (straight or curly) that follows one, a
# closing bracket/paren, a markdown emphasis/code marker, or a LaTeX math
# delimiter ($ -- e.g. "...\\boxed{10\\sqrt{3}}$", a real, complete ending
# for a math item, initially misflagged before this was added; caught by
# retrying a flagged item at a much higher token budget and finding it
# completed cleanly on exactly this ending). Deliberately a narrow,
# conservative set -- used only to flag a likely max_new_tokens cutoff for
# a human to check via a log warning, never to silently drop or alter
# data, so a false positive (a legitimate ending this set doesn't
# recognize) just costs a glance at the log, while a false negative
# (missing a real truncation) is the failure mode to keep small.
_LIKELY_COMPLETE_ENDINGS = set(".!?\"'”’)]*`$")


def looks_truncated(text: str) -> bool:
    """Best-effort heuristic: does ``text`` look like it was cut off before
    a natural stopping point (e.g. ``max_new_tokens`` reached mid-sentence)
    rather than ending cleanly?

    A reply this model (Qwen3-4B-Instruct-2507, prone to closing an
    enthusiastic/sycophantic-style reply with one or more emoji) often
    ends on is treated as complete regardless of ``_LIKELY_COMPLETE_ENDINGS``
    -- checked via Unicode category rather than an enumerated list, since
    there's no practical way to whitelist every emoji codepoint by hand.
    Otherwise checks the last non-whitespace character against
    ``_LIKELY_COMPLETE_ENDINGS``.

    Approximate, not exact -- found by inspecting real diagnostic runs
    where a majority of one candidate type was silently cut off
    mid-formula/mid-word at an earlier, lower ``max_new_tokens`` cap, and
    separately where most of a later run's flagged items turned out to be
    this exact emoji false positive rather than real truncation. Callers
    should log a warning on a true result, not drop or edit the text: this
    heuristic is a smoke detector, not a filter, and it will still miss a
    different failure mode entirely -- a reply that ends cleanly on
    punctuation but degenerated into a repetition loop well before that
    point never gets flagged by this function at all.
    """
    import unicodedata

    stripped = text.rstrip()
    if not stripped:
        return True
    last = stripped[-1]
    if unicodedata.category(last) in ("So", "Sk"):  # emoji and similar symbol/modifier chars
        return False
    return last not in _LIKELY_COMPLETE_ENDINGS


def get_hf_token() -> str:
    """Read HF_TOKEN from the environment (loading .env first).

    Required by ``generate_via_http_endpoint`` (both Hugging Face's
    serverless router and a dedicated Inference Endpoint expect a bearer
    token). Raises RuntimeError with an actionable message, not a stack
    trace, if unset -- same pattern as judge_common.get_api_key().
    """
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Copy .env.example to .env and fill in a "
            "real Hugging Face token (Inference-enabled), or export "
            "HF_TOKEN directly."
        )
    return token


def _render_chatml_prompt(messages: list[dict]) -> str:
    """Render a plain user/assistant-only message list as a single ChatML
    prompt string: ``<|im_start|>role\\ncontent<|im_end|>\\n`` per turn, plus
    a trailing ``<|im_start|>assistant\\n`` generation prompt.

    This is a narrow, verified reimplementation of Qwen3-4B-Instruct-2507's
    own ``chat_template`` (fetched from its tokenizer_config.json on the
    Hub) for exactly the case this project ever sends -- no system message,
    no tool calls, just alternating user/assistant turns. It is NOT a
    general chat-template renderer; a different model or a message list
    using those other features would need the real template, not this.
    Needed for ``api="text-generation"`` below, which takes raw text, not a
    messages list.
    """
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def generate_via_http_endpoint(
    messages: list[dict],
    endpoint_url: str,
    token: str,
    gen_cfg: dict,
    model_name: str | None = None,
    max_retries: int = 5,
    api: str = "chat",
    timeout: float = 300,
) -> str:
    """Generate one reply via a hosted HTTP endpoint, in one of two API
    shapes real Hugging Face deployments actually use -- which one you get
    is a property of how the endpoint was deployed, not something this
    function can detect automatically, so pick the one that matches
    (a 404 on ``api="chat"`` -- not a 503 -- is the tell that you have the
    other kind; see below).

    - ``api="chat"`` (default): OpenAI-style Messages API at
      ``<endpoint_url>/v1/chat/completions``, sending ``messages`` as-is.
      This is what Hugging Face's serverless Inference Providers router
      (``https://router.huggingface.co``, pass the HF model id as
      ``model_name`` since the router serves many models) exposes, and
      what a dedicated HF Inference Endpoint exposes *if* it was deployed
      with Messages API support.
    - ``api="text-generation"``: the older/default TGI shape at
      ``<endpoint_url>/`` -- ``{"inputs": <prompt text>, "parameters":
      {...}}`` in, ``[{"generated_text": <new text only>}]`` out (via
      ``return_full_text: false``). This is what a dedicated HF Inference
      Endpoint exposes if it was *not* deployed with Messages API support
      -- a common outcome of the default/basic deploy flow. ``messages``
      is rendered to a single prompt string first, via
      ``_render_chatml_prompt``.

    For a dedicated Inference Endpoint, leave ``model_name`` unset either
    way (it's already bound to one model). Bills per minute while running
    regardless of call volume, so pause or delete it when done.

    Retries on a 503 ("model is loading" -- a normal cold-start response)
    and on transient network errors, with exponential backoff, up to
    ``max_retries`` attempts. A 404 on ``api="chat"`` is raised immediately
    instead -- retrying it can never succeed, it means the wrong ``api``
    was picked for this endpoint. Raises RuntimeError if every retryable
    attempt fails.

    ``timeout`` (seconds, default 300) bounds how long one generation
    request is allowed to take, separately from the retry/backoff schedule
    above -- and unlike a 503, a read timeout during active generation
    will NOT be fixed by retrying, since each retry starts a fresh
    generation and hits the same wall again. Scale this up if you raise
    ``gen_cfg["max_new_tokens"]``: a real incident here was a request that
    legitimately needed longer than the previous 120s default once
    ``max_new_tokens`` was raised to 2048, which looked like (and logged
    as) the endpoint still warming up but was actually just slow
    generation.
    """
    import time

    import requests

    if api not in ("chat", "text-generation"):
        raise ValueError(f"api must be 'chat' or 'text-generation', got {api!r}")

    if api == "chat":
        url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
        body: dict = {
            "messages": messages,
            "max_tokens": gen_cfg.get("max_new_tokens", 2048),
            "temperature": gen_cfg.get("temperature", 0.7),
            "top_p": gen_cfg.get("top_p", 0.9),
        }
        if model_name:
            body["model"] = model_name
    else:
        url = f"{endpoint_url.rstrip('/')}/"
        body = {
            "inputs": _render_chatml_prompt(messages),
            "parameters": {
                "max_new_tokens": gen_cfg.get("max_new_tokens", 2048),
                "temperature": gen_cfg.get("temperature", 0.7),
                "top_p": gen_cfg.get("top_p", 0.9),
                "do_sample": gen_cfg.get("do_sample", True),
                "return_full_text": False,
            },
        }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=body, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - network error: retryable
            last_error = exc
        else:
            if resp.status_code == 404 and api == "chat":
                raise RuntimeError(
                    f"404 Not Found at {url} -- this endpoint doesn't expose the OpenAI-style "
                    "chat Messages API (common for a default/basic Inference Endpoint deploy). "
                    "Re-run with api='text-generation' (--endpoint-api text-generation on the "
                    "CLI) instead of retrying -- a 404 here won't resolve itself."
                )
            if resp.status_code == 503:
                last_error = RuntimeError(f"endpoint warming up (503): {resp.text[:200]}")
            else:
                try:
                    resp.raise_for_status()
                    if api == "chat":
                        return resp.json()["choices"][0]["message"]["content"].strip()
                    return resp.json()[0]["generated_text"].strip()
                except Exception as exc:  # noqa: BLE001 - bad response: retryable
                    last_error = exc

        if attempt < max_retries - 1:
            wait_s = min(2**attempt, 60)
            logger.info(
                "generation request did not succeed (attempt %d/%d): %s -- retrying in %ds "
                "(a repeated timeout, as opposed to a 503, likely means generation itself is "
                "slower than the %ds timeout -- raise the `timeout` argument, not max_retries)",
                attempt + 1,
                max_retries,
                last_error,
                wait_s,
                timeout,
            )
            time.sleep(wait_s)

    raise RuntimeError(f"HTTP generation endpoint failed after {max_retries} attempt(s): {last_error}")
