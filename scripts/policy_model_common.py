"""Shared "load the policy model and generate one chat reply" helpers.

Used by both ``scripts/generate_candidates.py`` (candidate generation for
judge ranking) and ``scripts/run_eval.py`` (the held-out eval harness), so
the generation backend -- real HF model or the offline ``--dry-run`` stub --
isn't duplicated and can't silently drift between the two call sites.

``torch``/``transformers``/``peft`` are heavy, GPU-run-only dependencies,
imported lazily inside the functions that need them, so this module (and
anything that imports it) stays importable and its CLI parseable on a plain
dev machine with only the core dependency group installed.
"""

from __future__ import annotations


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
            max_new_tokens=gen_cfg.get("max_new_tokens", 512),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
