#!/usr/bin/env python
"""Push trained LoRA adapters in outputs/ to the Hugging Face Hub.

Each model directory (e.g. outputs/constitutional_dpo_v2) becomes its own
repo, named "<namespace>/sycophancy-<model-dir-with-dashes>". Only the
top-level adapter files are uploaded (adapter_model.safetensors,
adapter_config.json, tokenizer files, chat_template.jinja, README.md);
checkpoint-* subfolders are skipped since they're intermediate training
state, not needed to use the adapter.

Usage:
    python scripts/push_to_hub.py                     # push all known models, public
    python scripts/push_to_hub.py --models generic_dpo_v2
    python scripts/push_to_hub.py --private
    python scripts/push_to_hub.py --namespace some-org
    python scripts/push_to_hub.py --dry-run
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

ALL_MODELS = [
    "constitutional_dpo",
    "constitutional_dpo_v2",
    "generic_dpo",
    "generic_dpo_v2",
]


def repo_name_for(model_dir: str) -> str:
    return f"sycophancy-{model_dir.replace('_', '-')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                         help="Which model directories under outputs/ to push (default: all)")
    parser.add_argument("--namespace", default=None,
                         help="HF username or org to push under (default: your logged-in user)")
    parser.add_argument("--private", action="store_true", help="Create/update repos as private")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without uploading")
    args = parser.parse_args()

    api = HfApi()
    namespace = args.namespace or api.whoami()["name"]

    for model_dir in args.models:
        local_path = OUTPUTS_DIR / model_dir
        if not local_path.is_dir():
            print(f"skip {model_dir}: {local_path} does not exist")
            continue

        repo_id = f"{namespace}/{repo_name_for(model_dir)}"
        print(f"{model_dir} -> https://huggingface.co/{repo_id} ({'private' if args.private else 'public'})")

        if args.dry_run:
            continue

        api.create_repo(repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            ignore_patterns=["checkpoint-*", "__pycache__"],
        )
        print(f"  done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
