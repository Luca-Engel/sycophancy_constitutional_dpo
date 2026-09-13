import json
import tempfile
from pathlib import Path

import pytest
import train_dpo as td
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _valid_cfg(**overrides) -> dict:
    cfg = {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset_path": "data/preference_pairs/constitutional_dpo.jsonl",
        "output_dir": "outputs/constitutional_dpo",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "learning_rate": 5e-5,
        "beta": 0.1,
        "num_train_epochs": 2,
        "max_steps": -1,
        "precision": "bf16",
    }
    cfg.update(overrides)
    return cfg


class TestLoadConfig:
    def test_round_trips_yaml(self, tmp_path):
        cfg = _valid_cfg()
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        loaded = td.load_config(path)

        assert loaded == cfg

    def test_non_mapping_yaml_raises(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")

        with pytest.raises(ValueError):
            td.load_config(path)


class TestValidateConfig:
    def test_valid_config_passes(self):
        td.validate_config(_valid_cfg())  # no raise

    def test_missing_required_key_raises(self):
        cfg = _valid_cfg()
        del cfg["base_model"]
        with pytest.raises(ValueError, match="base_model"):
            td.validate_config(cfg)

    def test_missing_dataset_path_raises(self):
        cfg = _valid_cfg(dataset_path=None)
        with pytest.raises(ValueError, match="dataset_path"):
            td.validate_config(cfg)

    def test_smoke_test_flag_exempts_dataset_path(self):
        cfg = _valid_cfg(dataset_path=None, _smoke_test_dataset=True)
        td.validate_config(cfg)  # no raise

    def test_missing_lora_key_raises(self):
        cfg = _valid_cfg()
        del cfg["lora"]["dropout"]
        with pytest.raises(ValueError, match="dropout"):
            td.validate_config(cfg)

    def test_lora_not_a_mapping_raises(self):
        cfg = _valid_cfg(lora=["r", 16])
        with pytest.raises(ValueError, match="mapping"):
            td.validate_config(cfg)

    def test_invalid_precision_raises(self):
        cfg = _valid_cfg(precision="int8")
        with pytest.raises(ValueError, match="precision"):
            td.validate_config(cfg)

    def test_use_deepspeed_without_path_raises(self):
        cfg = _valid_cfg(use_deepspeed=True, deepspeed_config_path=None)
        with pytest.raises(ValueError, match="deepspeed_config_path"):
            td.validate_config(cfg)

    def test_use_deepspeed_with_path_passes(self):
        cfg = _valid_cfg(use_deepspeed=True, deepspeed_config_path="configs/deepspeed_zero2.json")
        td.validate_config(cfg)  # no raise

    def test_no_epochs_or_max_steps_raises(self):
        cfg = _valid_cfg(num_train_epochs=None, max_steps=-1)
        with pytest.raises(ValueError, match="num_train_epochs"):
            td.validate_config(cfg)

    def test_positive_max_steps_alone_is_sufficient(self):
        cfg = _valid_cfg(num_train_epochs=None, max_steps=10)
        td.validate_config(cfg)  # no raise


class TestApplySmokeTestOverrides:
    def test_from_scratch_produces_valid_config(self):
        cfg = td.apply_smoke_test_overrides()
        td.validate_config(cfg)  # no raise
        assert cfg["base_model"] == td.SMOKE_TEST_MODEL
        assert cfg["max_steps"] == 1

    def test_layers_on_top_of_loaded_config(self):
        base_cfg = _valid_cfg(output_dir="outputs/constitutional_dpo")
        cfg = td.apply_smoke_test_overrides(base_cfg)

        td.validate_config(cfg)  # no raise
        assert cfg["base_model"] == td.SMOKE_TEST_MODEL
        # output_dir is preserved from the loaded config since the override
        # only sets it via setdefault.
        assert cfg["output_dir"] == "outputs/constitutional_dpo"
        assert cfg["dataset_path"] is None
        assert cfg["_smoke_test_dataset"] is True


class TestExampleConfigs:
    @pytest.mark.parametrize("name", ["train_generic_dpo.yaml", "train_constitutional_dpo.yaml"])
    def test_example_config_is_valid(self, name):
        cfg = td.load_config(REPO_ROOT / "configs" / name)
        td.validate_config(cfg)  # no raise

    def test_generic_and_constitutional_differ_only_in_dataset_and_output(self):
        cfg_generic = td.load_config(REPO_ROOT / "configs" / "train_generic_dpo.yaml")
        cfg_constitutional = td.load_config(REPO_ROOT / "configs" / "train_constitutional_dpo.yaml")

        diff_keys = {k for k in cfg_generic if cfg_generic.get(k) != cfg_constitutional.get(k)}
        assert diff_keys == {"dataset_path", "output_dir"}


class TestLoadPreferencePairs:
    def test_reads_jsonl_records(self, tmp_path):
        records = [
            {"id": "a", "prompt": "hi", "chosen": "x", "rejected": "y"},
            {"id": "b", "prompt": "yo", "chosen": "p", "rejected": "q"},
        ]
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        loaded = td.load_preference_pairs(path)

        assert loaded == records


class TestRecordsToDataset:
    def test_plain_string_prompt_needs_no_tokenizer(self):
        records = [{"id": "a", "prompt": "hi there", "chosen": "x", "rejected": "y"}]

        dataset = td.records_to_dataset(records)

        assert len(dataset) == 1
        assert dataset[0] == {"prompt": "hi there", "chosen": "x", "rejected": "y"}

    def test_conversational_prompt_without_tokenizer_raises(self):
        records = [
            {
                "id": "a",
                "prompt": [{"role": "user", "content": "hi"}],
                "chosen": "x",
                "rejected": "y",
            }
        ]

        with pytest.raises(ValueError, match="tokenizer"):
            td.records_to_dataset(records)

    def test_conversational_prompt_uses_tokenizer_chat_template(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert tokenize is False
                assert add_generation_prompt is True
                return "RENDERED:" + json.dumps(messages)

        records = [
            {
                "id": "a",
                "prompt": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "Are you sure?"},
                ],
                "chosen": "Yes, 4.",
                "rejected": "Maybe not.",
            }
        ]

        dataset = td.records_to_dataset(records, tokenizer=FakeTokenizer())

        assert dataset[0]["prompt"].startswith("RENDERED:")
        assert dataset[0]["chosen"] == "Yes, 4."
        assert dataset[0]["rejected"] == "Maybe not."


class TestBuildSyntheticSmokeDataset:
    def test_has_expected_shape(self):
        dataset = td.build_synthetic_smoke_dataset()

        assert 2 <= len(dataset) <= 4
        assert set(dataset.column_names) == {"prompt", "chosen", "rejected"}
        for row in dataset:
            assert isinstance(row["prompt"], str) and row["prompt"]
            assert isinstance(row["chosen"], str) and row["chosen"]
            assert isinstance(row["rejected"], str) and row["rejected"]


class TestArgParser:
    def test_smoke_test_flag_parses(self):
        args = td.build_arg_parser().parse_args(["--smoke-test"])
        assert args.smoke_test is True
        assert args.config is None

    def test_config_flag_parses(self):
        args = td.build_arg_parser().parse_args(["--config", "configs/train_constitutional_dpo.yaml"])
        assert args.config == "configs/train_constitutional_dpo.yaml"
        assert args.smoke_test is False

    def test_max_steps_and_output_dir_default_to_none(self):
        args = td.build_arg_parser().parse_args(["--config", "configs/train_constitutional_dpo.yaml"])
        assert args.max_steps is None
        assert args.output_dir is None

    def test_max_steps_and_output_dir_parse(self):
        args = td.build_arg_parser().parse_args(
            ["--config", "configs/train_generic_dpo.yaml", "--max-steps", "2", "--output-dir", "outputs/smoke"]
        )
        assert args.max_steps == 2
        assert args.output_dir == "outputs/smoke"


class TestSmokeTestTrainingLoop:
    """Exercises the real model-load -> LoRA wrap -> DPOTrainer -> one
    optimizer step -> adapter save path. Skipped (not failed) wherever the
    heavy `train` optional dependency group isn't installed."""

    def test_smoke_test_runs_end_to_end(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        pytest.importorskip("trl")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = td.apply_smoke_test_overrides()
            cfg["output_dir"] = tmp_dir

            td.run_training(cfg, smoke_test=True)

            assert any(Path(tmp_dir).iterdir())


class TestBuildTrainerRefModelBehavior:
    """``ref_model=None`` + ``peft_config`` (see build_trainer's docstring)
    is only correct if trl actually uses the disable-adapter path for the
    DPO reference model instead of silently referencing the live policy
    model. This is the check that assumption holds for the installed
    trl/peft versions. Skipped (not failed) without the heavy `train`
    optional dependency group installed."""

    def test_ref_model_is_none_and_model_is_peft_wrapped(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        pytest.importorskip("trl")

        from peft import PeftModel

        cfg = td.apply_smoke_test_overrides()
        model, tokenizer = td.build_model_and_tokenizer(cfg)
        lora_config = td.build_lora_config(cfg["lora"])
        dataset = td.build_synthetic_smoke_dataset()
        dpo_config = td.build_dpo_config(cfg)

        trainer = td.build_trainer(model, tokenizer, dataset, lora_config, dpo_config)

        # No separate reference model was loaded...
        assert trainer.ref_model is None
        # ...because the trainer wrapped the policy model with the LoRA
        # adapter itself, which is what makes the disable-adapter reference
        # path available at all.
        assert isinstance(trainer.model, PeftModel)
        assert hasattr(trainer.model, "disable_adapter")
