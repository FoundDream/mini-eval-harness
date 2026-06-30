from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # pyright: ignore[reportMissingModuleSource]


DEFAULT_CONFIG: dict[str, Any] = {
    "task": {
        "type": "qa",
    },
    "dataset": "data/demo_qa.jsonl",
    "prompt": "prompts/qa_v1.txt",
    "results_dir": "results",
    "reports_dir": "reports",
    "run_id": None,
    "knowledge_base": {
        "path": "docs/rag_demo",
        "glob": "*.md",
        "chunk_size": 700,
        "chunk_overlap": 80,
    },
    "retriever": {
        "type": "bm25",
        "top_k": 3,
    },
    "agent": {
        "max_steps": 6,
    },
    "evaluator": {
        "type": "heuristic_rag",
        "context_recall_threshold": 0.8,
        "faithfulness_threshold": 0.6,
        "answer_relevancy_threshold": 0.5,
    },
    "model": {
        "provider": "mock",
        "name": None,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "timeout_seconds": 60.0,
        "extra_body": {},
        "hf_device": "auto",
        "hf_dtype": "auto",
        "hf_chat_template": False,
        "hf_trust_remote_code": False,
    },
    "scorer": {
        "type": "exact_match",
    },
}


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_config(args: Any) -> dict[str, Any]:
    config = merge_config(DEFAULT_CONFIG, load_yaml_config(args.config))
    cli_override: dict[str, Any] = {"model": {}}

    top_level_fields = {
        "data": "dataset",
        "prompt": "prompt",
        "results_dir": "results_dir",
        "reports_dir": "reports_dir",
        "run_id": "run_id",
    }
    for arg_name, config_key in top_level_fields.items():
        value = getattr(args, arg_name)
        if value is not None:
            cli_override[config_key] = value

    model_fields = {
        "model_provider": "provider",
        "model_name": "name",
        "api_key_env": "api_key_env",
        "base_url": "base_url",
        "temperature": "temperature",
        "top_p": "top_p",
        "max_tokens": "max_tokens",
        "timeout_seconds": "timeout_seconds",
        "hf_device": "hf_device",
        "hf_dtype": "hf_dtype",
        "hf_chat_template": "hf_chat_template",
        "hf_trust_remote_code": "hf_trust_remote_code",
    }
    for arg_name, config_key in model_fields.items():
        value = getattr(args, arg_name)
        if value is not None:
            cli_override["model"][config_key] = value

    if (
        "provider" in cli_override["model"]
        and "name" not in cli_override["model"]
        and cli_override["model"]["provider"] == "mock"
    ):
        cli_override["model"]["name"] = None

    if not cli_override["model"]:
        cli_override.pop("model")

    return merge_config(config, cli_override)


def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
