from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # pyright: ignore[reportMissingModuleSource]


@dataclass(frozen=True)
class PromptSpec:
    id: str
    version: str
    template: str
    task: str | None = None
    description: str | None = None
    variables: tuple[str, ...] = ()


def load_prompt(path: str | Path) -> PromptSpec:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    if prompt_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Prompt must be a YAML file (.yaml/.yml), got: {prompt_path}"
        )

    raw = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt root must be a mapping: {prompt_path}")

    missing = [field for field in ("id", "version", "template") if field not in raw]
    if missing:
        raise ValueError(
            f"Prompt {prompt_path} missing required field(s): {', '.join(missing)}"
        )

    template = raw["template"]
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Prompt template must be a non-empty string: {prompt_path}")

    variables = raw.get("variables", ())
    if variables == () or variables is None:
        parsed_variables: tuple[str, ...] = ()
    elif isinstance(variables, list) and all(isinstance(item, str) for item in variables):
        parsed_variables = tuple(variables)
    else:
        raise ValueError(f"Prompt variables must be a list of strings: {prompt_path}")

    for field in ("id", "version", "task", "description"):
        value = raw.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Prompt {field} must be a string: {prompt_path}")

    return PromptSpec(
        id=str(raw["id"]),
        version=str(raw["version"]),
        template=template,
        task=raw.get("task"),
        description=raw.get("description"),
        variables=parsed_variables,
    )


def load_prompt_template(path: str | Path) -> str:
    return load_prompt(path).template
