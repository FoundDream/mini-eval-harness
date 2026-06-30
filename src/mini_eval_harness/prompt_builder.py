from __future__ import annotations

from pathlib import Path

from mini_eval_harness.dataset_loader import Sample


class PromptBuilder:
    def __init__(self, template_path: str | Path) -> None:
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.template_path}")
        self.template = self.template_path.read_text(encoding="utf-8")

    def build(self, sample: Sample) -> str:
        try:
            return self.template.format(
                id=sample.id,
                question=sample.question,
                answer=sample.answer,
                metadata=sample.metadata,
            )
        except KeyError as exc:
            raise ValueError(
                f"Prompt template references unknown field: {exc.args[0]}"
            ) from exc
