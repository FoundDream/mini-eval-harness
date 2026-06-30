from __future__ import annotations

from pathlib import Path

from mini_eval_harness.dataset_loader import Sample
from mini_eval_harness.prompt_loader import load_prompt


class PromptBuilder:
    def __init__(self, template_path: str | Path) -> None:
        self.template_path = Path(template_path)
        self.spec = load_prompt(self.template_path)
        self.template = self.spec.template

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
