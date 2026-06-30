from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RagQASample:
    id: str
    question: str
    reference_answer: str
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RagQADatasetLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[RagQASample]:
        if not self.path.exists():
            raise FileNotFoundError(f"RAG dataset not found: {self.path}")

        samples: list[RagQASample] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {self.path}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} must be a JSON object")
                samples.append(self._parse_sample(row, line_number))

        if not samples:
            raise ValueError(f"RAG dataset is empty: {self.path}")
        return samples

    @staticmethod
    def _parse_sample(row: dict[str, Any], line_number: int) -> RagQASample:
        required_fields = ("id", "question")
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Line {line_number} missing fields: {', '.join(missing)}")

        reference_answer = row.get("reference_answer", row.get("answer"))
        if reference_answer is None:
            raise ValueError(
                f"Line {line_number} missing fields: reference_answer or answer"
            )

        metadata = row.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Line {line_number} metadata must be an object")

        return RagQASample(
            id=str(row["id"]),
            question=str(row["question"]),
            reference_answer=str(reference_answer),
            gold_doc_ids=coerce_string_list(row.get("gold_doc_ids", [])),
            gold_evidence=coerce_string_list(row.get("gold_evidence", [])),
            metadata=metadata,
        )


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings")
    return [str(item) for item in value]
