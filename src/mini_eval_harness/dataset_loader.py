from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Sample]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        samples: list[Sample] = []
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

                samples.append(self._parse_sample(row, line_number))

        if not samples:
            raise ValueError(f"Dataset is empty: {self.path}")
        return samples

    @staticmethod
    def _parse_sample(row: dict[str, Any], line_number: int) -> Sample:
        required_fields = ("id", "question", "answer")
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Line {line_number} missing fields: {', '.join(missing)}")

        metadata = row.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Line {line_number} metadata must be an object")

        return Sample(
            id=str(row["id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            metadata=metadata,
        )
