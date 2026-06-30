from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentTask:
    id: str
    instruction: str
    tools: list[str]
    initial_state: dict[str, Any]
    success_conditions: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    max_steps: int | None = None


class AgentTaskLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[AgentTask]:
        if not self.path.exists():
            raise FileNotFoundError(f"Agent task dataset not found: {self.path}")

        tasks: list[AgentTask] = []
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
                tasks.append(self._parse_task(row, line_number))

        if not tasks:
            raise ValueError(f"Agent task dataset is empty: {self.path}")
        return tasks

    @staticmethod
    def _parse_task(row: dict[str, Any], line_number: int) -> AgentTask:
        required_fields = (
            "id",
            "instruction",
            "tools",
            "initial_state",
            "success_conditions",
        )
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Line {line_number} missing fields: {', '.join(missing)}")

        tools = row["tools"]
        if not isinstance(tools, list):
            raise ValueError(f"Line {line_number} tools must be a list")

        initial_state = row["initial_state"]
        if not isinstance(initial_state, dict):
            raise ValueError(f"Line {line_number} initial_state must be an object")

        success_conditions = row["success_conditions"]
        if not isinstance(success_conditions, dict):
            raise ValueError(
                f"Line {line_number} success_conditions must be an object"
            )

        metadata = row.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Line {line_number} metadata must be an object")

        max_steps = row.get("max_steps")
        if max_steps is not None:
            max_steps = int(max_steps)

        return AgentTask(
            id=str(row["id"]),
            instruction=str(row["instruction"]),
            tools=[str(tool) for tool in tools],
            initial_state=initial_state,
            success_conditions=success_conditions,
            metadata=metadata,
            max_steps=max_steps,
        )
