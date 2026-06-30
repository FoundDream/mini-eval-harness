from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentAction:
    thought: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None


@dataclass(frozen=True)
class TrajectoryStep:
    step: int
    model_output: str
    action: dict[str, Any]
    observation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model_output": self.model_output,
            "action": self.action,
            "observation": self.observation,
            "error": self.error,
        }


def parse_agent_action(text: str) -> AgentAction:
    parsed = parse_json_object(text)
    if not isinstance(parsed, dict):
        raise ValueError("Agent output must be a JSON object")

    thought = str(parsed.get("thought", ""))
    final_answer = parsed.get("final_answer")
    if final_answer is not None:
        return AgentAction(thought=thought, final_answer=str(final_answer))

    tool = parsed.get("tool")
    if tool is None:
        raise ValueError("Agent output must contain tool or final_answer")
    args = parsed.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Agent tool args must be an object")
    return AgentAction(thought=thought, tool=str(tool), args=args)


def parse_json_object(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in agent output")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])

    raise ValueError("Unclosed JSON object in agent output")


def compact_trajectory_json(trajectory: list[TrajectoryStep]) -> str:
    if not trajectory:
        return "[]"
    compact = []
    for step in trajectory:
        action = step.action
        compact.append(
            {
                "step": step.step,
                "action": action,
                "observation": step.observation,
                "error": step.error,
            }
        )
    return json.dumps(compact, ensure_ascii=False)
