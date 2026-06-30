from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay an agent eval trajectory.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)

    record = find_record(Path(args.results), args.task_id)
    print(render_replay(record))


def find_record(path: Path, task_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("sample_id") == task_id:
                return record
    raise SystemExit(f"Task id not found in {path}: {task_id}")


def render_replay(record: dict[str, Any]) -> str:
    lines = [
        f"Task: {record.get('sample_id', '<unknown>')}",
        f"Instruction: {record.get('instruction', '')}",
        f"Success: {record.get('correct', False)}",
        f"Failure reason: {(record.get('score_details') or {}).get('failure_reason')}",
        f"Step count: {record.get('step_count', 0)}",
        f"Tool errors: {record.get('tool_error_count', 0)}",
        "",
        "Initial state:",
        json.dumps(record.get("initial_state", {}), ensure_ascii=False, indent=2),
        "",
        "Trajectory:",
    ]

    trajectory = record.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        lines.append("  <empty>")
    else:
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            lines.append(f"  Step {step.get('step')}:")
            lines.append(
                "    action: "
                + json.dumps(step.get("action", {}), ensure_ascii=False)
            )
            lines.append(
                "    observation: "
                + json.dumps(step.get("observation", {}), ensure_ascii=False)
            )
            if step.get("error"):
                lines.append(f"    error: {step.get('error')}")

    lines.extend(
        [
            "",
            "Final state:",
            json.dumps(record.get("final_state", {}), ensure_ascii=False, indent=2),
            "",
            f"Final answer: {record.get('prediction', '')}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
