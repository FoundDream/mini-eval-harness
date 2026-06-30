from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    result_path = Path(path)
    with result_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {result_path}"
                ) from exc
            records.append(record)
    return records


def write_markdown_report(
    result_path: str | Path,
    report_path: str | Path,
    run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    records = load_records(result_path)
    summary = summarize_records(records)
    markdown = render_markdown(summary, records, run_id, config, result_path)

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return summary


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    total_score = sum(float(record.get("score", 0.0)) for record in records)
    errors = [record for record in records if record.get("error") is not None]
    incorrect = [record for record in records if not record.get("correct", False)]

    summary = {
        "num_samples": total,
        "accuracy": total_score / total if total else 0.0,
        "errors": len(errors),
        "incorrect": len(incorrect),
        "error_types": summarize_error_types(records),
        "metrics": summarize_metrics(records),
        "groups": {
            "category": summarize_by_metadata(records, "category"),
            "difficulty": summarize_by_metadata(records, "difficulty"),
        },
    }
    return summary


def summarize_by_metadata(
    records: list[dict[str, Any]], metadata_key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record.get("metadata") or {}
        value = metadata.get(metadata_key, "unknown")
        grouped[str(value)].append(record)

    rows = []
    for value, group in sorted(grouped.items()):
        total = len(group)
        total_score = sum(float(record.get("score", 0.0)) for record in group)
        rows.append(
            {
                "value": value,
                "num_samples": total,
                "accuracy": total_score / total if total else 0.0,
            }
        )
    return rows


def summarize_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        metrics = record.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped[str(name)].append(float(value))

    rows = []
    for name, values in sorted(grouped.items()):
        rows.append(
            {
                "metric": name,
                "count": len(values),
                "mean": sum(values) / len(values) if values else 0.0,
            }
        )
    return rows


def summarize_error_types(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        details = record.get("score_details") or {}
        error_type = details.get("error_type")
        if record.get("error") is not None:
            error_type = "runtime_error"
        if error_type is None:
            continue
        grouped[str(error_type)].append(record)

    rows = []
    for error_type, group in sorted(grouped.items()):
        rows.append({"error_type": error_type, "count": len(group)})
    return rows


def render_markdown(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    run_id: str,
    config: dict[str, Any],
    result_path: str | Path,
) -> str:
    lines = [
        f"# Evaluation Report: {run_id}",
        "",
        "## Summary",
        "",
        f"- Results: `{result_path}`",
        f"- Task: `{config.get('task', {}).get('type', 'qa')}`",
        f"- Dataset: `{config['dataset']}`",
        f"- Prompt: `{config['prompt']}`",
        f"- Model provider: `{config['model']['provider']}`",
        f"- Model: `{config['model'].get('name') or config['model']['provider']}`",
        f"- Samples: {summary['num_samples']}",
        f"- Accuracy: {summary['accuracy']:.3f}",
        f"- Incorrect: {summary['incorrect']}",
        f"- Errors: {summary['errors']}",
        "",
    ]
    task_type = config.get("task", {}).get("type", "qa")
    if task_type == "rag_qa":
        lines.insert(10, f"- Evaluator: `{config['evaluator']['type']}`")
        lines.insert(11, f"- Retriever: `{config['retriever']['type']}`")
    elif task_type == "agent":
        lines.insert(10, "- Scorer: `agent_state`")
        lines.insert(11, f"- Max steps: `{config.get('agent', {}).get('max_steps', '')}`")
    else:
        lines.insert(10, f"- Scorer: `{config['scorer']['type']}`")

    for group_name, rows in summary["groups"].items():
        lines.extend(render_group_table(group_name, rows))

    lines.extend(render_metric_table(summary["metrics"]))
    lines.extend(render_error_type_table(summary["error_types"]))

    wrong_records = [
        record
        for record in records
        if not record.get("correct", False) or record.get("error") is not None
    ]
    lines.extend(render_wrong_cases(wrong_records))

    return "\n".join(lines).rstrip() + "\n"


def render_group_table(group_name: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## By {group_name}",
        "",
        f"| {group_name} | samples | accuracy |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {escape_markdown(row['value'])} | "
            f"{row['num_samples']} | "
            f"{row['accuracy']:.3f} |"
        )
    lines.append("")
    return lines


def render_metric_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Metrics", ""]
    if not rows:
        lines.append("No extra metrics.")
        lines.append("")
        return lines

    lines.extend(["| metric | count | mean |", "|---|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {escape_markdown(row['metric'])} | "
            f"{row['count']} | "
            f"{row['mean']:.3f} |"
        )
    lines.append("")
    return lines


def render_error_type_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Error Types", ""]
    if not rows:
        lines.append("No error types.")
        lines.append("")
        return lines

    lines.extend(["| error_type | count |", "|---|---:|"])
    for row in rows:
        lines.append(f"| {escape_markdown(row['error_type'])} | {row['count']} |")
    lines.append("")
    return lines


def render_wrong_cases(records: list[dict[str, Any]]) -> list[str]:
    lines = ["## Wrong Cases", ""]
    if not records:
        lines.append("No wrong cases.")
        lines.append("")
        return lines

    for record in records:
        details = record.get("score_details") or {}
        metrics = record.get("metrics") or {}
        lines.extend(
            [
                f"### {record.get('sample_id', '<unknown>')}",
                "",
                f"- Gold: `{escape_inline_code(str(record.get('gold', '')))}`",
                f"- Prediction: `{escape_inline_code(str(record.get('prediction', '')))}`",
                f"- Retrieved docs: `{escape_inline_code(format_retrieved_docs(record))}`",
                f"- Metrics: `{escape_inline_code(format_metrics(metrics))}`",
                f"- Failure reason: `{escape_inline_code(str(details.get('failure_reason', '')))}`",
                f"- Step count: `{escape_inline_code(str(record.get('step_count', '')))}`",
                f"- Tool errors: `{escape_inline_code(str(record.get('tool_error_count', '')))}`",
                f"- Trajectory: `{escape_inline_code(format_trajectory(record))}`",
                f"- Extracted gold: `{escape_inline_code(str(details.get('gold_answer', '')))}`",
                f"- Extracted prediction: `{escape_inline_code(str(details.get('predicted_answer', '')))}`",
                f"- Error type: `{escape_inline_code(str(details.get('error_type', '')))}`",
                f"- Score: {float(record.get('score', 0.0)):.3f}",
                f"- Error: `{escape_inline_code(str(record.get('error')) if record.get('error') else '')}`",
                "",
            ]
        )
    return lines


def format_trajectory(record: dict[str, Any]) -> str:
    trajectory = record.get("trajectory") or []
    if not isinstance(trajectory, list):
        return ""

    pieces = []
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or {}
        if not isinstance(action, dict):
            action = {}
        if "tool" in action:
            pieces.append(
                f"{step.get('step')}: {action.get('tool')}({action.get('args', {})})"
            )
        elif "final_answer" in action:
            pieces.append(f"{step.get('step')}: final")
        else:
            pieces.append(f"{step.get('step')}: invalid")
    return " -> ".join(pieces)


def format_retrieved_docs(record: dict[str, Any]) -> str:
    doc_ids = record.get("retrieved_doc_ids") or []
    if not isinstance(doc_ids, list):
        return ""
    return ", ".join(str(doc_id) for doc_id in doc_ids)


def format_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    pairs = []
    for name, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            pairs.append(f"{name}={float(value):.3f}")
        else:
            pairs.append(f"{name}={value}")
    return ", ".join(pairs)


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def escape_inline_code(value: str) -> str:
    return value.replace("`", "\\`").replace("\n", "\\n")
