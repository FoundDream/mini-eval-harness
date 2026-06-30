from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mini_eval_harness.agent.task_loader import AgentTask
from mini_eval_harness.agent.trajectory import TrajectoryStep


@dataclass(frozen=True)
class AgentScore:
    score: float
    correct: bool
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class AgentStateScorer:
    name = "agent_state"

    def score(
        self,
        task: AgentTask,
        final_state: dict[str, Any],
        trajectory: list[TrajectoryStep],
        final_answer: str,
        runner_error: str | None,
    ) -> AgentScore:
        tool_error_count = sum(1 for step in trajectory if step.error is not None)
        step_count = len(trajectory)
        failure_reason = self._failure_reason(
            task=task,
            final_state=final_state,
            trajectory=trajectory,
            final_answer=final_answer,
            runner_error=runner_error,
            tool_error_count=tool_error_count,
        )
        correct = failure_reason is None
        metrics = {
            "success": 1.0 if correct else 0.0,
            "step_count": float(step_count),
            "tool_error_count": float(tool_error_count),
        }
        return AgentScore(
            score=1.0 if correct else 0.0,
            correct=correct,
            metrics=metrics,
            details={
                "error_type": failure_reason,
                "failure_reason": failure_reason,
                "final_answer": final_answer,
                "called_tools": [
                    str(step.action.get("tool"))
                    for step in trajectory
                    if step.action.get("tool") is not None
                ],
            },
        )

    def _failure_reason(
        self,
        task: AgentTask,
        final_state: dict[str, Any],
        trajectory: list[TrajectoryStep],
        final_answer: str,
        runner_error: str | None,
        tool_error_count: int,
    ) -> str | None:
        if runner_error is not None:
            return "runtime_error"

        conditions = task.success_conditions
        max_tool_errors = int(conditions.get("max_tool_errors", 0))
        if tool_error_count > max_tool_errors:
            return "tool_error_unhandled"

        for message_condition in conditions.get("messages", []):
            if not self._message_condition_met(final_state, message_condition):
                return "missing_required_message"

        todos = conditions.get("todos", {})
        if isinstance(todos, dict):
            for item_id, todo_condition in todos.items():
                reason = self._todo_failure_reason(
                    final_state,
                    str(item_id),
                    todo_condition,
                )
                if reason is not None:
                    return reason

        required_tools = conditions.get("required_tools", [])
        if required_tools:
            called_tools = {
                str(step.action.get("tool"))
                for step in trajectory
                if step.action.get("tool") is not None
            }
            missing = set(str(tool) for tool in required_tools) - called_tools
            if missing:
                return "missed_tool"

        if not final_answer:
            return "missing_final_answer"

        return None

    @staticmethod
    def _message_condition_met(
        final_state: dict[str, Any],
        condition: Any,
    ) -> bool:
        if not isinstance(condition, dict):
            return False
        target = str(condition.get("to", ""))
        required_terms = [str(term) for term in condition.get("contains", [])]
        messages = final_state.get("messages", [])
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("to", "")) != target:
                continue
            text = str(message.get("text", ""))
            if all(term in text for term in required_terms):
                return True
        return False

    @staticmethod
    def _todo_failure_reason(
        final_state: dict[str, Any],
        item_id: str,
        condition: Any,
    ) -> str | None:
        if not isinstance(condition, dict):
            return "todo_condition_invalid"
        todos = final_state.get("todos", {})
        if not isinstance(todos, dict):
            return "todo_missing"
        todo = todos.get(item_id)
        if not isinstance(todo, dict):
            return "todo_missing"

        expected_status = condition.get("status")
        if expected_status is not None and todo.get("status") != expected_status:
            return "todo_status_mismatch"

        required_note_terms = [str(term) for term in condition.get("note_contains", [])]
        note = str(todo.get("note", ""))
        if not all(term in note for term in required_note_terms):
            return "todo_note_mismatch"
        return None
