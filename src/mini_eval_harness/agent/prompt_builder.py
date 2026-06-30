from __future__ import annotations

from pathlib import Path

from mini_eval_harness.agent.task_loader import AgentTask
from mini_eval_harness.agent.trajectory import TrajectoryStep, compact_trajectory_json
from mini_eval_harness.prompt_loader import load_prompt


class AgentPromptBuilder:
    def __init__(self, template_path: str | Path) -> None:
        self.template_path = Path(template_path)
        self.spec = load_prompt(self.template_path)
        self.template = self.spec.template

    def build(
        self,
        task: AgentTask,
        tool_descriptions: list[str],
        trajectory: list[TrajectoryStep],
    ) -> str:
        try:
            return self.template.format(
                task_id=task.id,
                instruction=task.instruction,
                tool_descriptions="\n".join(f"- {item}" for item in tool_descriptions),
                step_count=len(trajectory),
                trajectory=compact_trajectory_json(trajectory),
            )
        except KeyError as exc:
            raise ValueError(
                f"Agent prompt template references unknown field: {exc.args[0]}"
            ) from exc
