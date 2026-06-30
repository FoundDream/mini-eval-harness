from __future__ import annotations

from time import perf_counter
from typing import Any

from mini_eval_harness.agent.environment import AgentEnvironment
from mini_eval_harness.agent.prompt_builder import AgentPromptBuilder
from mini_eval_harness.agent.scorer import AgentStateScorer
from mini_eval_harness.agent.task_loader import AgentTask, AgentTaskLoader
from mini_eval_harness.agent.trajectory import (
    TrajectoryStep,
    parse_agent_action,
)
from mini_eval_harness.model_adapter import ModelAdapter
from mini_eval_harness.result_logger import ResultLogger


class AgentEvalRunner:
    def __init__(
        self,
        task_loader: AgentTaskLoader,
        prompt_builder: AgentPromptBuilder,
        model: ModelAdapter,
        scorer: AgentStateScorer,
        logger: ResultLogger,
        run_id: str,
        max_steps: int = 6,
    ) -> None:
        self.task_loader = task_loader
        self.prompt_builder = prompt_builder
        self.model = model
        self.scorer = scorer
        self.logger = logger
        self.run_id = run_id
        self.max_steps = max_steps

    def run(self) -> dict[str, Any]:
        tasks = self.task_loader.load()
        success_count = 0
        total_score = 0.0
        error_count = 0

        for task in tasks:
            record = self._run_one(task)
            self.logger.write(record)
            total_score += float(record["score"])
            if record["correct"]:
                success_count += 1
            if record["error"] is not None:
                error_count += 1

        return {
            "run_id": self.run_id,
            "num_samples": len(tasks),
            "accuracy": success_count / len(tasks),
            "mean_score": total_score / len(tasks),
            "errors": error_count,
        }

    def _run_one(self, task: AgentTask) -> dict[str, Any]:
        start = perf_counter()
        environment = AgentEnvironment(task.initial_state, task.tools)
        trajectory: list[TrajectoryStep] = []
        raw_outputs: list[dict[str, object]] = []
        final_answer = ""
        error = None
        max_steps = task.max_steps or self.max_steps

        try:
            for step_number in range(1, max_steps + 1):
                prompt = self.prompt_builder.build(
                    task=task,
                    tool_descriptions=environment.tool_descriptions(),
                    trajectory=trajectory,
                )
                model_output = self.model.generate(prompt)
                raw_outputs.append(model_output.raw)

                try:
                    action = parse_agent_action(model_output.text)
                except ValueError as exc:
                    trajectory.append(
                        TrajectoryStep(
                            step=step_number,
                            model_output=model_output.text,
                            action={},
                            observation={},
                            error=f"invalid_action: {exc}",
                        )
                    )
                    break

                if action.final_answer is not None:
                    final_answer = action.final_answer
                    trajectory.append(
                        TrajectoryStep(
                            step=step_number,
                            model_output=model_output.text,
                            action={
                                "thought": action.thought,
                                "final_answer": action.final_answer,
                            },
                            observation={},
                        )
                    )
                    break

                if action.tool is None:
                    trajectory.append(
                        TrajectoryStep(
                            step=step_number,
                            model_output=model_output.text,
                            action={"thought": action.thought},
                            observation={},
                            error="invalid_action: missing tool",
                        )
                    )
                    break

                tool_result = environment.execute(action.tool, action.args)
                trajectory.append(
                    TrajectoryStep(
                        step=step_number,
                        model_output=model_output.text,
                        action={
                            "thought": action.thought,
                            "tool": action.tool,
                            "args": action.args,
                        },
                        observation=tool_result.output,
                        error=tool_result.error,
                    )
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        final_state = environment.snapshot()
        score = self.scorer.score(
            task=task,
            final_state=final_state,
            trajectory=trajectory,
            final_answer=final_answer,
            runner_error=error,
        )
        latency_ms = round((perf_counter() - start) * 1000, 3)
        tool_error_count = int(score.metrics["tool_error_count"])

        return {
            "run_id": self.run_id,
            "sample_id": task.id,
            "metadata": task.metadata,
            "instruction": task.instruction,
            "prompt": "",
            "prediction": final_answer,
            "gold": task.success_conditions,
            "score": score.score,
            "correct": score.correct,
            "scorer": self.scorer.name,
            "model": self.model.name,
            "raw_output": {"steps": raw_outputs},
            "metrics": score.metrics,
            "score_details": score.details,
            "initial_state": task.initial_state,
            "final_state": final_state,
            "trajectory": [step.to_json() for step in trajectory],
            "step_count": len(trajectory),
            "tool_error_count": tool_error_count,
            "latency_ms": latency_ms,
            "error": error,
        }
