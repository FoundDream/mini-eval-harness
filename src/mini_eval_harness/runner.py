from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_eval_harness.agent.prompt_builder import AgentPromptBuilder
from mini_eval_harness.agent.runner import AgentEvalRunner
from mini_eval_harness.agent.scorer import AgentStateScorer
from mini_eval_harness.agent.task_loader import AgentTaskLoader
from mini_eval_harness.config import build_config, save_yaml_config
from mini_eval_harness.dataset_loader import DatasetLoader, Sample
from mini_eval_harness.model_adapter import (
    HFTransformersAdapter,
    MockAdapter,
    ModelAdapter,
    OpenAIChatAdapter,
)
from mini_eval_harness.prompt_builder import PromptBuilder
from mini_eval_harness.rag.dataset import RagQADatasetLoader, RagQASample
from mini_eval_harness.rag.evaluator import (
    HeuristicRAGEvaluator,
    RAGEvaluator,
    RagasRAGEvaluator,
    contexts_to_json,
)
from mini_eval_harness.rag.pipeline import RAGPipeline
from mini_eval_harness.report import write_markdown_report
from mini_eval_harness.result_logger import ResultLogger
from mini_eval_harness.scorer import ExactMatchScorer, GSM8KFinalAnswerScorer


class Runner:
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        prompt_builder: PromptBuilder,
        model: ModelAdapter,
        scorer: Any,
        logger: ResultLogger,
        run_id: str,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.prompt_builder = prompt_builder
        self.model = model
        self.scorer = scorer
        self.logger = logger
        self.run_id = run_id

    def run(self) -> dict[str, Any]:
        samples = self.dataset_loader.load()
        total_score = 0.0
        error_count = 0

        for sample in samples:
            record = self._run_one(sample)
            self.logger.write(record)
            total_score += float(record["score"])
            if record["error"] is not None:
                error_count += 1

        return {
            "run_id": self.run_id,
            "num_samples": len(samples),
            "accuracy": total_score / len(samples),
            "errors": error_count,
        }

    def _run_one(self, sample: Sample) -> dict[str, Any]:
        prompt = self.prompt_builder.build(sample)
        start = perf_counter()
        error = None
        output_text = ""
        raw_output: dict[str, object] = {}
        score_value = 0.0
        correct = False

        try:
            model_output = self.model.generate(prompt)
            output_text = model_output.text.strip()
            raw_output = model_output.raw
            score = self.scorer.score(output_text, sample.answer)
            score_value = score.score
            correct = score.correct
            score_details = score.details
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            score_details = {}

        latency_ms = round((perf_counter() - start) * 1000, 3)
        return {
            "run_id": self.run_id,
            "sample_id": sample.id,
            "metadata": sample.metadata,
            "prompt": prompt,
            "prediction": output_text,
            "gold": sample.answer,
            "score": score_value,
            "correct": correct,
            "scorer": self.scorer.name,
            "model": self.model.name,
            "raw_output": raw_output,
            "score_details": score_details,
            "latency_ms": latency_ms,
            "error": error,
        }


class RAGRunner:
    def __init__(
        self,
        dataset_loader: RagQADatasetLoader,
        pipeline: RAGPipeline,
        evaluator: RAGEvaluator,
        logger: ResultLogger,
        run_id: str,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.logger = logger
        self.run_id = run_id

    def run(self) -> dict[str, Any]:
        samples = self.dataset_loader.load()
        total_score = 0.0
        correct_count = 0
        error_count = 0

        for sample in samples:
            record = self._run_one(sample)
            self.logger.write(record)
            total_score += float(record["score"])
            if record["correct"]:
                correct_count += 1
            if record["error"] is not None:
                error_count += 1

        return {
            "run_id": self.run_id,
            "num_samples": len(samples),
            "accuracy": correct_count / len(samples),
            "mean_score": total_score / len(samples),
            "errors": error_count,
        }

    def _run_one(self, sample: RagQASample) -> dict[str, Any]:
        start = perf_counter()
        error = None
        prompt = ""
        answer = ""
        raw_output: dict[str, object] = {}
        score_value = 0.0
        correct = False
        metrics: dict[str, float] = {}
        score_details: dict[str, Any] = {}
        retrieved_contexts: list[dict[str, object]] = []

        try:
            output = self.pipeline.run(sample.question)
            prompt = output.prompt
            answer = output.answer
            raw_output = output.raw_output
            retrieved_contexts = contexts_to_json(output.contexts)
            score = self.evaluator.evaluate(sample, output)
            score_value = score.score
            correct = score.correct
            metrics = score.metrics
            score_details = score.details
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = round((perf_counter() - start) * 1000, 3)
        return {
            "run_id": self.run_id,
            "sample_id": sample.id,
            "metadata": sample.metadata,
            "question": sample.question,
            "prompt": prompt,
            "prediction": answer,
            "gold": sample.reference_answer,
            "score": score_value,
            "correct": correct,
            "scorer": self.evaluator.name,
            "model": self.pipeline.model.name,
            "raw_output": raw_output,
            "metrics": metrics,
            "score_details": score_details,
            "retrieved_doc_ids": [
                str(context["doc_id"]) for context in retrieved_contexts
            ],
            "retrieved_contexts": retrieved_contexts,
            "latency_ms": latency_ms,
            "error": error,
        }


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal QA evaluation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-provider", choices=("mock", "openai", "hf"), default=None)
    parser.add_argument("--model-name", "--model-name-or-path", dest="model_name", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--hf-device", default=None)
    parser.add_argument("--hf-dtype", default=None)
    parser.add_argument("--hf-chat-template", action="store_true", default=None)
    parser.add_argument("--hf-trust-remote-code", action="store_true", default=None)
    return parser.parse_args(argv)


def build_model(config: dict[str, Any]) -> ModelAdapter:
    model_config = config["model"]
    provider = model_config["provider"]

    if provider == "mock":
        return MockAdapter()

    if provider == "openai":
        if not model_config["name"]:
            raise ValueError("--model-name is required when --model-provider=openai")
        return OpenAIChatAdapter(
            model=model_config["name"],
            api_key_env=model_config["api_key_env"],
            base_url=model_config["base_url"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            timeout_seconds=model_config["timeout_seconds"],
            extra_body=model_config.get("extra_body", {}),
        )

    if provider == "hf":
        if not model_config["name"]:
            raise ValueError("--model-name is required when --model-provider=hf")
        return HFTransformersAdapter(
            model_name_or_path=model_config["name"],
            device=model_config["hf_device"],
            dtype=model_config["hf_dtype"],
            max_new_tokens=model_config["max_tokens"],
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            use_chat_template=model_config["hf_chat_template"],
            trust_remote_code=model_config["hf_trust_remote_code"],
        )

    raise ValueError(f"Unsupported model provider: {provider}")


def build_scorer(config: dict[str, Any]) -> Any:
    scorer_type = config["scorer"]["type"]
    if scorer_type == "exact_match":
        return ExactMatchScorer()
    if scorer_type == "gsm8k_final_answer":
        return GSM8KFinalAnswerScorer()
    raise ValueError(f"Unsupported scorer type: {scorer_type}")


def build_rag_evaluator(config: dict[str, Any]) -> RAGEvaluator:
    evaluator_type = config.get("evaluator", {}).get("type", "heuristic_rag")
    if evaluator_type == "heuristic_rag":
        return HeuristicRAGEvaluator.from_config(config)
    if evaluator_type == "ragas":
        return RagasRAGEvaluator.from_config(config)
    raise ValueError(f"Unsupported RAG evaluator type: {evaluator_type}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    task_type = config.get("task", {}).get("type", "qa")
    run_id = config.get("run_id") or build_run_id()
    config["run_id"] = run_id

    results_dir = Path(config["results_dir"])
    reports_dir = Path(config["reports_dir"])
    output_path = results_dir / f"run_{run_id}.jsonl"
    saved_config_path = results_dir / f"run_{run_id}_config.yaml"
    report_path = reports_dir / f"run_{run_id}.md"
    config["output_path"] = str(output_path)
    config["saved_config_path"] = str(saved_config_path)
    config["report_path"] = str(report_path)

    try:
        model = build_model(config)
        if task_type == "qa":
            scorer = build_scorer(config)
            runner: Runner | RAGRunner | AgentEvalRunner = Runner(
                dataset_loader=DatasetLoader(config["dataset"]),
                prompt_builder=PromptBuilder(config["prompt"]),
                model=model,
                scorer=scorer,
                logger=ResultLogger(output_path),
                run_id=run_id,
            )
        elif task_type == "rag_qa":
            runner = RAGRunner(
                dataset_loader=RagQADatasetLoader(config["dataset"]),
                pipeline=RAGPipeline.from_config(config, model),
                evaluator=build_rag_evaluator(config),
                logger=ResultLogger(output_path),
                run_id=run_id,
            )
        elif task_type == "agent":
            runner = AgentEvalRunner(
                task_loader=AgentTaskLoader(config["dataset"]),
                prompt_builder=AgentPromptBuilder(config["prompt"]),
                model=model,
                scorer=AgentStateScorer(),
                logger=ResultLogger(output_path),
                run_id=run_id,
                max_steps=int(config.get("agent", {}).get("max_steps", 6)),
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        save_yaml_config(config, saved_config_path)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    summary = runner.run()
    write_markdown_report(output_path, report_path, run_id, config)
    print(f"run_id: {summary['run_id']}")
    print(f"samples: {summary['num_samples']}")
    print(f"accuracy: {summary['accuracy']:.3f}")
    if "mean_score" in summary:
        print(f"mean_score: {summary['mean_score']:.3f}")
    print(f"errors: {summary['errors']}")
    print(f"results: {output_path}")
    print(f"config: {saved_config_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
