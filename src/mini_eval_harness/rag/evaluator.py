from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from mini_eval_harness.rag.dataset import RagQASample
from mini_eval_harness.rag.pipeline import RAGRunOutput
from mini_eval_harness.rag.retriever import RetrievedChunk, tokenize_text


@dataclass(frozen=True)
class RAGScore:
    score: float
    correct: bool
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class RAGEvaluator(Protocol):
    name: str

    def evaluate(self, sample: RagQASample, output: RAGRunOutput) -> RAGScore:
        ...


class HeuristicRAGEvaluator:
    name = "heuristic_rag"

    def __init__(
        self,
        context_recall_threshold: float = 0.8,
        faithfulness_threshold: float = 0.6,
        answer_relevancy_threshold: float = 0.5,
    ) -> None:
        self.context_recall_threshold = context_recall_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevancy_threshold = answer_relevancy_threshold

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HeuristicRAGEvaluator":
        evaluator_config = config.get("evaluator", {})
        evaluator_type = evaluator_config.get("type", "heuristic_rag")
        if evaluator_type != "heuristic_rag":
            raise ValueError(f"Unsupported RAG evaluator type: {evaluator_type}")
        return cls(
            context_recall_threshold=float(
                evaluator_config.get("context_recall_threshold", 0.8)
            ),
            faithfulness_threshold=float(
                evaluator_config.get("faithfulness_threshold", 0.6)
            ),
            answer_relevancy_threshold=float(
                evaluator_config.get("answer_relevancy_threshold", 0.5)
            ),
        )

    def evaluate(self, sample: RagQASample, output: RAGRunOutput) -> RAGScore:
        retrieved_doc_ids = [context.chunk.doc_id for context in output.contexts]
        retrieved_context = "\n".join(context.chunk.text for context in output.contexts)

        hit_at_k = self._hit_at_k(sample.gold_doc_ids, retrieved_doc_ids)
        context_recall = self._context_recall(sample, retrieved_context, hit_at_k)
        faithfulness = token_overlap(output.answer, retrieved_context)
        answer_relevancy = token_overlap(output.answer, sample.reference_answer)

        metrics = {
            "hit_at_k": hit_at_k,
            "context_recall": context_recall,
            "faithfulness_proxy": faithfulness,
            "answer_relevancy_proxy": answer_relevancy,
        }
        score = sum(metrics.values()) / len(metrics)
        error_type = self._error_type(metrics)

        return RAGScore(
            score=score,
            correct=error_type is None,
            metrics=metrics,
            details={
                "error_type": error_type,
                "gold_doc_ids": sample.gold_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "missing_gold_doc_ids": sorted(
                    set(sample.gold_doc_ids) - set(retrieved_doc_ids)
                ),
                "gold_evidence": sample.gold_evidence,
            },
        )

    @staticmethod
    def _hit_at_k(gold_doc_ids: list[str], retrieved_doc_ids: list[str]) -> float:
        if not gold_doc_ids:
            return 0.0
        return 1.0 if set(gold_doc_ids).intersection(retrieved_doc_ids) else 0.0

    @staticmethod
    def _context_recall(
        sample: RagQASample,
        retrieved_context: str,
        fallback_hit_at_k: float,
    ) -> float:
        if not sample.gold_evidence:
            return fallback_hit_at_k

        normalized_context = normalize_text(retrieved_context)
        matched = 0
        for evidence in sample.gold_evidence:
            if normalize_text(evidence) in normalized_context:
                matched += 1
        return matched / len(sample.gold_evidence)

    def _error_type(self, metrics: dict[str, float]) -> str | None:
        if metrics["context_recall"] < self.context_recall_threshold:
            return "retrieval_miss"
        if metrics["faithfulness_proxy"] < self.faithfulness_threshold:
            return "generation_unfaithful"
        if metrics["answer_relevancy_proxy"] < self.answer_relevancy_threshold:
            return "answer_not_relevant"
        return None


class RagasRAGEvaluator:
    name = "ragas"

    def __init__(
        self,
        metric_names: list[str],
        llm_config: dict[str, Any],
        embeddings_config: dict[str, Any] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.metric_names = metric_names
        self.thresholds = thresholds or {}
        self.llm = self._build_llm(llm_config)
        self.embeddings = (
            self._build_embeddings(embeddings_config)
            if embeddings_config is not None
            else None
        )
        self.scorers = self._build_scorers()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RagasRAGEvaluator":
        evaluator_config = config.get("evaluator", {})
        evaluator_type = evaluator_config.get("type", "heuristic_rag")
        if evaluator_type != "ragas":
            raise ValueError(f"Unsupported RAG evaluator type: {evaluator_type}")

        metric_names = [
            str(metric_name)
            for metric_name in evaluator_config.get(
                "metrics",
                ["faithfulness", "context_recall", "context_precision"],
            )
        ]
        thresholds = {
            str(name): float(value)
            for name, value in evaluator_config.get("thresholds", {}).items()
        }
        return cls(
            metric_names=metric_names,
            llm_config=evaluator_config.get("llm", {}),
            embeddings_config=evaluator_config.get("embeddings"),
            thresholds=thresholds,
        )

    def evaluate(self, sample: RagQASample, output: RAGRunOutput) -> RAGScore:
        retrieved_doc_ids = [context.chunk.doc_id for context in output.contexts]
        contexts = [context.chunk.text for context in output.contexts]
        hit_at_k = HeuristicRAGEvaluator._hit_at_k(
            sample.gold_doc_ids,
            retrieved_doc_ids,
        )

        metrics = {"hit_at_k": hit_at_k}
        for metric_name, scorer in self.scorers.items():
            result = self._score_metric(metric_name, scorer, sample, output, contexts)
            metrics[f"ragas_{metric_name}"] = result

        score = sum(metrics.values()) / len(metrics)
        error_type = self._error_type(metrics)
        return RAGScore(
            score=score,
            correct=error_type is None,
            metrics=metrics,
            details={
                "error_type": error_type,
                "gold_doc_ids": sample.gold_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "missing_gold_doc_ids": sorted(
                    set(sample.gold_doc_ids) - set(retrieved_doc_ids)
                ),
                "gold_evidence": sample.gold_evidence,
                "ragas_metrics": self.metric_names,
                "thresholds": self.thresholds,
            },
        )

    @staticmethod
    def _build_llm(llm_config: dict[str, Any]) -> Any:
        try:
            openai_module: Any = importlib.import_module("openai")
            ragas_llms: Any = importlib.import_module("ragas.llms")
        except ImportError as exc:
            raise RuntimeError(
                "RagasRAGEvaluator requires ragas and openai. "
                'Install them with: pip install -e ".[ragas]"'
            ) from exc

        model = llm_config.get("model")
        if not model:
            raise ValueError("evaluator.llm.model is required for Ragas")

        api_key_env = str(llm_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing evaluator API key env var: {api_key_env}")

        client_kwargs: dict[str, object] = {"api_key": api_key}
        base_url = llm_config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = str(base_url)

        client = openai_module.AsyncOpenAI(**client_kwargs)
        return ragas_llms.llm_factory(str(model), client=client)

    @staticmethod
    def _build_embeddings(embeddings_config: dict[str, Any] | None) -> Any:
        if embeddings_config is None:
            return None
        try:
            openai_module: Any = importlib.import_module("openai")
            ragas_embeddings: Any = importlib.import_module("ragas.embeddings.base")
        except ImportError as exc:
            raise RuntimeError(
                "Ragas answer_relevancy requires ragas and openai. "
                'Install them with: pip install -e ".[ragas]"'
            ) from exc

        provider = str(embeddings_config.get("provider", "openai"))
        model = embeddings_config.get("model")
        if not model:
            raise ValueError("evaluator.embeddings.model is required")

        api_key_env = str(embeddings_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing embeddings API key env var: {api_key_env}")

        client_kwargs: dict[str, object] = {"api_key": api_key}
        base_url = embeddings_config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = str(base_url)

        client = openai_module.AsyncOpenAI(**client_kwargs)
        return ragas_embeddings.embedding_factory(
            provider,
            model=str(model),
            client=client,
        )

    def _build_scorers(self) -> dict[str, Any]:
        try:
            collections: Any = importlib.import_module("ragas.metrics.collections")
        except ImportError as exc:
            raise RuntimeError(
                "RagasRAGEvaluator requires ragas. "
                'Install it with: pip install -e ".[ragas]"'
            ) from exc

        scorers: dict[str, Any] = {}
        for metric_name in self.metric_names:
            if metric_name == "faithfulness":
                scorers[metric_name] = collections.Faithfulness(llm=self.llm)
            elif metric_name == "context_recall":
                scorers[metric_name] = collections.ContextRecall(llm=self.llm)
            elif metric_name == "context_precision":
                scorers[metric_name] = collections.ContextPrecision(llm=self.llm)
            elif metric_name == "answer_relevancy":
                if self.embeddings is None:
                    raise ValueError(
                        "Ragas answer_relevancy requires evaluator.embeddings config"
                    )
                scorers[metric_name] = collections.AnswerRelevancy(
                    llm=self.llm,
                    embeddings=self.embeddings,
                )
            else:
                raise ValueError(f"Unsupported Ragas metric: {metric_name}")
        return scorers

    @staticmethod
    def _score_metric(
        metric_name: str,
        scorer: Any,
        sample: RagQASample,
        output: RAGRunOutput,
        contexts: list[str],
    ) -> float:
        if metric_name == "faithfulness":
            result = scorer.score(
                user_input=sample.question,
                response=output.answer,
                retrieved_contexts=contexts,
            )
        elif metric_name == "context_recall":
            result = scorer.score(
                user_input=sample.question,
                reference=sample.reference_answer,
                retrieved_contexts=contexts,
            )
        elif metric_name == "context_precision":
            result = scorer.score(
                user_input=sample.question,
                reference=sample.reference_answer,
                retrieved_contexts=contexts,
            )
        elif metric_name == "answer_relevancy":
            result = scorer.score(
                user_input=sample.question,
                response=output.answer,
            )
        else:
            raise ValueError(f"Unsupported Ragas metric: {metric_name}")
        return float(getattr(result, "value", result))

    def _error_type(self, metrics: dict[str, float]) -> str | None:
        checks = [
            ("hit_at_k", 1.0, "retrieval_miss"),
            ("ragas_context_recall", 0.8, "retrieval_miss"),
            ("ragas_context_precision", 0.7, "retrieval_noise"),
            ("ragas_faithfulness", 0.8, "generation_unfaithful"),
            ("ragas_answer_relevancy", 0.7, "answer_not_relevant"),
        ]
        for metric_name, default_threshold, error_type in checks:
            if metric_name not in metrics:
                continue
            threshold = self.thresholds.get(metric_name, default_threshold)
            if metrics[metric_name] < threshold:
                return error_type
        return None


def token_overlap(candidate: str, reference: str) -> float:
    candidate_tokens = set(filter_meaningful_tokens(tokenize_text(candidate)))
    reference_tokens = set(filter_meaningful_tokens(tokenize_text(reference)))
    if not candidate_tokens:
        return 0.0
    if not reference_tokens:
        return 0.0
    return len(candidate_tokens.intersection(reference_tokens)) / len(candidate_tokens)


def filter_meaningful_tokens(tokens: list[str]) -> list[str]:
    stopwords = {
        "的",
        "了",
        "在",
        "和",
        "或",
        "与",
        "及",
        "中",
        "为",
        "是",
        "可以",
        "需要",
        "员工",
    }
    return [token for token in tokens if token not in stopwords]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def contexts_to_json(contexts: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [
        {
            "rank": context.rank,
            "doc_id": context.chunk.doc_id,
            "chunk_id": context.chunk.id,
            "score": round(context.score, 6),
            "text": context.chunk.text,
            "start_char": context.chunk.start_char,
            "end_char": context.chunk.end_char,
        }
        for context in contexts
    ]
