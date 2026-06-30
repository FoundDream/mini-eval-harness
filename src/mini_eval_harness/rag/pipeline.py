from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_eval_harness.model_adapter import ModelAdapter
from mini_eval_harness.rag.chunker import TextChunker
from mini_eval_harness.rag.document_loader import MarkdownDocumentLoader
from mini_eval_harness.rag.retriever import BM25Retriever, RetrievedChunk


@dataclass(frozen=True)
class RAGRunOutput:
    answer: str
    prompt: str
    contexts: list[RetrievedChunk]
    raw_output: dict[str, object]


class RAGPromptBuilder:
    def __init__(self, template_path: str | Path) -> None:
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"RAG prompt template not found: {self.template_path}")
        self.template = self.template_path.read_text(encoding="utf-8")

    def build(self, question: str, contexts: list[RetrievedChunk]) -> str:
        formatted_contexts = "\n\n".join(
            self._format_context(context) for context in contexts
        )
        try:
            return self.template.format(
                question=question,
                contexts=formatted_contexts,
            )
        except KeyError as exc:
            raise ValueError(
                f"RAG prompt template references unknown field: {exc.args[0]}"
            ) from exc

    @staticmethod
    def _format_context(context: RetrievedChunk) -> str:
        return (
            f"[{context.rank}] 来源: {context.chunk.doc_id}\n"
            f"{context.chunk.text.strip()}"
        )


class RAGPipeline:
    def __init__(
        self,
        prompt_builder: RAGPromptBuilder,
        retriever: BM25Retriever,
        model: ModelAdapter,
        top_k: int = 3,
    ) -> None:
        self.prompt_builder = prompt_builder
        self.retriever = retriever
        self.model = model
        self.top_k = top_k

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        model: ModelAdapter,
    ) -> "RAGPipeline":
        knowledge_base = config["knowledge_base"]
        retriever_config = config["retriever"]

        documents = MarkdownDocumentLoader(
            path=knowledge_base["path"],
            glob_pattern=knowledge_base.get("glob", "*.md"),
        ).load()
        chunks = TextChunker(
            chunk_size=int(knowledge_base.get("chunk_size", 700)),
            chunk_overlap=int(knowledge_base.get("chunk_overlap", 80)),
        ).split(documents)

        retriever_type = retriever_config.get("type", "bm25")
        if retriever_type != "bm25":
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        return cls(
            prompt_builder=RAGPromptBuilder(config["prompt"]),
            retriever=BM25Retriever(chunks),
            model=model,
            top_k=int(retriever_config.get("top_k", 3)),
        )

    def run(self, question: str) -> RAGRunOutput:
        contexts = self.retriever.retrieve(question, top_k=self.top_k)
        prompt = self.prompt_builder.build(question, contexts)
        model_output = self.model.generate(prompt)
        return RAGRunOutput(
            answer=model_output.text.strip(),
            prompt=prompt,
            contexts=contexts,
            raw_output=model_output.raw,
        )
