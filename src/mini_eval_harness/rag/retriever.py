from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from mini_eval_harness.rag.chunker import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class BM25Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize_text(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._build_idf()

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = tokenize_text(query)
        scored: list[tuple[float, Chunk]] = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)
            scored.append((score, chunk))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self.doc_tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))

        return {
            term: math.log(1 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _score(self, query_terms: list[str], document_index: int) -> float:
        score = 0.0
        frequencies = self.term_frequencies[document_index]
        document_length = self.doc_lengths[document_index]
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            idf = self.idf.get(term, 0.0)
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self.avg_doc_length
            )
            score += idf * numerator / denominator
        return score


def tokenize_text(text: str) -> list[str]:
    normalized = text.lower()
    segments = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)
    tokens: list[str] = []
    for segment in segments:
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            chars = list(segment)
            tokens.extend(chars)
            tokens.extend(
                "".join(chars[index : index + 2])
                for index in range(len(chars) - 1)
            )
        else:
            tokens.append(segment)
    return [token for token in tokens if token]
