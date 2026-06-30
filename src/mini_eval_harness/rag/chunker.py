from __future__ import annotations

from dataclasses import dataclass

from mini_eval_harness.rag.document_loader import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


class TextChunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))

        if not chunks:
            raise ValueError("No chunks were produced from the knowledge base")
        return chunks

    def _split_document(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}#chunk_{chunk_index:03d}",
                        doc_id=document.id,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = end - self.chunk_overlap

        return chunks
