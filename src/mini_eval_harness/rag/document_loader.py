from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    path: str
    text: str


class MarkdownDocumentLoader:
    def __init__(self, path: str | Path, glob_pattern: str = "*.md") -> None:
        self.path = Path(path)
        self.glob_pattern = glob_pattern

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Knowledge base path not found: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Knowledge base path must be a directory: {self.path}")

        documents: list[Document] = []
        for file_path in sorted(self.path.rglob(self.glob_pattern)):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(self.path)
            documents.append(
                Document(
                    id=relative_path.as_posix(),
                    path=str(file_path),
                    text=file_path.read_text(encoding="utf-8"),
                )
            )

        if not documents:
            raise ValueError(
                f"No documents found under {self.path} matching {self.glob_pattern}"
            )
        return documents
