from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentPage:
    """读取后的单页或单段文档。"""

    source: str
    page: int | None
    text: str


@dataclass(slots=True)
class TextChunk:
    """用于检索的制度片段。"""

    chunk_id: str
    text: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    chapter_title: str | None = None
    clause_numbers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            chapter_title=data.get("chapter_title"),
            clause_numbers=list(data.get("clause_numbers") or []),
        )


@dataclass(slots=True)
class SearchHit:
    """一次召回或重排后的检索结果。"""

    chunk: TextChunk
    similarity: float
    keyword_score: float = 0.0
    completeness_score: float = 0.0
    final_score: float = 0.0
    numeric_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunk"] = self.chunk.to_dict()
        return payload


@dataclass(slots=True)
class AnswerResult:
    """问答接口返回结果。"""

    question: str
    answer: str
    hits: list[SearchHit]
    refused: bool = False
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "hits": [hit.to_dict() for hit in self.hits],
            "refused": self.refused,
            "message": self.message,
        }


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
