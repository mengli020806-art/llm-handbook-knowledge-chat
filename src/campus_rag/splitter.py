from __future__ import annotations

import hashlib
import re

from .cleaner import extract_clause_numbers, normalize_text
from .schema import DocumentPage, TextChunk


HEADING_RE = re.compile(
    r"^(第[一二三四五六七八九十百千万零〇\d]+[章节篇].{0,40}|[一二三四五六七八九十]+、.{2,50}|\d+[、.．]\s*.{2,50})$"
)
SENTENCE_RE = re.compile(r"(?<=[。！？；;])")


def is_heading(line: str) -> bool:
    """识别制度文档中的章节或小节标题。"""

    compact = line.strip()
    if len(compact) > 60:
        return False
    return bool(HEADING_RE.match(compact))


def split_pages(
    pages: list[DocumentPage],
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    """按章节优先、页码辅助的策略切分文本。"""

    chunks: list[TextChunk] = []
    current_chapter: str | None = None
    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue
        sections, current_chapter = _split_page_into_sections(text, current_chapter)
        for chapter_title, section_text in sections:
            for part in _split_long_text(section_text, max_chars=max_chars, overlap_chars=overlap_chars):
                chunks.append(_make_chunk(part, page, chapter_title))
    return chunks


def _split_page_into_sections(text: str, current_chapter: str | None) -> tuple[list[tuple[str | None, str]], str | None]:
    sections: list[tuple[str | None, str]] = []
    buffer: list[str] = []
    chapter = current_chapter

    for line in text.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            continue
        if is_heading(clean_line):
            if buffer:
                sections.append((chapter, "\n".join(buffer).strip()))
                buffer = []
            chapter = clean_line
            buffer.append(clean_line)
        else:
            buffer.append(clean_line)

    if buffer:
        sections.append((chapter, "\n".join(buffer).strip()))
    return sections, chapter


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [item.strip() for item in SENTENCE_RE.split(text) if item.strip()]
    if not sentences:
        sentences = [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    result: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                result.append(buffer.strip())
                buffer = _tail(buffer, overlap_chars)
            for start in range(0, len(sentence), max_chars):
                result.append(sentence[start : start + max_chars].strip())
            continue

        candidate = f"{buffer}{sentence}" if not buffer else f"{buffer}\n{sentence}"
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                result.append(buffer.strip())
                buffer = _tail(buffer, overlap_chars)
            buffer = f"{buffer}\n{sentence}".strip() if buffer else sentence

    if buffer:
        result.append(buffer.strip())
    return [item for item in result if item]


def _tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    return text[-overlap_chars:].strip()


def _make_chunk(text: str, page: DocumentPage, chapter_title: str | None) -> TextChunk:
    digest = hashlib.sha1(f"{page.source}|{page.page}|{chapter_title}|{text}".encode("utf-8")).hexdigest()[:16]
    return TextChunk(
        chunk_id=digest,
        text=text,
        source=page.source,
        page_start=page.page,
        page_end=page.page,
        chapter_title=chapter_title,
        clause_numbers=extract_clause_numbers(text),
    )
