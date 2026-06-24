from __future__ import annotations

import re


PAGE_NO_RE = re.compile(r"^\s*[-—]?\s*\d+\s*[-—]?\s*$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
EMPTY_LINES_RE = re.compile(r"\n{3,}")
CLAUSE_RE = re.compile(r"(第[一二三四五六七八九十百千万零〇\d]+条|\b\d+(?:\.\d+)+\b)")


def normalize_text(text: str) -> str:
    """清洗常见空白和孤立页码，保留章节标题与条款编号。"""

    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = SPACE_RE.sub(" ", raw_line).strip()
        if not line or PAGE_NO_RE.match(line):
            continue
        lines.append(line)
    return EMPTY_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def extract_clause_numbers(text: str) -> list[str]:
    """提取片段中的条款编号，供前端展示和证据引用使用。"""

    seen: set[str] = set()
    result: list[str] = []
    for match in CLAUSE_RE.finditer(text):
        value = match.group(1)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
