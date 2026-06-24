from __future__ import annotations

from pathlib import Path

from .cleaner import normalize_text
from .ocr import IMAGE_SUFFIXES, extract_pages_from_image
from .schema import DocumentPage


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"} | IMAGE_SUFFIXES


def list_documents(input_dir: str | Path) -> list[Path]:
    """列出可导入的知识库文件。"""

    root = Path(input_dir)
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=lambda item: str(item).lower())


def load_document(path: str | Path) -> list[DocumentPage]:
    """按文件类型读取文档，页码无法识别时用空值表示。"""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [DocumentPage(source=file_path.name, page=None, text=normalize_text(text))]
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".docx":
        return _load_docx(file_path)
    if suffix in IMAGE_SUFFIXES:
        pages = extract_pages_from_image(file_path)
        return [
            DocumentPage(source=file_path.name, page=index, text=text)
            for index, text in enumerate(pages, start=1)
            if text.strip()
        ]
    return []


def load_documents(input_dir: str | Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for path in list_documents(input_dir):
        pages.extend(load_document(path))
    return [page for page in pages if page.text.strip()]


def _load_pdf(path: Path) -> list[DocumentPage]:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:  # pragma: no cover - 只在缺包时触发
        raise RuntimeError("读取 PDF 需要安装 PyPDF2。") from exc

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = normalize_text(text)
        if cleaned:
            pages.append(DocumentPage(source=path.name, page=index, text=cleaned))
    return pages


def _load_docx(path: Path) -> list[DocumentPage]:
    try:
        from docx import Document
    except Exception as exc:  # pragma: no cover - 只在缺包时触发
        raise RuntimeError("读取 Word 文档需要安装 python-docx。") from exc

    document = Document(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    text = normalize_text("\n".join(paragraphs))
    return [DocumentPage(source=path.name, page=None, text=text)] if text else []
