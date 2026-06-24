from campus_rag.document_loader import SUPPORTED_SUFFIXES
from campus_rag.schema import DocumentPage
from campus_rag.splitter import split_pages


def test_split_pages_keeps_chapter_and_clause() -> None:
    pages = [
        DocumentPage(
            source="学生手册.txt",
            page=12,
            text="第一章 学籍管理\n第十条 学生因病请假，应当履行请假手续。请假期满应及时销假。",
        )
    ]

    chunks = split_pages(pages, max_chars=200, overlap_chars=20)

    assert len(chunks) == 1
    assert chunks[0].chapter_title == "第一章 学籍管理"
    assert chunks[0].page_start == 12
    assert "第十条" in chunks[0].clause_numbers


def test_supported_formats_include_images() -> None:
    assert ".png" in SUPPORTED_SUFFIXES
    assert ".jpg" in SUPPORTED_SUFFIXES
    assert ".pdf" in SUPPORTED_SUFFIXES
