from __future__ import annotations

import json
import re
from pathlib import Path

from .config import AppConfig, load_config
from .document_loader import SUPPORTED_SUFFIXES, load_documents
from .embeddings import build_embedding_model
from .generator import AnswerGenerator
from .reranker import rerank_hits
from .schema import AnswerResult, SearchHit, ensure_dir
from .splitter import split_pages
from .vector_store import VectorStore


class CampusRagPipeline:
    """校园规章 RAG 主流程。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.store = VectorStore(self.config.index_dir)
        self.generator = AnswerGenerator(self.config)

    def build_index(self) -> dict[str, int | str]:
        ensure_dir(self.config.raw_dir)
        ensure_dir(self.config.processed_dir)
        ensure_dir(self.config.index_dir)

        pages = load_documents(self.config.raw_dir)
        if not pages:
            formats = "、".join(sorted(SUPPORTED_SUFFIXES))
            return {
                "documents": 0,
                "pages": 0,
                "chunks": 0,
                "message": f"未发现知识库文件，请将 {formats} 文件放入 {self.config.raw_dir}",
            }

        chunks = split_pages(
            pages,
            max_chars=self.config.max_chars,
            overlap_chars=self.config.overlap_chars,
        )
        if not chunks:
            return {"documents": len({page.source for page in pages}), "pages": len(pages), "chunks": 0, "message": "文档读取成功，但未切分出有效片段。"}

        processed_path = self.config.processed_dir / "chunks.jsonl"
        with processed_path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        embedding_model = build_embedding_model(
            backend=self.config.embedding_backend,
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
        )
        self.store.build(chunks, embedding_model)
        self.store.save(embedding_model)

        return {
            "documents": len({page.source for page in pages}),
            "pages": len(pages),
            "chunks": len(chunks),
            "embedding": embedding_model.name,
            "index_backend": self.store.backend,
            "message": "索引构建完成。",
        }

    def ask(self, question: str) -> AnswerResult:
        question = question.strip()
        if not question:
            return AnswerResult(question=question, answer="请输入有效问题。", hits=[], refused=True, message="问题为空。")

        self.store.load()
        if not self.store.is_ready:
            answer = self.generator.generate(question, [])
            return AnswerResult(question=question, answer=answer, hits=[], refused=True, message="知识库索引尚未构建。")

        embedding_model = build_embedding_model(
            backend=self.config.embedding_backend,
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
            index_dir=self.config.index_dir,
            for_query=True,
        )
        sub_questions = _split_question_for_retrieval(question)
        if len(sub_questions) > 1:
            sub_answer = _answer_sub_questions(
                sub_questions=sub_questions,
                store=self.store,
                embedding_model=embedding_model,
                generator=self.generator,
                recall_top_k=self.config.recall_top_k,
                min_score=self.config.min_score,
                similarity_weight=self.config.similarity_weight,
                keyword_weight=self.config.keyword_weight,
                completeness_weight=self.config.completeness_weight,
            )
            if sub_answer is not None:
                answer, focused_hits = sub_answer
                return AnswerResult(question=question, answer=answer, hits=focused_hits, refused=False)

        query_vector = embedding_model.encode([question])
        recall_hits = self.store.search(query_vector, top_k=self.config.recall_top_k)
        ranked_hits = rerank_hits(
            question=question,
            hits=recall_hits,
            similarity_weight=self.config.similarity_weight,
            keyword_weight=self.config.keyword_weight,
            completeness_weight=self.config.completeness_weight,
        )
        ranked_hits = _filter_conflicting_hits(question, ranked_hits)
        ranked_hits = _filter_conflicting_topics(question, ranked_hits)
        usable_hits = [hit for hit in ranked_hits if hit.final_score >= self.config.min_score]

        if not usable_hits:
            answer = self.generator.generate(question, [])
            return AnswerResult(question=question, answer=answer, hits=ranked_hits, refused=True, message="没有达到阈值的制度片段。")

        usable_hits = _focus_hits_on_subject(question, usable_hits)
        answer = self.generator.generate(question, usable_hits)
        return AnswerResult(question=question, answer=answer, hits=usable_hits, refused=False)


def get_pipeline(config_path: str | Path | None = None) -> CampusRagPipeline:
    return CampusRagPipeline(load_config(config_path))


SUBJECT_GROUPS = [
    ("学术学位硕士研究生",),
    ("专业学位硕士",),
    ("学术学位研究生",),
    ("专业学位研究生",),
    ("博士生", "博士研究生"),
    ("硕士生", "硕士研究生"),
]

TOPIC_GROUPS = [
    ("预审", "专家", "工作日", "学位论文外审"),
    ("考核小组", "小组成员", "专家小组"),
    ("学术活动", "学术报告", "学术会议"),
    ("开题", "开题报告"),
    ("中期考核",),
    ("查阅文献", "文献综述"),
    ("请假", "销假"),
]
SECTION_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+、.{1,20}$")


def _filter_conflicting_hits(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    """过滤与问题主体明显冲突的片段，减少大模型被错误片段带偏。"""

    target_group = None
    for group in SUBJECT_GROUPS:
        if any(term in question for term in group):
            target_group = group
            break

    if not target_group:
        return hits

    matched: list[SearchHit] = []
    neutral: list[SearchHit] = []
    for hit in hits:
        text = hit.chunk.text
        if any(term in text for term in target_group):
            matched.append(hit)
            continue

        conflict = False
        for group in SUBJECT_GROUPS:
            if group == target_group:
                continue
            if any(term in text for term in group):
                conflict = True
                break
        if not conflict:
            neutral.append(hit)

    if matched:
        return matched + neutral
    return hits


def _focus_hits_on_subject(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    """当片段内混有多个主体或主题时，仅保留与问题最相关的局部上下文。"""

    target_group = None
    for group in SUBJECT_GROUPS:
        if any(term in question for term in group):
            target_group = group
            break

    topic_group = None
    for group in TOPIC_GROUPS:
        if any(term in question for term in group):
            topic_group = group
            break

    if not target_group and not topic_group:
        return hits

    for hit in hits:
        lines = [line.strip() for line in hit.chunk.text.splitlines() if line.strip()]
        if not lines:
            continue
        matched_index = None
        if topic_group:
            matched_index = _find_topic_line_index(lines, topic_group)
        if matched_index is None and target_group:
            matched_index = next(
                (index for index, line in enumerate(lines) if any(term in line for term in target_group)),
                None,
            )
        if matched_index is None:
            continue

        focused_lines: list[str] = []
        if matched_index > 0:
            previous_line = lines[matched_index - 1]
            if (
                len(previous_line) <= 30
                or "第" in previous_line
                or "计划" in previous_line
                or (topic_group and any(term in previous_line for term in topic_group))
            ):
                focused_lines.append(previous_line)

        focused_lines.append(lines[matched_index])

        for line in lines[matched_index + 1 :]:
            if focused_lines and _is_new_section_heading(line, topic_group):
                break
            if target_group and _is_conflicting_subject_line(line, target_group):
                break
            focused_lines.append(line)
            if len(focused_lines) >= 12:
                break

        hit.chunk.text = "\n".join(focused_lines)
    return hits


def _split_question_for_retrieval(question: str) -> list[str]:
    """将复合精确题拆成多个检索子问题。"""

    compact = question.strip("？?。！! ")
    if not compact:
        return []

    if not any(token in compact for token in ["几", "多少", "不少于", "不低于", "不超过", "一般为"]):
        return [compact]

    separators = ["？", "?", "。", "；", ";", "，以及", "以及", "，或", "或", "并且", "并", "和", "、", "，"]
    parts = [compact]
    for separator in separators:
        new_parts: list[str] = []
        for part in parts:
            split_parts = [item.strip() for item in part.split(separator) if item.strip()]
            if len(split_parts) > 1:
                new_parts.extend(split_parts)
            else:
                new_parts.append(part)
        parts = new_parts

    result = [part for part in parts if any(token in part for token in ["几", "多少", "不少于", "不低于", "不超过", "一般为"])]
    if not result:
        return [compact]

    deduped: list[str] = []
    for part in result:
        if part not in deduped:
            deduped.append(part)
    return deduped


def _answer_sub_questions(
    sub_questions: list[str],
    store: VectorStore,
    embedding_model,
    generator,
    recall_top_k: int,
    min_score: float,
    similarity_weight: float,
    keyword_weight: float,
    completeness_weight: float,
) -> tuple[str, list[SearchHit]] | None:
    """对每个子问题单独抽取并合并答案。"""

    combined_answers: list[str] = []
    combined_hits: list[SearchHit] = []
    used_chunk_ids: set[str] = set()

    for sub_question in sub_questions:
        query_vector = embedding_model.encode([sub_question])
        recall_hits = store.search(query_vector, top_k=recall_top_k)
        ranked_hits = rerank_hits(
            question=sub_question,
            hits=recall_hits,
            similarity_weight=similarity_weight,
            keyword_weight=keyword_weight,
            completeness_weight=completeness_weight,
        )
        ranked_hits = _filter_conflicting_hits(sub_question, ranked_hits)
        ranked_hits = _filter_conflicting_topics(sub_question, ranked_hits)
        target_group = _infer_topic_group(sub_question)
        sub_hits = _select_hits_for_sub_question(sub_question, ranked_hits, target_group)
        sub_hits = [hit for hit in sub_hits if hit.final_score >= min_score]
        if not sub_hits:
            return None
        sub_hits = _focus_hits_on_subject(sub_question, sub_hits)
        sub_answer = generator.generate(sub_question, sub_hits)
        if "暂未在当前知识库中找到" in sub_answer:
            return None
        combined_answers.append(_extract_answer_body(sub_answer))
        for hit in sub_hits:
            if hit.chunk.chunk_id in used_chunk_ids:
                continue
            used_chunk_ids.add(hit.chunk.chunk_id)
            combined_hits.append(hit)

    if not combined_answers:
        return None
    answer = "，".join(item for item in combined_answers if item)
    if not answer:
        return None
    return answer, combined_hits


def _select_hits_for_sub_question(
    question: str,
    ranked_hits: list[SearchHit],
    target_group: tuple[str, ...] | None,
) -> list[SearchHit]:
    if not target_group:
        return ranked_hits

    matched: list[SearchHit] = []
    neutral: list[SearchHit] = []
    for hit in ranked_hits:
        text = f"{hit.chunk.chapter_title or ''}\n{hit.chunk.text}"
        if any(term in text for term in target_group):
            matched.append(hit)
            continue
        if not _is_conflicting_topic_text(text, target_group):
            neutral.append(hit)

    if matched:
        return matched + neutral
    return ranked_hits


def _infer_topic_group(question: str) -> tuple[str, ...] | None:
    for group in TOPIC_GROUPS:
        if any(term in question for term in group):
            return group
    if any(token in question for token in ["几人", "多少人", "人数", "考核小组", "小组成员", "专家小组"]):
        return ("考核小组", "小组成员", "专家小组")
    if any(token in question for token in ["几位", "多少位", "专家", "预审", "工作日", "学位论文外审"]):
        return ("预审", "专家", "工作日", "学位论文外审")
    if any(token in question for token in ["几次", "次数", "学术活动", "学术报告", "学术会议"]):
        return ("学术活动", "学术报告", "学术会议")
    if any(token in question for token in ["几篇", "多少篇", "文献"]):
        return ("查阅文献", "文献综述")
    if any(token in question for token in ["几个月", "多久", "月份", "月"]):
        return ("课程学习计划",)
    return None


def _is_conflicting_topic_text(text: str, target_group: tuple[str, ...]) -> bool:
    for group in TOPIC_GROUPS:
        if group == target_group:
            continue
        if any(term in text for term in group):
            return True
    return False


def _extract_answer_body(answer: str) -> str:
    for prefix in ("结论：", "结论:"):
        if prefix in answer:
            body = answer.split(prefix, 1)[1]
            return body.split("\n", 1)[0].strip()
    return answer.strip()


def _find_topic_line_index(lines: list[str], topic_group: tuple[str, ...]) -> int | None:
    """优先定位章节标题，其次才定位正文中的主题词。"""

    heading_index = next(
        (
            index
            for index, line in enumerate(lines)
            if SECTION_HEADING_RE.match(line) and any(term in line for term in topic_group)
        ),
        None,
    )
    if heading_index is not None:
        return heading_index

    return next(
        (index for index, line in enumerate(lines) if any(term in line for term in topic_group)),
        None,
    )


def _filter_conflicting_topics(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    """按问题主题过滤明显无关的片段，避免开题、中期考核等主题互相串扰。"""

    target_group = None
    for group in TOPIC_GROUPS:
        if any(term in question for term in group):
            target_group = group
            break

    if not target_group:
        return hits

    matched: list[SearchHit] = []
    neutral: list[SearchHit] = []
    for hit in hits:
        text = f"{hit.chunk.chapter_title or ''}\n{hit.chunk.text}"
        if any(term in text for term in target_group):
            matched.append(hit)
            continue

        conflict = False
        for group in TOPIC_GROUPS:
            if group == target_group:
                continue
            if any(term in text for term in group):
                conflict = True
                break
        if not conflict:
            neutral.append(hit)

    if matched:
        return matched + neutral
    return hits


def _is_conflicting_subject_line(line: str, target_group: tuple[str, ...]) -> bool:
    if any(term in line for term in target_group):
        return False
    for group in SUBJECT_GROUPS:
        if group == target_group:
            continue
        if any(term in line for term in group):
            return True
    return False


def _is_conflicting_topic_line(line: str, target_group: tuple[str, ...], focused_lines: list[str]) -> bool:
    for group in TOPIC_GROUPS:
        if group == target_group:
            continue
        if any(term in line for term in group):
            # 主题片段刚开始时允许保留上一主题的尾句，真正进入目标主题后再截断。
            if not any(any(term in kept for term in target_group) for kept in focused_lines):
                return False
            return True
    return False


def _is_new_section_heading(line: str, target_group: tuple[str, ...]) -> bool:
    return bool(SECTION_HEADING_RE.match(line)) and not any(term in line for term in target_group)
