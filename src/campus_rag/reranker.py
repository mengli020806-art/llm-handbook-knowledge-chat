from __future__ import annotations

import math
import re

from .schema import SearchHit


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{1}|[A-Za-z0-9_]+")
CLAUSE_HINT_RE = re.compile(r"(第.+?条|不得|应当|可以|按照|根据|给予|申请|办理|处分|奖励|学籍|请假|考试|违纪)")
NUMERIC_RE = re.compile(r"(?:不少于|不低于|不超过|应在|一般为|一般不少于|一般不低于)?\s*(?:\d+|[一二三四五六七八九十两百千]+)(?:个)?(?:月|周|天|年|学期|学年|篇|人|门|次|分|字)")
SUBJECT_GROUPS = [
    ("博士生", "博士研究生"),
    ("硕士生", "硕士研究生"),
    ("专业学位研究生",),
    ("学术学位研究生",),
    ("本科生", "本科学生"),
]

TOPIC_HINTS = [
    (("学术活动", "学术报告", "学术会议"), ("学术活动", "学术报告", "学术会议"), ("开题", "中期考核")),
    (("开题", "开题报告"), ("开题", "开题报告"), ("学术活动", "学术会议")),
    (("中期考核",), ("中期考核",), ("开题", "学术活动")),
    (("查阅文献", "文献综述"), ("查阅文献", "文献综述"), ("开题", "学术活动")),
    (("请假", "销假"), ("请假", "销假"), ("开题", "学术活动", "中期考核")),
]


def rerank_hits(
    question: str,
    hits: list[SearchHit],
    similarity_weight: float,
    keyword_weight: float,
    completeness_weight: float,
) -> list[SearchHit]:
    """结合相似度、关键词命中率和条款完整度进行二次排序。"""

    if not hits:
        return []
    query_terms = _terms(question)
    for hit in hits:
        hit.keyword_score = _keyword_score(query_terms, hit.chunk.text)
        hit.completeness_score = _completeness_score(hit.chunk.text)
        hit.numeric_score = _numeric_score(question, hit.chunk.text)
        subject_bonus = _subject_consistency_score(question, hit.chunk.text)
        topic_bonus = _topic_consistency_score(question, hit.chunk.chapter_title or "", hit.chunk.text)
        hit.final_score = (
            similarity_weight * _clip01(hit.similarity)
            + keyword_weight * hit.keyword_score
            + completeness_weight * hit.completeness_score
            + 0.16 * hit.numeric_score
            + subject_bonus
            + topic_bonus
        )
    return sorted(hits, key=lambda item: item.final_score, reverse=True)


def _terms(text: str) -> set[str]:
    values = {match.group(0).lower() for match in TOKEN_RE.finditer(text)}
    return {value for value in values if value.strip()}


def _keyword_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = _terms(text)
    hit_count = len(query_terms & content_terms)
    return hit_count / max(len(query_terms), 1)


def _completeness_score(text: str) -> float:
    score = 0.0
    if CLAUSE_HINT_RE.search(text):
        score += 0.45
    if len(text) >= 120:
        score += 0.30
    else:
        score += 0.30 * (len(text) / 120)
    if "。" in text or "；" in text:
        score += 0.25
    return min(score, 1.0)


def _clip01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _subject_consistency_score(question: str, text: str) -> float:
    """对“博士生/硕士生/专业学位研究生”等主体一致性做额外约束。"""

    matched_group = None
    for group in SUBJECT_GROUPS:
        if any(term in question for term in group):
            matched_group = group
            break

    if not matched_group:
        return 0.0

    if any(term in text for term in matched_group):
        return 0.18

    for group in SUBJECT_GROUPS:
        if group == matched_group:
            continue
        if any(term in text for term in group):
            return -0.22
    return 0.0


def _numeric_score(question: str, text: str) -> float:
    if not any(keyword in question for keyword in ["多少", "几", "不少于", "不低于", "几个月", "几篇", "几人", "几分", "几门"]):
        return 0.0

    matches = NUMERIC_RE.findall(text)
    if not matches:
        return 0.0

    score = 0.25
    if any(token in question for token in ["文献", "篇"]) and "篇" in "".join(matches):
        score += 0.45
    if any(token in question for token in ["几人", "多少人", "人数", "考核小组", "小组成员", "专家小组"]) and "人" in "".join(matches):
        score += 0.45
        if any(token in text for token in ["考核小组", "小组成员", "专家小组"]):
            score += 0.25
    if any(token in question for token in ["月", "时间", "期限"]) and "月" in "".join(matches):
        score += 0.45
    if any(token in question for token in ["工作日", "预审时间"]) and "工作日" in text:
        score += 0.45
    if any(token in question for token in ["几位", "多少位", "专家", "预审"]) and "位专家" in text:
        score += 0.45
    if "外文文献" in question and "外文文献" in text:
        score += 0.20
    if "专业学位" in question and "专业学位" in text:
        score += 0.20
    return min(score, 1.0)


def _topic_consistency_score(question: str, chapter_title: str, text: str) -> float:
    """根据问题主题与章节标题的一致性做额外排序约束。"""

    question_text = f"{question}{chapter_title}{text[:80]}"
    for positive_terms, preferred_terms, conflict_terms in TOPIC_HINTS:
        if any(term in question_text for term in positive_terms):
            if any(term in chapter_title for term in preferred_terms):
                return 0.20
            if any(term in chapter_title for term in conflict_terms):
                return -0.16
    return 0.0
