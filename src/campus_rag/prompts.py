from __future__ import annotations

import re

from .schema import SearchHit


SYSTEM_PROMPT = """你是校园规章智能问答助手。你只能依据给定制度片段回答。
若片段中没有足够依据，请明确说明暂未在知识库中找到依据，不得编造。
回答必须使用“结论、依据条款、注意事项”三段式。
如果问题涉及时间、数字、分数、月份、学期、比例、期限等精确信息，必须严格采用命中片段中的原文表述，不得把其他片段中的数值混入当前回答。
如果多个片段出现互相冲突的主体或数值，优先采用与用户问题主体最一致、且表述最直接的片段，并在“依据条款”中明确指出来自哪个片段。"""


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    numeric_hint = _build_numeric_hint(question, hits)
    evidence = "\n\n".join(
        f"片段{i}\n来源：{hit.chunk.source}；页码：{hit.chunk.page_start or '未知'}；章节：{hit.chunk.chapter_title or '未知'}\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )
    prompt = f"{SYSTEM_PROMPT}\n\n用户问题：{question}\n"
    if numeric_hint:
        prompt += f"\n关键数值提示：{numeric_hint}\n"
    prompt += f"\n可用制度片段：\n{evidence}\n\n请输出三段式答案。"
    return prompt


def _build_numeric_hint(question: str, hits: list[SearchHit]) -> str:
    if not hits:
        return ""
    if not any(keyword in question for keyword in ["几个月", "多久", "多长时间", "期限", "多少", "几学期", "几分"]):
        return ""

    pattern = re.compile(r"(?:\d+|[一二三四五六七八九十两]+)(?:个)?(?:月|周|天|年|学期|学年|分)")
    values: list[str] = []
    for hit in hits[:3]:
        values.extend(pattern.findall(hit.chunk.text))
    unique_values: list[str] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return "命中片段中出现的候选精确信息为：" + "、".join(unique_values[:6]) if unique_values else ""
