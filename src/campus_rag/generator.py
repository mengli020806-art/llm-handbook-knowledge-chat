from __future__ import annotations

import json
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import AppConfig
from .prompts import build_prompt
from .schema import SearchHit


class AnswerGenerator:
    """回答生成器，支持抽取式回答和 Qwen2 模型回答。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.backend = config.llm_backend.lower()
        self._pipeline = None

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        if _should_force_extractive(question, hits):
            return self._generate_precise_extractive(question, hits)
        if self.backend == "ollama":
            return self._generate_with_ollama(question, hits)
        if self.backend in {"qwen", "transformers"}:
            return self._generate_with_qwen(question, hits)
        return self._generate_extractive(question, hits)

    def _generate_extractive(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return _refusal_answer(question)

        best = hits[0]
        chunk = best.chunk
        clauses = "、".join(chunk.clause_numbers) if chunk.clause_numbers else chunk.chapter_title or "相关制度片段"
        page = f"第 {chunk.page_start} 页" if chunk.page_start else "页码未标注"
        source = f"{chunk.source}，{page}"
        note = "本回答仅依据当前知识库中命中的制度片段生成；如制度文件更新，请重新构建索引后再查询。"

        return (
            f"结论：根据当前知识库，问题“{question}”可参考命中的校园制度片段处理。\n"
            f"依据条款：{clauses}。来源：{source}。原文依据：{chunk.text}\n"
            f"注意事项：{note}"
        )

    def _generate_precise_extractive(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return _refusal_answer(question)

        best = hits[0]
        chunk = best.chunk
        page = f"第 {chunk.page_start} 页" if chunk.page_start else "页码未标注"
        source = f"{chunk.source}，{page}"
        sub_questions = _split_precise_question(question)
        if not sub_questions:
            sub_questions = [question]

        answers: list[str] = []
        for sub_question in sub_questions:
            answer_sentence = _extract_precise_sentence(sub_question, chunk.text)
            if not answer_sentence:
                return _refusal_answer(question)
            answers.append(answer_sentence)

        answer_sentence = "，".join(answers)
        if not answer_sentence:
            return _refusal_answer(question)
        return (
            f"结论：{answer_sentence}\n"
            f"依据条款：来源：{source}。原文依据：{answer_sentence}\n"
            "注意事项：本题属于精确信息问答，系统优先采用命中片段中的原文数值，不对数字做自由改写。"
        )

    def _generate_with_qwen(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return _refusal_answer(question)
        try:
            pipeline = self._get_pipeline()
        except RuntimeError as exc:
            return _model_error_answer(question, str(exc))

        prompt = build_prompt(question, hits)
        result = pipeline(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
            return_full_text=False,
        )
        if isinstance(result, list) and result:
            text = result[0].get("generated_text", "").strip()
            if text:
                return text
        return self._generate_extractive(question, hits)

    def _generate_with_ollama(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return _refusal_answer(question)

        prompt = build_prompt(question, hits)
        payload = {
            "model": self.config.llm_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens,
            },
        }
        if self.config.ollama_num_gpu is not None:
            payload["options"]["num_gpu"] = self.config.ollama_num_gpu
        request = Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            return _model_error_answer(question, f"Ollama 服务未启动或无法连接：{exc}")
        except TimeoutError:
            return _model_error_answer(question, "Ollama 大模型生成超时。")
        except Exception as exc:
            return _model_error_answer(question, f"Ollama 大模型调用失败：{exc}")

        text = str(data.get("response", "")).strip()
        if not text:
            return _model_error_answer(question, "Ollama 未返回有效文本。")
        return text

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as exc:  # pragma: no cover - 环境缺包时触发
            raise RuntimeError("当前环境未安装 transformers，已回退到抽取式回答。") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if self.config.lora_adapter:
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.config.lora_adapter)
            except Exception as exc:  # pragma: no cover - 适配器路径错误时触发
                raise RuntimeError(f"LoRA 适配器加载失败：{self.config.lora_adapter}") from exc

        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._pipeline


def _refusal_answer(question: str) -> str:
    return (
        f"结论：暂未在当前知识库中找到与“{question}”足够匹配的制度依据，因此不能给出确定回答。\n"
        "依据条款：无可引用条款。当前索引为空，或召回片段相似度低于阈值。\n"
        "注意事项：请先将学生手册等制度文件放入 data/raw 目录并重新构建索引；涉及具体处分、学籍或奖助事项时，以学校最新正式文件为准。"
    )


def _model_error_answer(question: str, reason: str) -> str:
    return (
        f"结论：已检索到与“{question}”相关的制度依据，但大模型生成环节没有成功完成。\n"
        f"依据条款：本次未生成最终条款解释。原因：{reason}\n"
        "注意事项：请确认本地大模型服务已启动、模型名称正确，并且显存或内存足够；问题依据不会被编造。"
    )


PRECISE_QUESTION_RE = re.compile(r"(多少|几|不少于|不低于|不超过|几个月|几篇|几人|几分|几门|一般为)")


def _should_force_extractive(question: str, hits: list[SearchHit]) -> bool:
    if not hits:
        return False
    if PRECISE_QUESTION_RE.search(question):
        return True
    return False


def _extract_precise_sentence(question: str, text: str) -> str | None:
    numeric_statement = _extract_structured_numeric_statement(question, text)
    if numeric_statement:
        return numeric_statement

    pieces = re.split(r"(?<=[。；;！？?])", text.replace("\n", ""))
    candidates = [piece.strip() for piece in pieces if piece.strip()]
    keywords = [token for token in ["文献", "外文文献", "课程学习计划", "博士生", "硕士", "专业学位", "学术学位", "不少于", "不低于", "不超过"] if token in question]
    needs_people = any(token in question for token in ["几人", "多少人", "人数", "考核小组", "小组成员", "专家小组"])
    needs_expert = any(token in question for token in ["几位", "多少位", "位专家", "专家", "聘请"])
    needs_activity = any(token in question for token in ["几次", "次数", "学术活动", "学术报告", "学术会议", "学分"])
    needs_paper = any(token in question for token in ["几篇", "多少篇", "文献"])
    needs_month = any(token in question for token in ["几个月", "多久", "月份", "月"])
    needs_workday = any(token in question for token in ["几个工作日", "多少个工作日", "工作日", "预审时间"])
    needs_score = "学分" not in question and any(token in question for token in ["几分", "多少分", "分数", "成绩", "合格标准"])
    if needs_people:
        keywords.extend(["考核小组", "小组成员", "专家小组", "中期考核", "不少于", "至少", "人"])
    if needs_expert:
        keywords.extend(["专家", "预审", "聘请", "至少", "位"])
    if needs_activity:
        keywords.extend(["学术活动", "学术报告", "学术会议", "次数", "次", "学分"])
    if needs_paper:
        keywords.extend(["文献", "外文文献", "篇"])
    if needs_month:
        keywords.extend(["月", "时间", "期限", "个月"])
    if needs_workday:
        keywords.extend(["预审时间", "预审", "工作日", "不少于"])
    if needs_score:
        keywords.extend(["成绩", "平均", "不低于", "合格标准", "分"])
        for token in ["必修课", "学位课", "外语", "外国语", "课程学习"]:
            if token in question:
                keywords.append(token)

    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        score = 0
        if any(char.isdigit() for char in candidate) or any(token in candidate for token in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]):
            score += 2
        for keyword in keywords:
            if keyword in candidate:
                score += 3
        if needs_people:
            if "人" in candidate:
                score += 4
            if any(token in candidate for token in ["考核小组", "小组成员", "专家小组"]):
                score += 6
            if any(token in candidate for token in ["次", "篇", "学分"]) and "人" not in candidate:
                score -= 4
        if needs_expert:
            if "位专家" in candidate:
                score += 6
            if any(token in candidate for token in ["专家", "预审", "聘请"]):
                score += 4
        if needs_activity:
            if "次" in candidate:
                score += 4
            if any(token in candidate for token in ["学术活动", "学术报告", "学术会议"]):
                score += 4
            if "人" in candidate and "次" not in candidate:
                score -= 2
        if needs_paper:
            if "篇" in candidate:
                score += 5
            if "外文文献" in candidate:
                score += 3
        if needs_month:
            if "月" in candidate:
                score += 5
            if any(token in candidate for token in ["个月", "月份"]):
                score += 3
        if needs_workday:
            if "工作日" in candidate:
                score += 6
            if "预审" in candidate:
                score += 3
        if needs_score:
            if "分" in candidate:
                score += 5
            if "成绩" in candidate:
                score += 4
            if "平均" in question and "平均" in candidate:
                score += 3
            for token in ["必修课", "学位课", "外语", "外国语", "课程学习"]:
                if token in question and token in candidate:
                    score += 4
            if "学分" in candidate:
                score -= 5
        if "外文文献" in question and "外文文献" in candidate:
            score += 3
        if "文献" in question and "文献" in candidate:
            score += 2
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None

    best_score, best_candidate = scored[0]
    if best_score < 7:
        return None
    if needs_people and "人" not in best_candidate:
        return None
    if needs_expert and "位专家" not in best_candidate:
        return None
    if needs_activity and "次" not in best_candidate and "学分" not in best_candidate:
        return None
    if needs_paper and "篇" not in best_candidate:
        return None
    if needs_month and "月" not in best_candidate:
        return None
    if needs_workday and "工作日" not in best_candidate:
        return None
    if needs_score and "分" not in best_candidate:
        return None
    return best_candidate


def _split_precise_question(question: str) -> list[str]:
    """把复合精确题拆成多个子问题，便于分别抽取数值。"""

    compact = question.strip("？?。！! ")
    if not compact:
        return []

    separators = [
        "？",
        "?",
        "。",
        "，或",
        "或",
        "以及",
        "和",
        "并",
        "，",
    ]
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

    normalized: list[str] = []
    for part in parts:
        if any(token in part for token in ["几", "多少", "不少于", "不低于", "不超过", "一般为"]):
            normalized.append(part)

    expert_parts: list[str] = []
    workday_parts: list[str] = []
    other_parts: list[str] = []
    for part in normalized:
        if any(token in part for token in ["几个工作日", "工作日", "预审时间"]):
            workday_parts.append(part)
            continue
        if any(token in part for token in ["几位", "位专家", "专家", "聘请"]):
            expert_parts.append(part)
            continue
        other_parts.append(part)

    ordered = workday_parts + expert_parts + other_parts
    deduped: list[str] = []
    for part in ordered:
        if part not in deduped:
            deduped.append(part)
    return deduped


def _question_type(question: str) -> str:
    if any(token in question for token in ["几人", "多少人", "人数", "考核小组", "小组成员", "专家小组"]):
        return "people"
    if any(token in question for token in ["几个工作日", "多少个工作日", "工作日", "预审时间"]):
        return "workday"
    if any(token in question for token in ["几位", "多少位", "位专家", "专家", "聘请"]):
        return "expert"
    if any(token in question for token in ["几次", "次数", "学术活动", "学术报告", "学术会议"]):
        return "activity"
    if any(token in question for token in ["几篇", "多少篇", "文献"]):
        return "paper"
    if any(token in question for token in ["几个月", "多久", "月份", "月"]):
        return "month"
    if "学分" in question:
        return "credit"
    if any(token in question for token in ["几分", "多少分", "分数", "成绩", "合格标准"]):
        return "score"
    return "other"


def _extract_structured_numeric_statement(question: str, text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    subject = _infer_subject(question, compact)

    if any(keyword in question for keyword in ["几个工作日", "多少个工作日", "工作日", "预审时间"]):
        workday_statement = _extract_workday_statement(question, compact, subject)
        if workday_statement:
            return workday_statement

    if any(keyword in question for keyword in ["几位", "多少位", "位专家", "专家", "聘请"]):
        expert_statement = _extract_expert_count_statement(question, compact, subject)
        if expert_statement:
            return expert_statement

    if any(keyword in question for keyword in ["几人", "多少人", "人数", "考核小组", "小组成员", "专家小组"]):
        committee_statement = _extract_committee_size_statement(question, compact, subject)
        if committee_statement:
            return committee_statement

    if "学分" not in question and any(keyword in question for keyword in ["几分", "多少分", "分数", "成绩", "合格标准"]):
        score_statement = _extract_score_statement(question, compact, subject)
        if score_statement:
            return score_statement

    if any(keyword in question for keyword in ["学术活动", "学术报告", "学术会议", "学分", "主讲次数"]):
        academic_statement = _extract_academic_activity_statement(question, compact, subject)
        if academic_statement:
            return academic_statement

    if "查阅文献" in question:
        total_match = re.search(r"查阅文献一般不少于([0-9一二三四五六七八九十百两]+)篇", compact)
        foreign_match = re.search(r"外文文献.{0,80}?一般不少于([0-9一二三四五六七八九十百两]+)篇", compact)
        if total_match and foreign_match:
            prefix = subject or "查阅文献要求"
            return f"{prefix}查阅文献一般不少于{total_match.group(1)}篇，其中外文文献一般不少于{foreign_match.group(1)}篇。"
        if total_match:
            prefix = subject or "查阅文献要求"
            return f"{prefix}查阅文献一般不少于{total_match.group(1)}篇。"

    if "课程学习计划" in question or "几个月" in question or "多久" in question:
        month_match = re.search(
            r"(博士生|博士研究生|硕士生|硕士研究生|专业学位研究生|学术学位研究生)?.{0,24}?应在([0-9一二三四五六七八九十百两]+)个月内.{0,50}?课程学习计划",
            compact,
        )
        if month_match:
            prefix = month_match.group(1) or subject or "相关研究生"
            return f"{prefix}应在{month_match.group(2)}个月内按照学科（专业）培养方案制定课程学习计划。"

    return ""


def _extract_score_statement(question: str, compact: str, subject: str) -> str:
    """抽取成绩分数类问题的原文数值，避免和学分、篇数等数字混淆。"""

    if "学分" in question:
        return ""

    target_terms = [term for term in ["必修课", "学位课", "外语", "外国语", "课程学习", "成绩平均", "平均"] if term in question]
    patterns = [
        r"((?:必修课|学位课|外语学位课|外国语学位课)[^。；;！？?\n]{0,12}?成绩[^。；;！？?\n]{0,12}?(?:不低于|不少于|至少|达到|为)([0-9一二三四五六七八九十百两!！]+)分)",
        r"((?:必修课|学位课|外语学位课|外国语学位课|课程学习)?成绩[^。；;！？?\n]{0,12}?(?:平均)?(?:不低于|不少于|至少|达到|为)([0-9一二三四五六七八九十百两!！]+)分)",
        r"((?:不低于|不少于|至少|达到|为)([0-9一二三四五六七八九十百两!！]+)分)",
    ]

    scored: list[tuple[int, str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact):
            phrase = match.group(1).replace(match.group(2), _normalize_confusable_numeric(match.group(2)))
            if "学分" in phrase or "分制" in phrase:
                continue
            score = 0
            if "成绩" in phrase:
                score += 4
            if "平均" in question and "平均" in phrase:
                score += 4
            for term in target_terms:
                if term in phrase:
                    score += 5
            if "必修课" in question and "必修课" not in phrase:
                score -= 6
            if any(term in question for term in ["外语", "外国语"]) and not any(term in phrase for term in ["外语", "外国语"]):
                score -= 6
            scored.append((score, phrase, match.group(2)))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    _, phrase, _ = scored[0]
    return f"{phrase}。"


def _extract_expert_count_statement(question: str, compact: str, subject: str) -> str:
    """抽取专家数量类问题的原文数值，避免误用人数、次数、篇数等其他数字。"""

    if not any(keyword in question for keyword in ["几位", "多少位", "位专家", "专家", "聘请"]):
        return ""

    patterns = [
        r"((?:必须|应当|须)?聘请至少([0-9一二三四五六七八九十百两!！]+)位专家.{0,30}?预审)",
        r"((?:必须|应当|须)?.{0,12}?至少([0-9一二三四五六七八九十百两!！]+)位专家.{0,30}?预审)",
        r"(至少([0-9一二三四五六七八九十百两!！]+)位专家)",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if not match:
            continue
        phrase = match.group(1).replace(match.group(2), _normalize_confusable_numeric(match.group(2)))
        phrase = _trim_precise_phrase(phrase, ["预审", "位专家"], keep_after=0)
        if "位专家" in phrase:
            return f"{phrase}。"

    return ""


def _extract_workday_statement(question: str, compact: str, subject: str) -> str:
    """抽取工作日时长类问题的原文数值，避免把月份、次数等数字混入答案。"""

    if not any(keyword in question for keyword in ["几个工作日", "多少个工作日", "工作日", "预审时间"]):
        return ""

    patterns = [
        r"(预审时间.{0,12}?(?:不少于|不低于|至少|一般为)([0-9一二三四五六七八九十百两!！]+)个工作日)",
        r"(预审.{0,20}?时间.{0,12}?(?:不少于|不低于|至少|一般为)([0-9一二三四五六七八九十百两!！]+)个工作日)",
        r"((?:不少于|不低于|至少|一般为)([0-9一二三四五六七八九十百两!！]+)个工作日)",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if not match:
            continue
        phrase = match.group(1).replace(match.group(2), _normalize_confusable_numeric(match.group(2)))
        if "工作日" in phrase:
            return f"{phrase}。"

    return ""


def _extract_committee_size_statement(question: str, compact: str, subject: str) -> str:
    """抽取考核小组、专家小组等人数类问题的原文数值。"""

    if not any(keyword in question for keyword in ["几人", "多少人", "人数", "考核小组", "小组成员", "专家小组"]):
        return ""

    patterns = [
        r"(考核小组.{0,18}?(?:不少于|至少|一般为|应为)([0-9一二三四五六七八九十百两!！]+)人)",
        r"(小组成员.{0,18}?(?:不少于|至少|一般为|应为)([0-9一二三四五六七八九十百两!！]+)人)",
        r"(专家小组.{0,18}?(?:不少于|至少|一般为|应为)([0-9一二三四五六七八九十百两!！]+)人)",
        r"((?:不少于|至少|一般为|应为)([0-9一二三四五六七八九十百两!！]+)人)",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if not match:
            continue
        phrase = match.group(1).replace(match.group(2), _normalize_confusable_numeric(match.group(2)))
        if any(token in phrase for token in ["考核小组", "小组成员", "专家小组"]):
            return f"{phrase}。"
        if subject:
            return f"{subject}{phrase}。"
        return f"{phrase}。"

    return ""


def _trim_precise_phrase(text: str, stop_tokens: list[str], keep_after: int = 0) -> str:
    """把命中的原文短语截到关键数字后面的最小完整句段。"""

    if not text:
        return text

    end = len(text)
    for token in stop_tokens:
        index = text.find(token)
        if index == -1:
            continue
        candidate_end = index + len(token) + keep_after
        if candidate_end < end:
            end = candidate_end
    trimmed = text[:end]
    trimmed = trimmed.rstrip("，,；;：: ")  # 保留句号由调用方统一补齐
    return trimmed


def _extract_academic_activity_statement(question: str, compact: str, subject: str) -> str:
    if "学术学位硕士研究生" not in question and "学术学位硕士研究生" not in compact:
        return ""

    report_match = re.search(r"至少公开做学术报告([0-9一二三四五六七八九十百两!！]+)次", compact)
    report_bang_match = re.search(r"至少公开做学术报告[!！]", compact)
    conference_match = re.search(r"参加国内外学术会议([0-9一二三四五六七八九十百两!！]+)次", compact)
    credit_match = re.search(r"学术报告考核通过计([0-9一二三四五六七八九十百两!！]+)学分", compact)

    report_value = _normalize_confusable_numeric(report_match.group(1)) if report_match else ""
    if not report_value and report_bang_match:
        report_value = "1"
    conference_value = _normalize_confusable_numeric(conference_match.group(1)) if conference_match else ""
    credit_value = _normalize_confusable_numeric(credit_match.group(1)) if credit_match else ""

    if not any([report_value, conference_value, credit_value]):
        return ""

    prefix = subject or "学术学位硕士研究生"
    parts: list[str] = []
    if report_value:
        parts.append(f"至少公开做学术报告{report_value}次")
    if conference_value:
        parts.append(f"或参加国内外学术会议{conference_value}次")
    if credit_value:
        parts.append(f"学术报告考核通过计{credit_value}学分")

    return f"{prefix}{'，'.join(parts)}。"


def _normalize_confusable_numeric(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned

    confusable_map = {
        "!": "1",
        "！": "1",
        "I": "1",
        "l": "1",
        "|": "1",
    }
    cleaned = "".join(confusable_map.get(char, char) for char in cleaned)
    if cleaned in {"一", "壹"}:
        return "1"
    if cleaned in {"二", "两", "贰"}:
        return "2"
    return cleaned


def _infer_subject(question: str, text: str) -> str:
    subject_terms = [
        "专业学位硕士",
        "学术学位硕士研究生",
        "博士研究生",
        "博士生",
        "硕士研究生",
        "硕士生",
        "专业学位研究生",
        "学术学位研究生",
    ]
    for term in subject_terms:
        if term in question:
            return term
    for term in subject_terms:
        if term in text:
            return term
    return ""
