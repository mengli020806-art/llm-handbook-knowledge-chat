from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_MESSAGE = (
    "你是校园规章智能问答助手。只能依据给定制度依据回答；没有依据时必须拒答。"
    "回答固定为“结论、依据条款、注意事项”三段式。"
)


def convert_qa_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    """将问答样本整理为 Qwen2 对话微调格式。"""

    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with source.open("r", encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            item: dict[str, Any] = json.loads(line)
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            if not question or not answer:
                continue

            user_content = f"用户问题：{question}"
            if evidence:
                user_content += f"\n制度依据：{evidence}"

            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ]
            }
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count
