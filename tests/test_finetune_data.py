import json
from pathlib import Path

from campus_rag.finetune_data import convert_qa_jsonl


def test_convert_qa_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "qa.jsonl"
    target = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "考试违纪如何处理？",
                "evidence": "第十二条 考试违纪按规定处理。",
                "answer": "结论：按规定处理。\n依据条款：第十二条。\n注意事项：以正式文件为准。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    count = convert_qa_jsonl(source, target)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

    assert count == 1
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][2]["role"] == "assistant"
