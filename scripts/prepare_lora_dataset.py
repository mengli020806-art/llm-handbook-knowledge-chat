from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_rag.finetune_data import convert_qa_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 Qwen2 LoRA 微调数据")
    parser.add_argument("--input", default="data/finetune/qa_samples.jsonl", help="原始问答样本路径")
    parser.add_argument("--output", default="data/finetune/qwen_lora_train.jsonl", help="输出训练数据路径")
    args = parser.parse_args()

    count = convert_qa_jsonl(Path(args.input), Path(args.output))
    print(f"已生成 {count} 条训练样本：{args.output}")


if __name__ == "__main__":
    main()
