from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2-7B-Instruct LoRA 微调入口")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct", help="基础模型名称或本地路径")
    parser.add_argument("--data", default="data/finetune/qwen_lora_train.jsonl", help="训练数据路径")
    parser.add_argument("--output", default="outputs/qwen2_lora_campus", help="LoRA 输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError("请先在 meng 环境安装 transformers、datasets、peft、accelerate。") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=args.data)["train"]

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        result = tokenizer(text, truncation=True, max_length=2048)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_steps=200,
        fp16=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
