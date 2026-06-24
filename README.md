# 校园规章智能问答系统

本项目用于构建“河南理工大学学生手册”等校园制度知识库，支持文档清洗、章节优先切分、向量索引、相似度召回、二次重排和三段式问答输出。

## 当前流程

1. 将知识库文件放入 `data/raw`，支持 `txt`、`md`、`pdf`、`docx` 和常见图片格式。
2. 执行索引构建，系统会读取文件、清洗文本、保留章节标题、条款编号和页码信息。
3. 系统将制度片段转为向量，并保存索引到 `data/index`。
4. 用户提问时，系统会对问题做向量化，召回 Top 5 片段，再结合相似度、关键词命中率和条款完整度重排。
5. 命中依据足够时调用本地大语言模型生成“结论、依据条款、注意事项”；依据不足时拒答。

## 图片知识库

图片会先经过 OCR 识别，再进入清洗、切分和建索引流程。当前代码支持 `rapidocr-onnxruntime`、`paddleocr` 或 `pytesseract`。如果环境里没有 OCR 引擎，构建索引时会给出明确提示。

推荐先在 `meng` 环境中安装轻量 OCR：

```powershell
conda activate meng
python -m pip install rapidocr-onnxruntime
```

## 运行方式

安装项目依赖：

```powershell
conda activate meng
python -m pip install -r requirements.txt
python -m pip install -e .
```

初始化目录：

```powershell
campus-rag init-dirs
```

构建索引：

```powershell
campus-rag build
```

命令行提问：

```powershell
campus-rag ask "学生因病请假需要办理什么手续？"
```

启动接口和网页：

```powershell
uvicorn campus_rag.api:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

## 大模型调用

当前默认使用本地 Ollama 调用 `qwen2.5:1.5b`，适合 4GB 显存机器先跑通真实大模型生成。由于 GTX 1650 在当前 Ollama 版本上可能遇到 CUDA 内核兼容问题，配置里默认设置了 `num_gpu: 0`，即使用本地 CPU 跑量化模型：

```powershell
ollama pull qwen2.5:1.5b
ollama serve
```

如果后续换成更大显存机器，可以把 `configs/config.yaml` 改为：

```yaml
llm:
  backend: qwen
  model_name: Qwen/Qwen2-7B-Instruct
```

## 嵌入模型说明

当前默认将 `embedding.backend` 设为 `tfidf`，这是因为本机访问 Hugging Face 超时，`Sentence-BERT` 还没有缓存到本地。这样系统依然可以完成“检索 + 重排 + 本地大模型生成”全流程。

如果后续网络畅通并成功下载嵌入模型，可以切回：

```yaml
embedding:
  backend: auto
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## LoRA 微调数据

将原始问答样本保存为 `data/finetune/qa_samples.jsonl`，每行格式如下：

```json
{"question":"学生因病请假怎么办？","evidence":"第十条 学生因病请假，应当履行请假手续。","answer":"结论：学生因病请假应履行请假手续。\n依据条款：第十条。\n注意事项：请假期满应及时销假。"}
```

整理为 Qwen2 对话微调格式：

```powershell
python scripts/prepare_lora_dataset.py
```

启动 LoRA 微调前，请确认显存和依赖满足要求：

```powershell
python scripts/train_lora_qwen2.py
```

## 重要说明

知识库上传或更新后需要重新构建索引。每次提问不会重新切分全部文档，也不会重新计算所有制度片段的向量；提问阶段只会计算用户问题向量，然后检索、重排和生成回答。
