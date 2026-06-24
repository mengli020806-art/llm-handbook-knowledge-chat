from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import load_config
from .document_loader import SUPPORTED_SUFFIXES, list_documents
from .pipeline import CampusRagPipeline
from .schema import ensure_dir


cfg = load_config()
pipeline = CampusRagPipeline(cfg)

app = FastAPI(title="校园规章智能问答系统", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _html_page()


@app.get("/health")
def health() -> dict[str, str | bool]:
    store_ready = (cfg.index_dir / "chunks.jsonl").exists() and (cfg.index_dir / "vectors.npy").exists()
    return {"status": "ok", "index_ready": store_ready}


@app.get("/status")
def status() -> dict[str, object]:
    ensure_dir(cfg.raw_dir)
    docs = list_documents(cfg.raw_dir)
    store_ready = (cfg.index_dir / "chunks.jsonl").exists() and (cfg.index_dir / "vectors.npy").exists()
    return {
        "status": "ok",
        "index_ready": store_ready,
        "document_count": len(docs),
        "llm_backend": cfg.llm_backend,
        "llm_model_name": cfg.llm_model_name,
        "embedding_backend": cfg.embedding_backend,
        "retrieval_top_k": cfg.recall_top_k,
        "raw_dir": str(cfg.raw_dir.relative_to(cfg.project_root)),
    }


@app.get("/admin/files")
def files() -> dict[str, object]:
    ensure_dir(cfg.raw_dir)
    docs = list_documents(cfg.raw_dir)
    return {"count": len(docs), "files": [str(path.relative_to(cfg.project_root)) for path in docs]}


@app.post("/admin/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    ensure_dir(cfg.raw_dir)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式：{suffix}")
    target = cfg.raw_dir / Path(file.filename or "knowledge_file").name
    content = await file.read()
    target.write_bytes(content)
    return {"message": "上传成功，请重新构建索引。", "path": str(target.relative_to(cfg.project_root))}


@app.post("/admin/build-index")
def build_index() -> dict[str, object]:
    return pipeline.build_index()


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, object]:
    return pipeline.ask(request.question).to_dict()


def _html_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>校园规章智能问答系统</title>
  <style>
    :root {
      color-scheme: light;
      --bg: linear-gradient(180deg, #f2efe6 0%, #f7f6f2 38%, #eef3f1 100%);
      --panel: rgba(255, 255, 255, 0.88);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --ink: #1d2528;
      --muted: #617074;
      --line: rgba(41, 68, 70, 0.12);
      --line-strong: rgba(31, 66, 66, 0.18);
      --primary: #1c6a60;
      --primary-strong: #124b44;
      --primary-soft: #e5f1ee;
      --accent: #b9682e;
      --accent-soft: #f5ebe1;
      --shadow: 0 24px 60px rgba(26, 42, 38, 0.08);
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", system-ui, sans-serif;
    }
    .shell {
      width: min(1360px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 28px;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.95fr);
      gap: 18px;
      align-items: stretch;
    }
    .hero, .overview {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 24px;
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(circle at top right, rgba(28, 106, 96, 0.14), transparent 28%),
        radial-gradient(circle at left bottom, rgba(185, 104, 46, 0.10), transparent 26%),
        var(--panel-strong);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      padding: 0 12px;
      border-radius: 999px;
      background: var(--primary-soft);
      color: var(--primary-strong);
      font-size: 13px;
      font-weight: 700;
    }
    .eyebrow::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--primary);
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.18;
      letter-spacing: 0;
    }
    .hero-copy {
      max-width: 760px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.8;
    }
    .hero-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .stat {
      min-height: 94px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.72);
    }
    .stat-label {
      color: var(--muted);
      font-size: 13px;
    }
    .stat-value {
      margin-top: 10px;
      font-size: 24px;
      font-weight: 700;
    }
    .overview {
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .overview-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .overview-title {
      font-size: 18px;
      font-weight: 700;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.78);
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--accent);
    }
    .overview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.72);
    }
    .metric-name {
      color: var(--muted);
      font-size: 12px;
    }
    .metric-value {
      margin-top: 7px;
      font-size: 14px;
      line-height: 1.55;
      word-break: break-word;
    }
    .grid {
      display: grid;
      grid-template-columns: 350px minmax(0, 1fr);
      gap: 18px;
      padding-top: 18px;
    }
    aside, main {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
    }
    aside {
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 16px;
    }
    main {
      min-height: 720px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
    }
    h2, h3 {
      letter-spacing: 0;
      margin: 0;
    }
    h2 {
      font-size: 18px;
    }
    h3 {
      font-size: 15px;
    }
    .panel-copy {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }
    .actions {
      display: grid;
      gap: 10px;
    }
    .dropzone {
      display: grid;
      gap: 10px;
      padding: 14px;
      border: 1px dashed var(--line-strong);
      border-radius: var(--radius);
      background: rgba(249, 251, 250, 0.82);
    }
    .dropzone-note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    input[type="file"] {
      width: 100%;
      padding: 10px;
      border: 1px dashed var(--line);
      border-radius: 6px;
      background: #fbfcfd;
    }
    button {
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--primary);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      transition: transform 180ms ease, background 180ms ease, opacity 180ms ease;
    }
    button:hover { background: var(--primary-strong); transform: translateY(-1px); }
    button:disabled { opacity: 0.64; cursor: wait; transform: none; }
    button.secondary {
      background: var(--primary-soft);
      color: var(--primary-strong);
      border: 1px solid #c8ded8;
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .notice {
      min-height: 44px;
      padding: 12px 14px;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .notice.warn {
      border-color: rgba(185, 104, 46, 0.22);
      background: var(--accent-soft);
      color: #7c4b22;
    }
    .notice.ok {
      border-color: rgba(28, 106, 96, 0.18);
      background: var(--primary-soft);
      color: var(--primary-strong);
    }
    .files-panel {
      display: grid;
      gap: 10px;
    }
    .file-list {
      max-height: 280px;
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 4px;
    }
    .file-item {
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.72);
      font-size: 13px;
      line-height: 1.55;
      word-break: break-all;
    }
    .workspace-path {
      padding: 10px 12px;
      border-radius: 6px;
      background: rgba(20, 75, 68, 0.06);
      color: var(--primary-strong);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-all;
    }
    .qa-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px 12px;
      border-bottom: 1px solid var(--line);
    }
    .qa-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .pill {
      min-height: 30px;
      display: inline-flex;
      align-items: center;
      padding: 0 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.75);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      padding: 14px 18px 0;
    }
    .summary-card {
      min-height: 74px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.72);
    }
    .summary-card strong {
      display: block;
      margin-top: 8px;
      font-size: 18px;
    }
    .file-list {
      color: var(--muted);
    }
    .messages {
      padding: 18px;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .welcome {
      padding: 18px;
      border: 1px dashed var(--line-strong);
      border-radius: var(--radius);
      background: rgba(255, 255, 255, 0.56);
      color: var(--muted);
      line-height: 1.8;
    }
    .bubble {
      display: grid;
      gap: 10px;
      max-width: min(900px, 100%);
      padding: 14px 16px;
      border-radius: var(--radius);
      line-height: 1.7;
      white-space: pre-wrap;
    }
    .user {
      justify-self: end;
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-strong) 100%);
      color: white;
      box-shadow: 0 14px 30px rgba(28, 106, 96, 0.14);
    }
    .bot {
      justify-self: start;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid var(--line);
    }
    .bubble-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
      color: inherit;
      opacity: 0.92;
    }
    .bot .bubble-head {
      color: var(--muted);
    }
    .answer-sections {
      display: grid;
      gap: 10px;
    }
    .answer-block {
      padding: 12px 13px;
      border-radius: 6px;
      background: rgba(28, 106, 96, 0.05);
      border: 1px solid rgba(28, 106, 96, 0.08);
    }
    .answer-block strong {
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
    }
    .evidence-list {
      display: grid;
      gap: 10px;
    }
    .evidence-item {
      padding: 12px 13px;
      border-radius: 6px;
      background: rgba(17, 41, 43, 0.04);
      border: 1px solid var(--line);
    }
    .evidence-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .score {
      color: var(--primary-strong);
      font-weight: 700;
    }
    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.6);
    }
    textarea {
      width: 100%;
      min-height: 56px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: rgba(255, 255, 255, 0.92);
    }
    .send-col {
      display: grid;
      align-content: stretch;
      gap: 8px;
      min-width: 128px;
    }
    .send-hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      text-align: center;
    }
    .loading {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .loading::before {
      content: "";
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid rgba(28, 106, 96, 0.18);
      border-top-color: var(--primary);
      animation: spin 0.8s linear infinite;
    }
    .hidden { display: none !important; }
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    @media (max-width: 1080px) {
      .topbar, .grid {
        grid-template-columns: 1fr;
      }
      .hero-stats, .summary-strip {
        grid-template-columns: 1fr;
      }
      main {
        min-height: 680px;
      }
    }
    @media (max-width: 820px) {
      .shell {
        width: min(100vw - 20px, 100%);
      }
      .overview-grid, .button-row {
        grid-template-columns: 1fr;
      }
      .composer {
        grid-template-columns: 1fr;
      }
      .send-col {
        min-width: 0;
      }
      .qa-toolbar {
        align-items: flex-start;
        flex-direction: column;
      }
      .bubble {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="topbar">
      <div class="hero">
        <div class="eyebrow">校园制度知识库问答台</div>
        <h1 style="margin-top:14px;">校园规章智能问答系统</h1>
        <div class="hero-copy">
          面向图片版学生手册与制度文件的问答工作台。系统会先完成 OCR、结构化切分和检索重排，再调用本地大模型生成“结论、依据条款、注意事项”三段式答案。
        </div>
        <div class="hero-stats">
          <div class="stat">
            <div class="stat-label">知识库文件</div>
            <div class="stat-value" id="hero-doc-count">0</div>
          </div>
          <div class="stat">
            <div class="stat-label">检索配置</div>
            <div class="stat-value" id="hero-topk">Top 5</div>
          </div>
          <div class="stat">
            <div class="stat-label">问答引擎</div>
            <div class="stat-value" id="hero-llm">正在检测</div>
          </div>
        </div>
      </div>
      <div class="overview">
        <div class="overview-head">
          <div class="overview-title">运行状态</div>
          <div class="status"><span class="dot" id="dot"></span><span id="status">正在检测索引</span></div>
        </div>
        <div class="overview-grid">
          <div class="metric">
            <div class="metric-name">大模型后端</div>
            <div class="metric-value" id="metric-llm-backend">检测中</div>
          </div>
          <div class="metric">
            <div class="metric-name">大模型名称</div>
            <div class="metric-value" id="metric-llm-model">检测中</div>
          </div>
          <div class="metric">
            <div class="metric-name">嵌入方式</div>
            <div class="metric-value" id="metric-embedding">检测中</div>
          </div>
          <div class="metric">
            <div class="metric-name">知识库目录</div>
            <div class="metric-value" id="metric-raw-dir">检测中</div>
          </div>
        </div>
      </div>
    </section>
    <div class="grid">
      <aside>
        <div>
          <h2>知识库管理</h2>
          <div class="panel-copy" style="margin-top:8px;">
            支持上传图片、PDF、Word 和文本文件。知识库更新后需要重新构建索引，前端会直接显示当前文件清单和建库反馈。
          </div>
        </div>
        <div class="dropzone">
          <h3>上传资料</h3>
          <div class="dropzone-note">适合放入学生手册页面截图、规章扫描图和结构化文本文件。</div>
          <input id="file" type="file" accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp" />
          <div class="actions">
            <button id="upload">上传文件</button>
            <div class="button-row">
              <button class="secondary" id="build">构建索引</button>
              <button class="secondary" id="refresh">刷新列表</button>
            </div>
          </div>
        </div>
        <div id="notice" class="notice">等待操作。</div>
        <div class="workspace-path">
          当前知识库目录：<span id="raw-dir-path">data/raw</span>
        </div>
        <div class="files-panel">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
            <h3>已上传文件</h3>
            <span class="pill" id="file-count-pill">0 个文件</span>
          </div>
          <div class="file-list" id="files"></div>
        </div>
      </aside>
      <main>
        <section class="qa-toolbar">
          <div>
            <h2>问答工作台</h2>
            <div class="panel-copy" style="margin-top:6px;">提问时会先召回相关制度片段，再交给本地大模型生成最终回答。</div>
          </div>
          <div class="qa-meta">
            <span class="pill" id="qa-pill-index">索引状态检测中</span>
            <span class="pill" id="qa-pill-model">模型检测中</span>
          </div>
        </section>
        <section class="summary-strip">
          <div class="summary-card">
            当前知识库
            <strong id="summary-documents">0 份文件</strong>
          </div>
          <div class="summary-card">
            当前嵌入
            <strong id="summary-embedding">检测中</strong>
          </div>
          <div class="summary-card">
            当前生成模型
            <strong id="summary-llm">检测中</strong>
          </div>
        </section>
        <section class="messages" id="messages">
          <div class="welcome">
            在左侧上传图片或文档并构建索引后，就可以在这里直接提问。回答区会同时显示最终结论和命中的制度依据片段，方便你拿去做项目演示或论文截图。
          </div>
        </section>
        <section class="composer">
          <textarea id="question" placeholder="请输入校园规章问题"></textarea>
          <div class="send-col">
            <button id="send">提问</button>
            <div class="send-hint">按 Enter 发送，Shift + Enter 换行</div>
            <div id="ask-loading" class="loading hidden">正在检索并生成回答</div>
          </div>
        </section>
      </main>
    </div>
  </div>
  <script>
    const statusEl = document.querySelector("#status");
    const dotEl = document.querySelector("#dot");
    const messagesEl = document.querySelector("#messages");
    const filesEl = document.querySelector("#files");
    const questionEl = document.querySelector("#question");
    const noticeEl = document.querySelector("#notice");
    const askLoadingEl = document.querySelector("#ask-loading");
    const rawDirPathEl = document.querySelector("#raw-dir-path");
    const fileCountPillEl = document.querySelector("#file-count-pill");
    const heroDocCountEl = document.querySelector("#hero-doc-count");
    const heroTopkEl = document.querySelector("#hero-topk");
    const heroLlmEl = document.querySelector("#hero-llm");
    const metricLlmBackendEl = document.querySelector("#metric-llm-backend");
    const metricLlmModelEl = document.querySelector("#metric-llm-model");
    const metricEmbeddingEl = document.querySelector("#metric-embedding");
    const metricRawDirEl = document.querySelector("#metric-raw-dir");
    const qaPillIndexEl = document.querySelector("#qa-pill-index");
    const qaPillModelEl = document.querySelector("#qa-pill-model");
    const summaryDocumentsEl = document.querySelector("#summary-documents");
    const summaryEmbeddingEl = document.querySelector("#summary-embedding");
    const summaryLlmEl = document.querySelector("#summary-llm");
    const uploadBtn = document.querySelector("#upload");
    const buildBtn = document.querySelector("#build");
    const refreshBtn = document.querySelector("#refresh");
    const sendBtn = document.querySelector("#send");

    function setNotice(text, type = "") {
      noticeEl.className = "notice" + (type ? ` ${type}` : "");
      noticeEl.textContent = text;
    }

    function addMessage(text, type) {
      const node = document.createElement("div");
      node.className = `bubble ${type}`;
      const head = document.createElement("div");
      head.className = "bubble-head";
      const role = document.createElement("span");
      role.textContent = type === "user" ? "用户问题" : "系统回答";
      const time = document.createElement("span");
      time.textContent = new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
      head.appendChild(role);
      head.appendChild(time);

      const content = document.createElement("div");
      content.textContent = text;
      node.appendChild(head);
      node.appendChild(content);
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addAnswerCard(data) {
      const node = document.createElement("div");
      node.className = "bubble bot";

      const head = document.createElement("div");
      head.className = "bubble-head";
      const role = document.createElement("span");
      role.textContent = data.refused ? "系统回答｜未命中足够依据" : "系统回答｜已生成";
      const meta = document.createElement("span");
      meta.textContent = `命中 ${data.hits?.length || 0} 条片段`;
      head.appendChild(role);
      head.appendChild(meta);

      const sections = document.createElement("div");
      sections.className = "answer-sections";
      const parts = splitAnswerSections(data.answer || "");
      if (parts.length) {
        for (const part of parts) {
          const block = document.createElement("div");
          block.className = "answer-block";
          const title = document.createElement("strong");
          title.textContent = part.title;
          const body = document.createElement("div");
          body.textContent = part.body;
          block.appendChild(title);
          block.appendChild(body);
          sections.appendChild(block);
        }
      } else {
        const fallback = document.createElement("div");
        fallback.textContent = data.answer || "未获得回答。";
        sections.appendChild(fallback);
      }

      node.appendChild(head);
      node.appendChild(sections);

      if (Array.isArray(data.hits) && data.hits.length) {
        const evidenceList = document.createElement("div");
        evidenceList.className = "evidence-list";
        data.hits.forEach((hit, index) => {
          const item = document.createElement("div");
          item.className = "evidence-item";
          const metaLine = document.createElement("div");
          metaLine.className = "evidence-meta";
          metaLine.innerHTML = `
            <span class="score">片段 ${index + 1}｜分数 ${Number(hit.final_score || hit.similarity || 0).toFixed(3)}</span>
            <span>来源 ${hit.chunk?.source || "未知"}</span>
            <span>章节 ${hit.chunk?.chapter_title || "未知"}</span>
            <span>页码 ${hit.chunk?.page_start || "未知"}</span>
          `;
          const text = document.createElement("div");
          text.textContent = hit.chunk?.text || "";
          item.appendChild(metaLine);
          item.appendChild(text);
          evidenceList.appendChild(item);
        });
        node.appendChild(evidenceList);
      }

      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function splitAnswerSections(text) {
      const normalized = String(text || "").replace(/\\r/g, "");
      const titles = ["结论：", "依据条款：", "注意事项："];
      const positions = titles
        .map((title) => ({ title: title.replace("：", ""), index: normalized.indexOf(title) }))
        .filter((item) => item.index >= 0)
        .sort((a, b) => a.index - b.index);

      if (!positions.length) {
        return [];
      }

      return positions.map((item, idx) => {
        const start = item.index + item.title.length + 1;
        const end = idx + 1 < positions.length ? positions[idx + 1].index : normalized.length;
        return {
          title: item.title,
          body: normalized.slice(start, end).trim()
        };
      });
    }

    async function refreshStatus() {
      const res = await fetch("/status");
      const data = await res.json();
      const indexReady = Boolean(data.index_ready);
      statusEl.textContent = indexReady ? "索引已就绪" : "索引未构建";
      dotEl.style.background = indexReady ? "#1c6a60" : "#b9682e";
      qaPillIndexEl.textContent = indexReady ? "索引已构建" : "索引待构建";
      qaPillModelEl.textContent = `${data.llm_backend} / ${data.llm_model_name}`;

      metricLlmBackendEl.textContent = data.llm_backend || "未知";
      metricLlmModelEl.textContent = data.llm_model_name || "未知";
      metricEmbeddingEl.textContent = data.embedding_backend || "未知";
      metricRawDirEl.textContent = data.raw_dir || "data/raw";

      rawDirPathEl.textContent = data.raw_dir || "data/raw";
      heroDocCountEl.textContent = String(data.document_count || 0);
      heroTopkEl.textContent = `Top ${data.retrieval_top_k || 5}`;
      heroLlmEl.textContent = data.llm_model_name || "未知";
      summaryDocumentsEl.textContent = `${data.document_count || 0} 份文件`;
      summaryEmbeddingEl.textContent = data.embedding_backend || "未知";
      summaryLlmEl.textContent = data.llm_model_name || "未知";
    }

    async function refreshFiles() {
      const res = await fetch("/admin/files");
      const data = await res.json();
      filesEl.innerHTML = "";
      fileCountPillEl.textContent = `${data.count || 0} 个文件`;
      if (!data.files.length) {
        const empty = document.createElement("div");
        empty.className = "file-item";
        empty.textContent = "暂无文件";
        filesEl.appendChild(empty);
        return;
      }
      for (const file of data.files) {
        const item = document.createElement("div");
        item.className = "file-item";
        item.textContent = file;
        filesEl.appendChild(item);
      }
    }

    async function refreshAll() {
      await Promise.all([refreshStatus(), refreshFiles()]);
    }

    uploadBtn.addEventListener("click", async () => {
      const file = document.querySelector("#file").files[0];
      if (!file) return;
      uploadBtn.disabled = true;
      setNotice(`正在上传：${file.name}`, "warn");
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await fetch("/admin/upload", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) {
          setNotice(data.detail || "上传失败。", "warn");
          return;
        }
        setNotice(data.message || "上传成功，请重新构建索引。", "ok");
        addMessage(`已上传文件：${file.name}`, "user");
        await refreshAll();
      } catch (error) {
        setNotice(`上传失败：${error.message}`, "warn");
      } finally {
        uploadBtn.disabled = false;
      }
    });

    buildBtn.addEventListener("click", async () => {
      buildBtn.disabled = true;
      refreshBtn.disabled = true;
      setNotice("正在构建索引，请稍候。", "warn");
      try {
        const res = await fetch("/admin/build-index", { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          setNotice(data.detail || "索引构建失败。", "warn");
          return;
        }
        setNotice(
          `索引构建完成。文件 ${data.documents || 0} 份，页/段 ${data.pages || 0}，片段 ${data.chunks || 0}。`,
          "ok"
        );
        addMessage(
          `索引构建完成：文件 ${data.documents || 0} 份，片段 ${data.chunks || 0}，嵌入 ${data.embedding || "未知"}。`,
          "bot"
        );
        await refreshAll();
      } catch (error) {
        setNotice(`索引构建失败：${error.message}`, "warn");
      } finally {
        buildBtn.disabled = false;
        refreshBtn.disabled = false;
      }
    });

    refreshBtn.addEventListener("click", async () => {
      refreshBtn.disabled = true;
      try {
        await refreshAll();
        setNotice("文件列表与系统状态已刷新。", "ok");
      } finally {
        refreshBtn.disabled = false;
      }
    });

    sendBtn.addEventListener("click", async () => {
      const question = questionEl.value.trim();
      if (!question) return;
      addMessage(question, "user");
      questionEl.value = "";
      sendBtn.disabled = true;
      askLoadingEl.classList.remove("hidden");
      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        if (!res.ok) {
          setNotice(data.detail || "提问失败。", "warn");
          addMessage("提问失败，请稍后重试。", "bot");
          return;
        }
        addAnswerCard(data);
        setNotice(data.refused ? "本次问题未命中足够依据，系统已拒答。" : "问答完成，可查看证据片段。", data.refused ? "warn" : "ok");
      } catch (error) {
        setNotice(`提问失败：${error.message}`, "warn");
        addMessage("请求未成功完成，请检查本地服务状态。", "bot");
      } finally {
        sendBtn.disabled = false;
        askLoadingEl.classList.add("hidden");
      }
    });

    questionEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendBtn.click();
      }
    });

    refreshAll();
  </script>
</body>
</html>
"""
