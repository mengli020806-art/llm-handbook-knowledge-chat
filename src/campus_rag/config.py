from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AppConfig:
    """系统运行配置。"""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    index_dir: Path
    finetune_dir: Path
    max_chars: int
    overlap_chars: int
    recall_top_k: int
    min_score: float
    similarity_weight: float
    keyword_weight: float
    completeness_weight: float
    embedding_backend: str
    embedding_model_name: str
    embedding_device: str
    llm_backend: str
    llm_model_name: str
    lora_adapter: str | None
    max_new_tokens: int
    temperature: float
    ollama_num_gpu: int | None = None


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """读取配置文件，并允许环境变量覆盖关键模型设置。"""

    env_path = os.getenv("CAMPUS_RAG_CONFIG")
    selected_path = Path(config_path or env_path or PROJECT_ROOT / "configs" / "config.yaml")
    if not selected_path.is_absolute():
        selected_path = PROJECT_ROOT / selected_path

    with selected_path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}

    paths = data.get("paths", {})
    splitter = data.get("splitter", {})
    retrieval = data.get("retrieval", {})
    embedding = data.get("embedding", {})
    llm = data.get("llm", {})

    llm_backend = os.getenv("CAMPUS_RAG_LLM_BACKEND", llm.get("backend", "extractive"))
    llm_model = os.getenv("CAMPUS_RAG_QWEN_MODEL", llm.get("model_name", "Qwen/Qwen2-7B-Instruct"))
    lora_adapter = os.getenv("CAMPUS_RAG_LORA_ADAPTER", llm.get("lora_adapter"))
    lora_adapter = lora_adapter or None
    ollama_num_gpu_value = os.getenv("CAMPUS_RAG_OLLAMA_NUM_GPU", llm.get("num_gpu"))
    ollama_num_gpu = int(ollama_num_gpu_value) if ollama_num_gpu_value not in {None, ""} else None

    return AppConfig(
        project_root=PROJECT_ROOT,
        raw_dir=_resolve_path(PROJECT_ROOT, paths.get("raw_dir", "data/raw")),
        processed_dir=_resolve_path(PROJECT_ROOT, paths.get("processed_dir", "data/processed")),
        index_dir=_resolve_path(PROJECT_ROOT, paths.get("index_dir", "data/index")),
        finetune_dir=_resolve_path(PROJECT_ROOT, paths.get("finetune_dir", "data/finetune")),
        max_chars=int(splitter.get("max_chars", 900)),
        overlap_chars=int(splitter.get("overlap_chars", 120)),
        recall_top_k=int(retrieval.get("recall_top_k", 5)),
        min_score=float(retrieval.get("min_score", 0.18)),
        similarity_weight=float(retrieval.get("similarity_weight", 0.65)),
        keyword_weight=float(retrieval.get("keyword_weight", 0.25)),
        completeness_weight=float(retrieval.get("completeness_weight", 0.10)),
        embedding_backend=str(embedding.get("backend", "auto")),
        embedding_model_name=str(
            embedding.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        ),
        embedding_device=str(embedding.get("device", "auto")),
        llm_backend=str(llm_backend),
        llm_model_name=str(llm_model),
        lora_adapter=lora_adapter,
        max_new_tokens=int(llm.get("max_new_tokens", 512)),
        temperature=float(llm.get("temperature", 0.2)),
        ollama_num_gpu=ollama_num_gpu,
    )
