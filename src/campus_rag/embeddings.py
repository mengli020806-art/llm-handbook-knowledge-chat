from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """嵌入模型统一接口。"""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        ...

    def save(self, index_dir: Path) -> None:
        ...


class SentenceBertEmbedding:
    """Sentence-BERT 嵌入模型。"""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - 环境缺包时触发
            raise RuntimeError("当前环境未安装 sentence-transformers。") from exc

        model_kwargs = {}
        if device != "auto":
            model_kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **model_kwargs)
        self.name = f"sentence-transformers:{model_name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def save(self, index_dir: Path) -> None:
        meta_path = index_dir / "embedding_backend.txt"
        meta_path.write_text(self.name, encoding="utf-8")


class TfidfEmbedding:
    """无需下载模型的本地回退嵌入方案，用于开发测试和演示。"""

    def __init__(self, vectorizer=None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = vectorizer or TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.name = "tfidf-char-ngram"
        self._fitted = vectorizer is not None

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        matrix = self.vectorizer.transform(texts)
        dense = matrix.astype(np.float32).toarray()
        return _normalize(dense)

    def save(self, index_dir: Path) -> None:
        with (index_dir / "tfidf_vectorizer.pkl").open("wb") as file:
            pickle.dump(self.vectorizer, file)
        (index_dir / "embedding_backend.txt").write_text(self.name, encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> "TfidfEmbedding":
        with (index_dir / "tfidf_vectorizer.pkl").open("rb") as file:
            vectorizer = pickle.load(file)
        return cls(vectorizer=vectorizer)


def build_embedding_model(
    backend: str,
    model_name: str,
    device: str,
    index_dir: Path | None = None,
    for_query: bool = False,
) -> EmbeddingModel:
    """根据配置创建嵌入模型。"""

    backend = backend.lower()
    if for_query and index_dir and (index_dir / "tfidf_vectorizer.pkl").exists():
        return TfidfEmbedding.load(index_dir)

    if backend in {"sentence-bert", "sentence_transformers", "sbert"}:
        return SentenceBertEmbedding(model_name=model_name, device=device)
    if backend == "tfidf":
        return TfidfEmbedding()
    if backend == "auto":
        try:
            return SentenceBertEmbedding(model_name=model_name, device=device)
        except RuntimeError:
            return TfidfEmbedding()
    raise ValueError(f"不支持的嵌入后端：{backend}")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return vectors / norm
