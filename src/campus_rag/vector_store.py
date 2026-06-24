from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import EmbeddingModel
from .schema import SearchHit, TextChunk, ensure_dir


class VectorStore:
    """向量索引封装，优先使用 FAISS，缺包时回退为 NumPy 检索。"""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunks: list[TextChunk] = []
        self.vectors: np.ndarray | None = None
        self.faiss_index = None
        self.backend = "numpy"

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and (self.vectors is not None or self.faiss_index is not None)

    def build(self, chunks: list[TextChunk], embedding_model: EmbeddingModel) -> None:
        if not chunks:
            raise ValueError("没有可索引的制度片段，请先放入知识库文档。")

        self.chunks = chunks
        self.vectors = embedding_model.encode([chunk.text for chunk in chunks]).astype(np.float32)
        self.faiss_index = _try_build_faiss(self.vectors)
        self.backend = "faiss" if self.faiss_index is not None else "numpy"

    def save(self, embedding_model: EmbeddingModel) -> None:
        ensure_dir(self.index_dir)
        with (self.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        if self.vectors is not None:
            np.save(self.index_dir / "vectors.npy", self.vectors)

        if self.faiss_index is not None:
            try:
                import faiss

                faiss.write_index(self.faiss_index, str(self.index_dir / "index.faiss"))
            except Exception:
                pass

        embedding_model.save(self.index_dir)
        (self.index_dir / "store_backend.txt").write_text(self.backend, encoding="utf-8")

    def load(self) -> None:
        chunks_path = self.index_dir / "chunks.jsonl"
        vectors_path = self.index_dir / "vectors.npy"
        if not chunks_path.exists() or not vectors_path.exists():
            self.chunks = []
            self.vectors = None
            self.faiss_index = None
            return

        self.chunks = []
        with chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.chunks.append(TextChunk.from_dict(json.loads(line)))

        self.vectors = np.load(vectors_path).astype(np.float32)
        faiss_path = self.index_dir / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss

                self.faiss_index = faiss.read_index(str(faiss_path))
                self.backend = "faiss"
                return
            except Exception:
                self.faiss_index = None
        self.backend = "numpy"

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if not self.is_ready:
            return []

        query = query_vector.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query, top_k)
            return [
                SearchHit(chunk=self.chunks[int(index)], similarity=float(score))
                for score, index in zip(scores[0], indices[0])
                if int(index) >= 0
            ]

        assert self.vectors is not None
        scores = (self.vectors @ query[0]).astype(float)
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(chunk=self.chunks[int(index)], similarity=float(scores[index])) for index in order]


def _try_build_faiss(vectors: np.ndarray):
    try:
        import faiss
    except Exception:
        return None

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index
