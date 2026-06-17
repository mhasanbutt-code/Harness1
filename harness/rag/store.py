"""A tiny on-disk vector store: a numpy matrix plus cosine search.

Rows are stored L2-normalized so a normalized query dot-product *is* cosine
similarity. Small enough to read in full; swap for FAISS/Chroma when the corpus
outgrows memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    source: str


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (N, dim), rows L2-normalized

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        self._chunks.extend(chunks)
        self._matrix = embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])

    def search(self, query_emb: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._matrix @ q
        k = min(k, len(self._chunks))
        idx = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(path / "matrix.npy", matrix)
        (path / "chunks.json").write_text(json.dumps([asdict(c) for c in self._chunks], indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        store = cls()
        matrix = np.load(path / "matrix.npy")
        store._matrix = matrix if matrix.size else None
        store._chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text())]
        return store
