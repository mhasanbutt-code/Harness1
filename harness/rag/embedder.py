"""Text embeddings for RAG.

Backend priority (``config.embed_backend``):

* ``ollama`` — POST ``/api/embeddings`` with ``config.embed_model`` (e.g.
  ``nomic-embed-text``). Real semantic embeddings, on the box running Ollama.
* ``hash``   — a deterministic signed hashing-trick bag-of-words vector. No
  deps, no network; gives meaningful *lexical* similarity so retrieval and
  tests work fully offline.

``auto`` uses Ollama if a daemon and the embed model are reachable, else hash.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from harness.config import HarnessConfig

_WORD = re.compile(r"\w+")


class Embedder:
    """Embed text into fixed-dimension, L2-normalized vectors."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()
        self.dim = self.config.embed_dim
        self._backend: str | None = None

    @property
    def backend(self) -> str:
        if self._backend is None:
            self._resolve()
        return self._backend  # type: ignore[return-value]

    def _resolve(self) -> None:
        pref = self.config.embed_backend
        if pref in ("auto", "ollama"):
            try:
                from harness.llm.ollama import ping

                if ping(self.config.ollama_host):
                    probe = self._ollama_embed_one("ping")
                    if probe is not None:
                        self.dim = probe.shape[0]
                        self._backend = "ollama"
                        return
            except Exception:
                pass
            if pref == "ollama":
                self._backend = "hash"
                return
        self._backend = "hash"

    # ------------------------------------------------------------------ #
    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Return a ``(N, dim)`` float32 matrix of L2-normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "ollama":
            vecs = []
            for t in texts:
                v = self._ollama_embed_one(t)
                vecs.append(v if v is not None else self._hash_one(t))
            return np.stack(vecs).astype(np.float32)
        return np.stack([self._hash_one(t) for t in texts]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _ollama_embed_one(self, text: str) -> np.ndarray | None:
        try:
            from harness.llm.ollama import post

            out = post(
                self.config.ollama_host,
                "/api/embeddings",
                {"model": self.config.embed_model, "prompt": text},
                timeout=30.0,
            )
            emb = out.get("embedding")
            if not emb:
                return None
            v = np.asarray(emb, dtype=np.float32)
            return v / (np.linalg.norm(v) or 1.0)
        except Exception:
            return None

    def _hash_one(self, text: str) -> np.ndarray:
        """Signed hashing-trick embedding: stable, offline, lexical."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0  # signed -> fewer collisions
            vec[idx] += sign
        return vec / (np.linalg.norm(vec) or 1.0)
