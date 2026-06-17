"""RAG: retrieve project-knowledge chunks and answer with the local LLM.

This is how you give the model knowledge of the harness/JEPA stack WITHOUT
fine-tuning: ingest the docs once, then every question retrieves the most
relevant chunks and grounds the model's answer in them. Facts stay current
(re-ingest when docs change) and the model cites its sources.

Pairs naturally with the Ollama backend: ingest with the ``hash`` or
``nomic-embed-text`` embedder, answer with your ``qwen3`` model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harness.config import HarnessConfig
from harness.llm.model import LocalLLM
from harness.rag.embedder import Embedder
from harness.rag.ingest import make_chunks
from harness.rag.store import VectorStore

RAG_PROMPT = """You answer questions about this project using ONLY the context below.
If the context does not contain the answer, say you don't know — do not invent details.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class Answer:
    answer: str
    sources: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


class RAG:
    """Ingest documents and answer questions grounded in them."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self.embedder = embedder or Embedder(self.config)
        self.store = store or VectorStore()
        self.llm = llm or LocalLLM(self.config)

    # ------------------------------------------------------------------ #
    def ingest(self, paths: list[str | Path]) -> int:
        """Chunk, embed, and index documents. Returns the chunk count added."""
        chunks = make_chunks(paths, self.config.rag_chunk_size, self.config.rag_chunk_overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        return len(chunks)

    def save(self) -> Path:
        self.store.save(self.config.rag_store_path)
        return self.config.rag_store_path

    def load(self) -> "RAG":
        self.store = VectorStore.load(self.config.rag_store_path)
        return self

    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, k: int | None = None):
        k = k or self.config.rag_top_k
        q = self.embedder.embed(question)[0]
        return self.store.search(q, k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k)
        contexts = [c.text for c, _ in hits]
        sources = [c.source for c, _ in hits]
        if contexts:
            context_block = "\n\n---\n\n".join(
                f"[{i + 1}] (from {s})\n{t}" for i, (t, s) in enumerate(zip(contexts, sources))
            )
        else:
            context_block = "(no documents have been ingested)"
        prompt = RAG_PROMPT.format(context=context_block, question=question)
        answer = self.llm.generate(prompt, temperature=0.2)
        return Answer(answer=answer, sources=sources, contexts=contexts)
