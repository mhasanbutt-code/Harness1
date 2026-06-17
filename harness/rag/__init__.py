"""Retrieval-augmented generation over project knowledge.

Teach the local model your harness/JEPA stack WITHOUT fine-tuning: ingest the
docs once, then every question retrieves the most relevant chunks and grounds
the model's answer in them.
"""

from harness.rag.embedder import Embedder
from harness.rag.pipeline import RAG, Answer
from harness.rag.store import Chunk, VectorStore

__all__ = ["Embedder", "RAG", "Answer", "Chunk", "VectorStore"]
