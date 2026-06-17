import numpy as np

from harness.config import HarnessConfig
from harness.rag.embedder import Embedder
from harness.rag.ingest import chunk_text, make_chunks
from harness.rag.pipeline import RAG
from harness.rag.store import Chunk, VectorStore


def _cfg(tmp_path):
    return HarnessConfig(
        embed_backend="hash",
        embed_dim=256,
        llm_backend="echo",
        rag_store_path=tmp_path / "store",
    )


def test_embedder_deterministic_and_shaped():
    e = Embedder(HarnessConfig(embed_backend="hash", embed_dim=128))
    a = e.embed("the harness uses jepa")
    assert a.shape == (1, 128)
    assert np.allclose(a, e.embed("the harness uses jepa"))


def test_embedder_lexical_similarity():
    e = Embedder(HarnessConfig(embed_backend="hash", embed_dim=512))
    v = e.embed(["jepa encoder perception", "jepa encoder perception layer", "billing invoice tax"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_store_orders_by_similarity():
    store = VectorStore()
    store.add(
        [Chunk("a", "x", "s"), Chunk("b", "y", "s"), Chunk("c", "z", "s")],
        np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32),
    )
    hits = store.search(np.array([1, 0, 0], dtype=np.float32), k=2)
    assert hits[0][0].id == "a"
    assert hits[1][0].id == "c"


def test_store_roundtrip(tmp_path):
    store = VectorStore()
    store.add([Chunk("a", "hello", "s")], np.array([[1, 0, 0]], dtype=np.float32))
    store.save(tmp_path / "s")
    loaded = VectorStore.load(tmp_path / "s")
    assert len(loaded) == 1
    assert loaded.search(np.array([1, 0, 0], dtype=np.float32), k=1)[0][0].text == "hello"


def test_chunking_bounds():
    chunks = chunk_text("para line.\n" * 60, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_make_chunks_from_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("JEPA encoder turns inputs into embeddings.")
    (d / "b.txt").write_text("The agent loop is ReAct style.")
    chunks = make_chunks([str(d)])
    assert len(chunks) == 2
    assert {c.source.split("/")[-1] for c in chunks} == {"a.md", "b.txt"}


def test_rag_end_to_end(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "jepa.md").write_text("The JEPA encoder turns raw inputs into abstract embeddings for the LLM.")
    (d / "agents.md").write_text("The agent loop is ReAct style and calls tools like perceive.")
    rag = RAG(_cfg(tmp_path))
    n = rag.ingest([str(d)])
    assert n >= 2

    hits = rag.retrieve("what does the jepa encoder do", k=1)
    assert "JEPA" in hits[0][0].text

    ans = rag.ask("what does the jepa encoder do")
    assert isinstance(ans.answer, str) and ans.answer
    assert ans.sources and any("jepa" in s for s in ans.sources)


def test_rag_persists_and_reloads(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "x.md").write_text("The harness app exposes a CLI and a FastAPI server.")
    cfg = _cfg(tmp_path)
    rag = RAG(cfg)
    rag.ingest([str(d)])
    rag.save()
    reloaded = RAG(cfg).load()
    assert len(reloaded.store) >= 1
