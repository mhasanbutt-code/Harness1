import pytest

from harness.config import HarnessConfig
from harness.router import HarnessRouter, RouteError


def _cfg(tmp_path):
    return HarnessConfig(
        embed_backend="hash",
        embed_dim=128,
        llm_backend="echo",
        jepa_embed_dim=64,
        rag_store_path=tmp_path / "store",
    )


def test_health_route(tmp_path):
    out = HarnessRouter(_cfg(tmp_path)).route("health", {}).output
    assert out["status"] == "ok"
    assert out["llm_backend"] == "echo"
    assert out["embed_backend"] == "hash"
    assert out["indexed_chunks"] == 0


def test_perceive_route(tmp_path):
    out = HarnessRouter(_cfg(tmp_path)).route("perceive", {"input": "current"}).output
    assert "[PERCEIVED STATE]" in out["prompt_block"]


def test_unknown_kind_raises(tmp_path):
    with pytest.raises(RouteError):
        HarnessRouter(_cfg(tmp_path)).route("nope", {})


def test_missing_payload_raises(tmp_path):
    router = HarnessRouter(_cfg(tmp_path))
    with pytest.raises(RouteError):
        router.route("ask", {})
    with pytest.raises(RouteError):
        router.route("perceive", {})


def test_ingest_then_ask_through_router(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "jepa.md").write_text("The JEPA encoder turns raw inputs into abstract embeddings.")
    (d / "app.md").write_text("The harness app exposes a CLI and a FastAPI server.")
    router = HarnessRouter(_cfg(tmp_path))

    ing = router.route("ingest", {"paths": [str(d)]}).output
    assert ing["ingested"] >= 2
    assert ing["total_chunks"] >= 2

    ans = router.route("ask", {"question": "what does the jepa encoder do"}).output
    assert isinstance(ans["answer"], str) and ans["answer"]
    assert any("jepa" in s for s in ans["sources"])


def test_agent_route(tmp_path):
    out = HarnessRouter(_cfg(tmp_path)).route("chat", {"goal": "check state", "trace": True}).output
    assert isinstance(out["answer"], str) and out["answer"]
    assert "trace" in out
