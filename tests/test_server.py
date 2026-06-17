"""HTTP-level tests for the harness server.

These exercise the real FastAPI request/response cycle (not just the router in
Python), which is where body-vs-query binding bugs show up. Skipped if the
optional serve stack (fastapi + httpx for TestClient) isn't installed.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from harness.config import HarnessConfig  # noqa: E402
from harness.app.server import create_app  # noqa: E402


def _client(tmp_path) -> TestClient:
    cfg = HarnessConfig(
        llm_backend="echo",
        embed_backend="hash",
        embed_dim=128,
        jepa_embed_dim=64,
        rag_store_path=tmp_path / "store",
    )
    return TestClient(create_app(cfg))


def test_health_ok(tmp_path):
    r = _client(tmp_path).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_perceive_is_a_body_endpoint(tmp_path):
    # Regression: with stringized annotations FastAPI treated this as a query
    # param and returned 422. It must accept a JSON body.
    r = _client(tmp_path).post("/perceive", json={"input": "current"})
    assert r.status_code == 200
    assert "[PERCEIVED STATE]" in r.json()["prompt_block"]


def test_ingest_then_ask(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "j.md").write_text("The JEPA encoder turns inputs into abstract embeddings.")
    client = _client(tmp_path)

    r = client.post("/ingest", json={"paths": [str(d)]})
    assert r.status_code == 200 and r.json()["ingested"] >= 1

    r = client.post("/ask", json={"question": "what does the jepa encoder do"})
    assert r.status_code == 200
    assert r.json()["answer"]


def test_route_dispatch_and_errors(tmp_path):
    client = _client(tmp_path)
    ok = client.post("/route", json={"kind": "perceive", "payload": {"input": "x"}})
    assert ok.status_code == 200
    bad = client.post("/route", json={"kind": "bogus", "payload": {}})
    assert bad.status_code == 400
