"""FastAPI server exposing the harness — every endpoint routes through the
shared :class:`~harness.router.HarnessRouter`.

Endpoints:
  GET  /health             liveness + which backends are active
  POST /perceive           {"input": ...}                  -> JEPA description
  POST /ask                {"question": ..., "k"?: int}    -> RAG answer + sources
  POST /ingest             {"paths": [...], "append"?: b}  -> index documents
  POST /chat               {"goal": ..., "trace"?: bool}   -> agent answer + trace
  POST /eval               {"data_path": ...}              -> eval metrics
  POST /route              {"kind": ..., "payload": {...}} -> generic dispatch

Built with `create_app(config)`; FastAPI/uvicorn are optional (`serve` extras).

Note: this module intentionally does NOT use ``from __future__ import
annotations``. FastAPI must see the request-model classes as real objects (not
lazy strings) to treat them as request bodies; with stringized annotations the
locally-defined Pydantic models can't be resolved and become query params.
"""

from typing import Any

from harness.config import HarnessConfig


def create_app(config: HarnessConfig | None = None):
    """Construct the FastAPI app. Imports FastAPI lazily so core stays light."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The server requires the 'serve' extras: pip install -e \".[serve]\""
        ) from e

    config = config or HarnessConfig()

    from harness.router import HarnessRouter, RouteError

    router = HarnessRouter(config)

    app = FastAPI(title="Harness", version="0.1.0")

    class PerceiveRequest(BaseModel):
        input: Any

    class AskRequest(BaseModel):
        question: str
        k: int | None = None

    class IngestRequest(BaseModel):
        paths: list[str]
        append: bool = True

    class ChatRequest(BaseModel):
        goal: str
        trace: bool = False

    class EvalRequest(BaseModel):
        data_path: str

    class RouteRequest(BaseModel):
        kind: str
        payload: dict[str, Any] = {}

    def _route(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return router.route(kind, payload).output
        except RouteError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/health")
    def health() -> dict:
        return _route("health", {})

    @app.post("/perceive")
    def perceive(req: PerceiveRequest) -> dict:
        return _route("perceive", {"input": req.input})

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        return _route("ask", {"question": req.question, "k": req.k})

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        return _route("ingest", {"paths": req.paths, "append": req.append})

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        return _route("chat", {"goal": req.goal, "trace": req.trace})

    @app.post("/eval")
    def eval_(req: EvalRequest) -> dict:
        from harness.llm.eval import evaluate_file

        result = evaluate_file(req.data_path, llm=router.llm, config=config)
        return {"n": result.n, "exact_match": result.exact_match, "token_f1": result.token_f1}

    @app.post("/route")
    def route(req: RouteRequest) -> dict:
        return _route(req.kind, req.payload)

    return app
