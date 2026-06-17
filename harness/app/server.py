"""FastAPI server exposing the harness.

Endpoints:
  GET  /health           liveness + which backends are active
  POST /perceive         {"input": ...}        -> JEPA state description
  POST /chat             {"goal": ...}         -> agent answer + trace
  POST /eval             {"data_path": ...}    -> eval metrics

Built with `create_app(config)` so it can be embedded or launched via
`harness serve`. FastAPI/uvicorn are optional (`serve` extras).
"""

from __future__ import annotations

from typing import Any

from harness.config import HarnessConfig


def create_app(config: HarnessConfig | None = None):
    """Construct the FastAPI app. Imports FastAPI lazily so core stays light."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The server requires the 'serve' extras: pip install -e \".[serve]\""
        ) from e

    config = config or HarnessConfig()

    from harness.agents.agent import Agent
    from harness.jepa.interface import JEPABridge
    from harness.llm.model import LocalLLM

    # Shared, lazily-initialized components.
    bridge = JEPABridge(config)
    llm = LocalLLM(config)

    app = FastAPI(title="Harness", version="0.1.0")

    class PerceiveRequest(BaseModel):
        input: Any

    class ChatRequest(BaseModel):
        goal: str
        trace: bool = False

    class EvalRequest(BaseModel):
        data_path: str

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "llm_backend": llm.kind,
            "jepa_backend": bridge.encoder.backend,
            "base_model": config.base_model,
        }

    @app.post("/perceive")
    def perceive(req: PerceiveRequest) -> dict:
        desc = bridge.describe(req.input)
        return {
            "summary": desc.summary,
            "norm": desc.norm,
            "top_features": desc.top_features,
            "nearest_reference": desc.nearest_reference,
            "prompt_block": desc.as_prompt_block(),
        }

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        # Fresh agent per request so memory doesn't leak across users.
        agent = Agent(config, llm=llm, bridge=bridge)
        result = agent.run(req.goal)
        out: dict = {"answer": result.answer, "stopped_reason": result.stopped_reason}
        if req.trace:
            out["trace"] = [
                {"action": s.action, "input": s.action_input, "observation": s.observation}
                for s in result.steps
            ]
        return out

    @app.post("/eval")
    def eval_(req: EvalRequest) -> dict:
        from harness.llm.eval import evaluate_file

        result = evaluate_file(req.data_path, llm=llm, config=config)
        return {"n": result.n, "exact_match": result.exact_match, "token_f1": result.token_f1}

    return app
