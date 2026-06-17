"""The harness router: one entry point that dispatches a request to the right
capability.

Every surface routes through this — the CLI, the HTTP server, and therefore
every machine (PC1 hosting it, PC2 and the Mac as LAN clients) hits the same
logic. The router owns the shared, lazily-initialized components (LLM/Ollama,
JEPA bridge, RAG index) so they're built once and reused across requests.

Capabilities (``kind``):
* ``health``   — which backends are live.
* ``perceive`` — JEPA description of an input.            payload: {"input": ...}
* ``ask``      — RAG answer grounded in indexed docs.     payload: {"question", "k"?}
* ``ingest``   — index documents for RAG.                 payload: {"paths", "append"?}
* ``chat`` / ``agent`` — run the tool-using agent loop.   payload: {"goal", "trace"?}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.config import HarnessConfig


class RouteError(ValueError):
    """Raised for an unknown route kind or a malformed payload."""


@dataclass
class RouteResult:
    kind: str
    output: dict[str, Any] = field(default_factory=dict)


class HarnessRouter:
    """Dispatch a typed request to the matching harness capability."""

    KINDS = ("health", "perceive", "ask", "ingest", "chat", "agent")

    def __init__(self, config: HarnessConfig | None = None, llm=None, bridge=None, rag=None) -> None:
        from harness.jepa.interface import JEPABridge
        from harness.llm.model import LocalLLM
        from harness.rag.pipeline import RAG

        self.config = config or HarnessConfig()
        self.llm = llm or LocalLLM(self.config)
        self.bridge = bridge or JEPABridge(self.config)
        self.rag = rag or RAG(self.config, llm=self.llm)
        if self.config.rag_store_path.exists():
            try:
                self.rag.load()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def route(self, kind: str, payload: dict[str, Any] | None = None) -> RouteResult:
        payload = payload or {}
        handler = {
            "health": self._health,
            "perceive": self._perceive,
            "ask": self._ask,
            "ingest": self._ingest,
            "chat": self._agent,
            "agent": self._agent,
        }.get(kind)
        if handler is None:
            raise RouteError(f"unknown route kind '{kind}'. Valid: {', '.join(self.KINDS)}")
        return RouteResult(kind=kind, output=handler(payload))

    # ------------------------------------------------------------------ #
    def _health(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_backend": self.llm.kind,
            "jepa_backend": self.bridge.encoder.backend,
            "embed_backend": self.rag.embedder.backend,
            "base_model": self.config.base_model,
            "ollama_model": self.config.ollama_model,
            "active_model": self.llm.active_model(),
            "indexed_chunks": len(self.rag.store),
        }

    def _perceive(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "input" not in payload:
            raise RouteError("perceive requires 'input'")
        desc = self.bridge.describe(payload["input"])
        return {
            "summary": desc.summary,
            "norm": desc.norm,
            "top_features": desc.top_features,
            "nearest_reference": desc.nearest_reference,
            "prompt_block": desc.as_prompt_block(),
        }

    def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "question" not in payload:
            raise RouteError("ask requires 'question'")
        ans = self.rag.ask(payload["question"], k=payload.get("k"))
        return {
            "answer": ans.answer,
            "sources": sorted(set(ans.sources)),
            "contexts": ans.contexts,
        }

    def _ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        paths = payload.get("paths")
        if not paths:
            raise RouteError("ingest requires a non-empty 'paths' list")
        if not payload.get("append", True):
            from harness.rag.store import VectorStore

            self.rag.store = VectorStore()
        n = self.rag.ingest(paths)
        self.rag.save()
        return {"ingested": n, "total_chunks": len(self.rag.store), "embed_backend": self.rag.embedder.backend}

    def _agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        from harness.agents.agent import Agent

        if "goal" not in payload:
            raise RouteError("chat/agent requires 'goal'")
        # Fresh agent per call so working memory doesn't leak across requests.
        agent = Agent(self.config, llm=self.llm, bridge=self.bridge)
        result = agent.run(payload["goal"])
        out: dict[str, Any] = {"answer": result.answer, "stopped_reason": result.stopped_reason}
        if payload.get("trace"):
            out["trace"] = [
                {"action": s.action, "input": s.action_input, "observation": s.observation}
                for s in result.steps
            ]
        return out
