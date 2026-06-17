"""Ollama backend: drive a locally-served model (e.g. Qwen3) over HTTP.

Ollama (https://ollama.com) serves GGUF models for *inference* on
``localhost:11434``. This wraps its ``/api/generate``, ``/api/chat`` and
``/api/embeddings`` endpoints with the standard library only, so the harness
can use a model you've already ``ollama pull``ed without importing
torch/transformers.

Note: Ollama does **inference, not training**. To give the model knowledge of
your project without fine-tuning, use the RAG pipeline (``harness/rag``), which
retrieves context and feeds it to the Ollama model at query time.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    pass


def post(host: str, path: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    """POST JSON to an Ollama endpoint and return the decoded response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost)
        return json.loads(resp.read().decode("utf-8"))


def ping(host: str, timeout: float = 3.0) -> bool:
    """Return True if an Ollama daemon answers at ``host``."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            json.loads(resp.read().decode("utf-8"))
        return True
    except Exception:
        return False


class OllamaLLM:
    """Generation interface backed by an Ollama-served model."""

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen3", timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return ping(self.host)

    def _options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if "temperature" in overrides:
            opts["temperature"] = overrides["temperature"]
        if "max_new_tokens" in overrides:
            opts["num_predict"] = overrides["max_new_tokens"]
        return opts

    def generate(self, prompt: str, **overrides: Any) -> str:
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        opts = self._options(overrides)
        if opts:
            payload["options"] = opts
        out = post(self.host, "/api/generate", payload, self.timeout)
        return (out.get("response") or "").strip()

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        opts = self._options(overrides)
        if opts:
            payload["options"] = opts
        out = post(self.host, "/api/chat", payload, self.timeout)
        return (out.get("message", {}).get("content") or "").strip()

    def embed(self, text: str, model: str) -> list[float] | None:
        try:
            out = post(self.host, "/api/embeddings", {"model": model, "prompt": text}, timeout=30.0)
            return out.get("embedding")
        except Exception:
            return None
