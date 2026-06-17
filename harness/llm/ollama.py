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


def list_models(host: str, timeout: float = 3.0) -> list[str]:
    """Return the model tags currently available in the Ollama daemon."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            out = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in out.get("models", []) if m.get("name")]
    except Exception:
        return []


def select_model(requested: str, available: list[str]) -> str:
    """Pick the best matching model tag from what's actually installed.

    Order of preference: exact match -> same family (same name before ':',
    e.g. a requested ``qwen3:32b`` matches an installed ``qwen3:8b``) ->
    first non-embedding chat model -> the requested string unchanged (so a
    daemon that doesn't list tags still gets a best effort).
    """
    if not available:
        return requested
    if requested in available:
        return requested
    base = requested.split(":")[0].lower()
    family = [m for m in available if m.split(":")[0].lower() == base]
    if family:
        return family[0]
    chat = [m for m in available if "embed" not in m.lower()]
    return chat[0] if chat else available[0]


class OllamaLLM:
    """Generation interface backed by an Ollama-served model."""

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen3", timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return ping(self.host)

    def ensure_model(self) -> str:
        """Resolve ``self.model`` to a tag that's actually installed."""
        self.model = select_model(self.model, list_models(self.host))
        return self.model

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
