"""Thin HTTP client for talking to a remote harness server.

Lets a second machine (PC2, a Mac, ...) drive a harness *server* running on a
third machine over the LAN — without installing torch/transformers locally.
Standard library only, so it works on a fresh Python with no extras.

    from harness.client import HarnessClient
    hc = HarnessClient("http://192.168.1.50:8000")   # the box running `harness serve`
    print(hc.health())
    print(hc.chat("Inspect the current state and report health."))
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any


class HarnessClient:
    """Minimal HTTP client for the harness FastAPI server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
        token: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (trusted LAN URL)
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def perceive(self, input: Any) -> dict[str, Any]:
        return self._request("POST", "/perceive", {"input": input})

    def chat(self, goal: str, trace: bool = False) -> dict[str, Any]:
        return self._request("POST", "/chat", {"goal": goal, "trace": trace})

    def ask(self, question: str, k: int | None = None) -> dict[str, Any]:
        return self._request("POST", "/ask", {"question": question, "k": k})

    def ingest(self, paths: list[str], append: bool = True) -> dict[str, Any]:
        return self._request("POST", "/ingest", {"paths": paths, "append": append})

    def eval(self, data_path: str) -> dict[str, Any]:
        return self._request("POST", "/eval", {"data_path": data_path})

    def route(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generic dispatch through the harness router."""
        return self._request("POST", "/route", {"kind": kind, "payload": payload or {}})
