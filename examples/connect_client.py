"""Run this on PC2 or the Mac to talk to the harness router on the host (PC1).

Usage:
    python examples/connect_client.py [host_url] [question]

Defaults to http://127.0.0.1:8000. On a client machine, pass the host's LAN IP:
    python examples/connect_client.py http://192.168.1.50:8000 "What is the harness router?"

No torch / weights / GPU needed here — this just calls the router over HTTP.
"""

from __future__ import annotations

import sys

from harness.client import HarnessClient


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    question = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "How does the JEPA bridge feed state to the LLM?"
    )

    hc = HarnessClient(host)
    try:
        health = hc.health()
    except Exception as e:
        print(f"Could not reach the harness router at {host}: {e}")
        print("Is `harness serve` running on the host, and is port 8000 open in its firewall?")
        return 1

    print(f"Connected to {host}")
    print(f"  llm backend:    {health.get('llm_backend')}  ({health.get('ollama_model')})")
    print(f"  embed backend:  {health.get('embed_backend')}")
    print(f"  indexed chunks: {health.get('indexed_chunks')}")
    print()

    print(f"Q: {question}")
    ans = hc.ask(question)
    print(f"A: {ans.get('answer')}")
    sources = ans.get("sources") or []
    if sources:
        print("Sources: " + ", ".join(sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
