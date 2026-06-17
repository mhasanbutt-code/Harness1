# Harness

A local-first stack that ties together four pieces, built in sequence:

1. **JEPA ↔ LLM interface** (`harness/jepa/`) — a Joint-Embedding Predictive
   Architecture encoder that turns raw inputs into abstract state embeddings,
   plus a bridge that makes those embeddings usable by a language model.
2. **Local LLM training** (`harness/llm/`) — data prep, LoRA/QLoRA fine-tuning,
   and an evaluation loop for an open-weights model you run yourself.
3. **Agent framework** (`harness/agents/`) — tools, memory, and a ReAct-style
   orchestration loop that lets the model perceive (via JEPA) and act.
4. **Harness app** (`harness/app/`) — a CLI and an HTTP server that expose the
   whole thing: perceive, chat, and train.

## Design at a glance

```
        raw input (image / sensor / state)
                     │
                     ▼
        ┌────────────────────────┐
        │  JEPA encoder          │  perception / world-model
        │  harness/jepa/encoder  │  -> abstract embedding
        └───────────┬────────────┘
                    │  JEPABridge.describe() / .project()
                    ▼
        ┌────────────────────────┐
        │  Local LLM             │  reasoning / language
        │  harness/llm/model     │  (LoRA-tuned, runs locally)
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  Agent loop            │  acting
        │  harness/agents/agent  │  tools + memory + ReAct
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │  Harness app           │  CLI + FastAPI
        │  harness/app           │
        └────────────────────────┘
```

JEPA "sees and understands," the LLM "reasons and talks," the agents "act,"
and the harness app is how you drive all of it.

## Runs with no heavy dependencies

Every heavy dependency (`torch`, `transformers`, `peft`, `fastapi`) is
imported lazily and guarded. If a dependency or model weights are missing the
component falls back to a deterministic stub so the whole pipeline still runs
end-to-end — handy for development, CI, and trying the design before you commit
GPU time.

## Quickstart

```bash
pip install -e .                 # core only; add extras below as needed
pip install -e ".[llm]"          # transformers + peft for real training
pip install -e ".[serve]"        # fastapi + uvicorn for the server

# Perceive an input and get a textual description (uses fallback encoder)
harness perceive "examples/state.json"

# Talk to the agent (uses fallback LLM if no weights are present)
harness chat "Summarize the current harness state and propose a next step"

# Fine-tune the local LLM with LoRA on your data
harness train --data data/sample_train.jsonl --epochs 3

# Run the HTTP server
harness serve --port 8000
```

## Layout

```
harness/
  config.py            central configuration
  jepa/
    encoder.py         JEPA encoder (real loader + deterministic fallback)
    interface.py       JEPABridge: embeddings -> text/projection for the LLM
  llm/
    model.py           local causal-LM loader (+ echo fallback)
    data.py            JSONL instruction dataset prep
    train.py           LoRA/QLoRA fine-tuning
    eval.py            evaluation loop + metrics
  agents/
    tools.py           Tool base + registry + built-in tools
    memory.py          conversation/working memory
    agent.py           ReAct orchestration loop
  app/
    cli.py             `harness` command-line entry point
    server.py          FastAPI server
examples/              runnable examples + sample state
data/                  sample training data
tests/                 fallback-path tests (no heavy deps required)
```

See `CONTRIBUTING` notes inline in each module's docstring.

## Running across machines (this PC, PC2, Mac)

The harness runs **locally on each machine** — there's no cloud component. To
use it from several machines, run the server on **one** box (ideally the one
with the GPU) and have the others connect to it as clients.

### 1. Set up each machine

```bash
# macOS / Linux
git clone <your-repo-url> && cd <repo>
./scripts/setup.sh all          # or: llm / serve / (none) for fewer deps
```

```powershell
# Windows PowerShell
git clone <your-repo-url>; cd <repo>
./scripts/setup.ps1 all
```

### 2. Pick a host and run the server there

On the machine that will do the heavy lifting (say its LAN IP is
`192.168.1.50`):

```bash
make serve            # binds 0.0.0.0:8000 so the LAN can reach it
# or: harness serve --host 0.0.0.0 --port 8000
```

Open that port in the host's firewall.

### 3. Connect the other machines as clients

From **PC2** and the **Mac**, point the stdlib client at the host — no torch or
weights needed on these boxes:

```python
from harness.client import HarnessClient

hc = HarnessClient("http://192.168.1.50:8000")
print(hc.health())                                  # which backends are live
print(hc.chat("Inspect the current state and report health."))
print(hc.perceive("examples/state.json"))
```

Topology:

```
   ┌────────────┐        HTTP/LAN        ┌────────────┐
   │   PC2      │ ─────────────────────▶ │  Host PC   │  harness serve
   │ (client)   │                        │ (GPU, LLM, │  + JEPA + agents
   └────────────┘                        │  JEPA)     │
   ┌────────────┐        HTTP/LAN        │            │
   │   Mac      │ ─────────────────────▶ │            │
   │ (client)   │                        └────────────┘
   └────────────┘
```

For access beyond the LAN, put the server behind a reverse proxy / VPN and add
authentication — the built-in server is unauthenticated and meant for trusted
networks.

