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
  router.py            HarnessRouter: one entry point that dispatches to every capability
  client.py            stdlib HTTP client (for PC2/Mac -> host over the LAN)
  jepa/
    encoder.py         JEPA encoder (real loader + deterministic fallback)
    interface.py       JEPABridge: embeddings -> text/projection for the LLM
  llm/
    model.py           backend resolution: ollama -> transformers -> echo
    ollama.py          stdlib client for an Ollama-served model (e.g. Qwen3)
    data.py            JSONL instruction dataset prep
    train.py           LoRA/QLoRA fine-tuning
    eval.py            evaluation loop + metrics
  rag/
    embedder.py        embeddings (Ollama or offline hashing fallback)
    store.py           numpy vector store + cosine search
    ingest.py          document loading + chunking
    pipeline.py        RAG: retrieve project knowledge, answer, cite sources
  agents/
    tools.py           Tool base + registry + built-in tools
    memory.py          conversation/working memory
    agent.py           ReAct orchestration loop
  app/
    cli.py             `harness` command-line entry point (routes via HarnessRouter)
    server.py          FastAPI server (every endpoint routes via HarnessRouter)
examples/              runnable examples + sample state
data/                  sample training data
tests/                 fallback-path tests (no heavy deps required)
```

### The harness router

Every surface — the CLI, the HTTP server, and therefore every machine — goes
through a single `HarnessRouter` that dispatches a typed request (`perceive`,
`ask`, `ingest`, `chat`/`agent`, `health`) to the matching capability and owns
the shared LLM/JEPA/RAG components. The server is just a thin HTTP skin over the
router, so PC2 and the Mac hit exactly the same logic the host runs locally.
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

hc = HarnessClient("http://192.168.1.50:8000")      # the host running `harness serve`
print(hc.health())                                  # which backends are live
print(hc.ask("How does the JEPA bridge feed the LLM?"))   # RAG over the host's index
print(hc.chat("Inspect the current state and report health."))
print(hc.perceive("examples/state.json"))
# or the generic router entry point:
print(hc.route("ask", {"question": "what is the harness router?"}))
```

Index the knowledge once on the host (`harness ingest README.md harness`), then
every client `ask` is answered by the host's Ollama model, grounded in that
index — all through the same router.

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

## Teaching the model your stack (Ollama + RAG)

To make the local model *know* the harness/JEPA project, prefer **RAG over
fine-tuning**. Fine-tuning a small model on a little doc set tends to
hallucinate and forget, and must be redone whenever docs change. RAG indexes
your docs once and grounds every answer in the most relevant chunks — current,
citable, and no GPU training. It also rides directly on a model you already
serve with [Ollama](https://ollama.com).

### Point the harness at your Ollama model

```bash
ollama pull qwen3:32b             # your chat model
ollama pull nomic-embed-text      # embeddings for RAG (optional but better)

export HARNESS_LLM_BACKEND=ollama
export HARNESS_OLLAMA_MODEL=qwen3:32b   # default; override to your `ollama list` tag
export HARNESS_EMBED_BACKEND=ollama
```

The backend resolves automatically: `auto` uses Ollama if it's running, then
`transformers`, then a deterministic echo stub. The embedder falls back to an
offline lexical hashing embedder if no embed model is available, so RAG works
even with nothing pulled.

### Index the project, then ask

```bash
harness ingest README.md harness            # chunk + embed the docs
harness ask "How does the JEPA bridge feed state to the LLM?"
```

`ask` retrieves the top-k chunks, grounds the prompt in them, answers with your
Ollama model, and prints the source files it used.

### When you *do* want fine-tuning

For changing the model's *behavior/style* (not facts), use the LoRA path with
the `transformers` backend:

```bash
pip install -e ".[llm]"
harness train --data data/train.jsonl --epochs 3   # saves a LoRA adapter
harness eval  --data data/val.jsonl
```

Ollama itself does inference only; to serve a LoRA-tuned model through Ollama,
merge the adapter into the base weights, convert to GGUF, and import it with a
`Modelfile`.

