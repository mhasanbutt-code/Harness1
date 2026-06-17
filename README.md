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
