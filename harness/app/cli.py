"""`harness` command-line interface.

Subcommands:
  perceive  run the JEPA bridge on an input and print the state description
  chat      run the agent loop on a goal and print the answer + trace
  train     LoRA fine-tune the local LLM on a JSONL dataset
  eval      evaluate the local LLM on a JSONL dataset
  serve     launch the FastAPI server

Everything works with fallbacks except `train`, which needs the `llm` extras
and says so clearly.
"""

from __future__ import annotations

import argparse
import json
import sys

from harness.config import HarnessConfig


def _cmd_perceive(args: argparse.Namespace, config: HarnessConfig) -> int:
    from harness.router import HarnessRouter

    router = HarnessRouter(config)
    out = router.route("perceive", {"input": args.input}).output
    print(out["prompt_block"])
    print(f"\n(jepa backend: {router.bridge.encoder.backend})")
    return 0


def _cmd_chat(args: argparse.Namespace, config: HarnessConfig) -> int:
    from harness.router import HarnessRouter

    router = HarnessRouter(config)
    out = router.route("chat", {"goal": args.goal, "trace": args.trace}).output
    print(out["answer"])
    if args.trace:
        print("\n--- trace ---", file=sys.stderr)
        for i, step in enumerate(out.get("trace", []), 1):
            print(f"[{i}] action={step['action']!r} input={step['input']!r}", file=sys.stderr)
            if step["observation"]:
                print(f"    observation: {step['observation'].splitlines()[0]}", file=sys.stderr)
        print(f"stopped: {out['stopped_reason']}", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace, config: HarnessConfig) -> int:
    from harness.llm.train import MissingTrainingDeps, train_lora

    if args.epochs:
        config.num_epochs = args.epochs
    try:
        adapter = train_lora(config, args.data)
    except MissingTrainingDeps as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"Saved LoRA adapter to: {adapter}")
    return 0


def _cmd_eval(args: argparse.Namespace, config: HarnessConfig) -> int:
    from harness.llm.eval import evaluate_file

    result = evaluate_file(args.data, config=config)
    print(result.summary())
    if args.show_samples:
        print(json.dumps(result.samples, indent=2))
    return 0


def _cmd_ingest(args: argparse.Namespace, config: HarnessConfig) -> int:
    from harness.router import HarnessRouter

    router = HarnessRouter(config)
    out = router.route("ingest", {"paths": args.paths, "append": args.append}).output
    print(f"Ingested {out['ingested']} chunk(s) from {len(args.paths)} path(s) -> {config.rag_store_path}")
    print(f"(embedder backend: {out['embed_backend']}, total chunks: {out['total_chunks']})")
    return 0


def _cmd_ask(args: argparse.Namespace, config: HarnessConfig) -> int:
    from harness.router import HarnessRouter

    if not config.rag_store_path.exists():
        print("No index found. Run `harness ingest <paths>` first.", file=sys.stderr)
        return 2
    router = HarnessRouter(config)
    out = router.route("ask", {"question": args.question}).output
    print(out["answer"])
    if out["sources"]:
        print("\nSources: " + ", ".join(out["sources"]), file=sys.stderr)
    print(f"(llm backend: {router.llm.kind})", file=sys.stderr)
    return 0


def _cmd_serve(args: argparse.Namespace, config: HarnessConfig) -> int:
    try:
        import uvicorn
    except Exception:
        print("Serving requires the 'serve' extras: pip install -e \".[serve]\"", file=sys.stderr)
        return 2
    from harness.app.server import create_app

    uvicorn.run(create_app(config), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="JEPA + LLM + agent harness")
    p.add_argument("--config", help="path to a JSON config file", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("perceive", help="describe an input via JEPA")
    sp.add_argument("input", help="path, state string, or 'current'")
    sp.set_defaults(func=_cmd_perceive)

    sc = sub.add_parser("chat", help="run the agent on a goal")
    sc.add_argument("goal", help="the goal for the agent")
    sc.add_argument("--trace", action="store_true", help="print the step trace to stderr")
    sc.set_defaults(func=_cmd_chat)

    st = sub.add_parser("train", help="LoRA fine-tune the local LLM")
    st.add_argument("--data", required=True, help="JSONL training file")
    st.add_argument("--epochs", type=int, default=None)
    st.set_defaults(func=_cmd_train)

    se = sub.add_parser("eval", help="evaluate the local LLM")
    se.add_argument("--data", required=True, help="JSONL eval file")
    se.add_argument("--show-samples", action="store_true")
    se.set_defaults(func=_cmd_eval)

    si = sub.add_parser("ingest", help="index documents for RAG over project knowledge")
    si.add_argument("paths", nargs="+", help="files or directories to index")
    si.add_argument("--append", action="store_true", help="add to the existing index")
    si.set_defaults(func=_cmd_ingest)

    sa = sub.add_parser("ask", help="ask a question grounded in ingested docs (RAG)")
    sa.add_argument("question", help="the question to answer from indexed knowledge")
    sa.set_defaults(func=_cmd_ask)

    sv = sub.add_parser("serve", help="run the HTTP server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = HarnessConfig.load(args.config)
    return args.func(args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
