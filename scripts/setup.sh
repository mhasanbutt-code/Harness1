#!/usr/bin/env bash
# Set up the harness on this machine (macOS or Linux).
#
# Usage:
#   ./scripts/setup.sh           # core + dev (fallbacks, no GPU)
#   ./scripts/setup.sh llm       # + real training stack (torch/transformers/peft)
#   ./scripts/setup.sh serve     # + server stack (fastapi/uvicorn)
#   ./scripts/setup.sh all       # everything
set -euo pipefail

PYTHON="${PYTHON:-python3}"
echo "==> Using $("$PYTHON" --version)"

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null

EXTRAS="${1:-dev}"
case "$EXTRAS" in
  llm)   pip install -e ".[llm,dev]" ;;
  serve) pip install -e ".[serve,dev]" ;;
  all)   pip install -e ".[llm,serve,dev]" ;;
  *)     pip install -e ".[dev]" ;;
esac

echo
echo "==> Done. Activate the env with:  source .venv/bin/activate"
echo "==> Smoke test:                   harness perceive examples/state.json"
echo "==> Run the tests:                pytest -q"
