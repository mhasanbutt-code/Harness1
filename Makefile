.PHONY: install install-llm install-serve install-all test perceive chat serve train clean

# Core install (fallbacks only — no GPU/heavy deps)
install:
	pip install -e ".[dev]"

# Add the real training stack (torch + transformers + peft)
install-llm:
	pip install -e ".[llm]"

# Add the server stack (fastapi + uvicorn)
install-serve:
	pip install -e ".[serve]"

# Everything
install-all:
	pip install -e ".[llm,serve,dev]"

test:
	pytest -q

perceive:
	harness perceive examples/state.json

chat:
	harness chat "Inspect the current harness state and report health." --trace

# Bind 0.0.0.0 so PC2 / the Mac on the LAN can reach this server
serve:
	harness serve --host 0.0.0.0 --port 8000

train:
	harness train --data data/sample_train.jsonl

clean:
	rm -rf outputs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
