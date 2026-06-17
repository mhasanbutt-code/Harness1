# Set up the harness on this machine (Windows PowerShell).
#
# Usage:
#   ./scripts/setup.ps1            # core + dev (fallbacks, no GPU)
#   ./scripts/setup.ps1 llm        # + real training stack
#   ./scripts/setup.ps1 serve      # + server stack
#   ./scripts/setup.ps1 all        # everything
param([string]$Extras = "dev")
$ErrorActionPreference = "Stop"

Write-Host "==> Using $(python --version)"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip | Out-Null

switch ($Extras) {
  "llm"   { pip install -e ".[llm,dev]" }
  "serve" { pip install -e ".[serve,dev]" }
  "all"   { pip install -e ".[llm,serve,dev]" }
  default { pip install -e ".[dev]" }
}

Write-Host ""
Write-Host "==> Done. Activate the env with:  . .\.venv\Scripts\Activate.ps1"
Write-Host "==> Smoke test:                   harness perceive examples/state.json"
Write-Host "==> Run the tests:                pytest -q"
