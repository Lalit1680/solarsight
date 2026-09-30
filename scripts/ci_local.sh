#!/usr/bin/env bash
# Run the same steps as .github/workflows/ci.yml, locally.
# Usage: bash scripts/ci_local.sh
set -euo pipefail

PY=${PY:-python}

echo "== 1. Dependency consistency =="
"$PY" -m pip check

echo "== 2. Tests =="
"$PY" -m pytest -v

echo "== 3. Import check without secrets =="
"$PY" -c "import app; print('app module imports cleanly without secrets')"

echo "ALL CI STEPS PASSED"
