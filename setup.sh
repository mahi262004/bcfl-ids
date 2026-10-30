#!/usr/bin/env bash
# One-time setup on macOS or Linux. Needs internet once. Afterwards the demo runs offline.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then echo "Python 3.11+ is required. Install it from python.org or run: brew install python@3.13"; exit 1; fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || { echo "Python 3.11+ is required, found $("$PY" --version)"; exit 1; }
if ! command -v npm >/dev/null 2>&1; then echo "Node.js is required. Install the LTS from nodejs.org or run: brew install node"; exit 1; fi

echo "== Python environment =="
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "== Blockchain tools =="
(cd blockchain && npm ci --no-audit --no-fund)

echo "== Tests =="
.venv/bin/python -m pytest -q
(cd blockchain && HARDHAT_DISABLE_TELEMETRY_PROMPT=true npx hardhat test)

echo
echo "Setup finished. Start the demo with:  .venv/bin/python scripts/start_demo.py"
