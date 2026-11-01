#!/usr/bin/env bash
# One-time setup on macOS or Linux. Needs internet once. Afterwards the demo runs offline.
set -euo pipefail
cd "$(dirname "$0")"

# The pinned PyTorch has wheels for Python 3.11 to 3.13 only, so a newer default python3 will not do.
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then echo "Python 3.11, 3.12 or 3.13 is required. Install it with: brew install python@3.13"; exit 1; fi
if ! command -v node >/dev/null 2>&1 || ! node -e "process.exit(+process.versions.node.split('.')[0] >= 18 ? 0 : 1)"; then echo "Node.js 18 or newer is required. Install the LTS from nodejs.org or run: brew install node"; exit 1; fi

echo "== Python environment =="
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "== Blockchain tools =="
(cd blockchain && npm ci --no-audit --no-fund)

echo "== IPFS node =="
.venv/bin/python scripts/install_ipfs.py

echo "== Tests =="
.venv/bin/python -m pytest -q
(cd blockchain && HARDHAT_DISABLE_TELEMETRY_PROMPT=true npx hardhat test)

echo
echo "Setup finished. Start the demo with:  bash start.sh"
