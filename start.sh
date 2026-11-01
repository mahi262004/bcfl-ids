#!/usr/bin/env bash
# Start the demo on macOS or Linux. Press Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then echo "Run setup first: bash setup.sh"; exit 1; fi
exec .venv/bin/python scripts/start_demo.py "$@"
