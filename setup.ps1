# One-time setup on Windows. Needs internet once. Afterwards the demo runs offline.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m venv .venv
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -r requirements.txt
Push-Location blockchain
npm ci --no-audit --no-fund
Pop-Location

.venv\Scripts\python -m pytest -q
Push-Location blockchain
$env:HARDHAT_DISABLE_TELEMETRY_PROMPT = "true"
npx hardhat test
Pop-Location

Write-Host "Setup finished. Start the demo with:  .venv\Scripts\python scripts\start_demo.py"
