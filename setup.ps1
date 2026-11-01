# One-time setup on Windows. Needs internet once. Afterwards the demo runs offline.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Windows PowerShell does not stop on a failed native command, so check each one.
function Run($exe) {
    & $exe @args
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped: $exe $args failed." }
}

# The pinned PyTorch has wheels for Python 3.11 to 3.13 only, so a newer default python will not do.
$check = 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'
$py = $null
foreach ($candidate in @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("python"), @("python3"))) {
    try {
        $flags = @($candidate | Select-Object -Skip 1)
        & $candidate[0] @flags -c $check 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $candidate; break }
    } catch { }
}
if (-not $py) { throw "Python 3.11, 3.12 or 3.13 is required. Install it from python.org and tick 'Add python.exe to PATH'." }

$nodeOk = $false
try {
    & node -e "process.exit(+process.versions.node.split('.')[0] >= 18 ? 0 : 1)" 2>$null
    $nodeOk = $LASTEXITCODE -eq 0
} catch { }
if (-not $nodeOk) { throw "Node.js 18 or newer is required. Install the LTS from nodejs.org." }

Write-Host "== Python environment =="
$flags = @($py | Select-Object -Skip 1)
Run $py[0] @flags -m venv .venv
Run .venv\Scripts\python -m pip install --quiet --upgrade pip
Run .venv\Scripts\python -m pip install --quiet -r requirements.txt

Write-Host "== Blockchain tools =="
Push-Location blockchain
Run npm ci --no-audit --no-fund
Pop-Location

Write-Host "== IPFS node =="
Run .venv\Scripts\python scripts\install_ipfs.py

Write-Host "== Tests =="
Run .venv\Scripts\python -m pytest -q
Push-Location blockchain
$env:HARDHAT_DISABLE_TELEMETRY_PROMPT = "true"
Run npx hardhat test
Pop-Location

Write-Host ""
Write-Host "Setup finished. Start the demo by double-clicking start.bat"
