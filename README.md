# BCFL-IDS

Explainable, trust-aware blockchain federated learning for IoT intrusion detection.
Minor Project, Group 275, Amity University Uttar Pradesh.

Nine IoT devices train one botnet detector without sharing raw traffic. The server scores every
model update and rejects suspicious ones, a trimmed mean absorbs what gets through, every decision
is recorded on a local blockchain, and each alert can be explained in terms of traffic features.

All attacks are simulated, and all traffic is replayed from the N-BaIoT dataset. No real malware is used.

- Measured results: [docs/RESULTS.md](docs/RESULTS.md)
- How to present it: [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)

## Run the demo

You need Python 3.11, 3.12 or 3.13 and Node.js 18 or newer. Setup needs internet once; afterwards the demo runs offline.

**macOS or Linux**

```bash
bash setup.sh
.venv/bin/python scripts/start_demo.py
```

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.venv\Scripts\python scripts\start_demo.py
```

The dashboard opens at <http://127.0.0.1:8000>. Press Ctrl-C to stop.

To check everything end to end, leave the demo running and run this in a second terminal:

```bash
.venv/bin/python scripts/preflight.py
```

Add `--no-chain` to `start_demo.py` to run without the blockchain. The dashboard then says it is using the local ledger.

The dataset is not needed for the demo: the trained model and a small data sample are in `artifacts/demo`.

## What is in the repository

| Path | Purpose |
|---|---|
| `bcfl_ids/` | Data pipeline, model, attacks, defences, simulator, ledger, chain bridge, explanations |
| `blockchain/` | `TrustLayer` Solidity contract, its tests, compiled artifact |
| `backend/` | FastAPI service |
| `frontend/` | Dashboard, with no external dependencies |
| `experiments/` | Scripts that produce every number in `artifacts/results` |
| `tests/` | Python test suite |
| `scripts/` | Demo launcher and end-to-end preflight check |

## Reproduce the results

Download [N-BaIoT from UCI](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) to `data/raw/nbaiot.zip` (1.8 GB; about 12 GB once unpacked and converted), then:

```bash
python -m bcfl_ids.data                 # unpack and convert, checks row counts
python -m experiments.calibrate         # gate thresholds, validation split only
python -m experiments.tune              # training settings, validation split only
python -m experiments.audit_data        # how hard is the dataset?
python -m experiments.run_grid          # every attack against every defence, resumable
python -m experiments.summarise_grid
python -m experiments.explain_eval
python -m experiments.build_demo        # deployed model and demo sample
python -m experiments.write_results     # docs/RESULTS.md
```

Tests: `python -m pytest` and, in `blockchain/`, `npx hardhat test`.

## Networked run with Flower

The experiments use an in-process simulator because it is fast and exactly repeatable. The same trust
logic also runs as a real server and nine client processes over gRPC:

```bash
python -m venv .venv-flower
.venv-flower/bin/python -m pip install -r requirements-flower.txt
.venv-flower/bin/python -m bcfl_ids.flower_app --rounds 3
```

## Not built

Zero-knowledge verification and the physical camera testbed are out of scope for this phase. The chain is a single local node, as agreed with the project guide.
