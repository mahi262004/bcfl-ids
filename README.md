# BCFL-IDS

Explainable, trust-aware blockchain federated learning for IoT intrusion detection.
Minor Project, Group 275, Amity University Uttar Pradesh.

**Team:** Mahi Nagar, Aanika Bajaj, Anshul Sharma. **Guide:** Dr. Rashmi.

Nine IoT devices train one botnet detector without sharing raw traffic. The server scores every model update and rejects suspicious ones, a trimmed mean absorbs what gets through, every decision is recorded on a local blockchain, and each alert can be explained in terms of traffic features.

All attacks are simulated, and all traffic is replayed from the N-BaIoT dataset. No real malware is used.

## Set it up on your device

### 1. Install what it needs

| | macOS | Windows |
|---|---|---|
| Python 3.11, 3.12 or 3.13 | `brew install python@3.13` | [python.org](https://www.python.org/downloads/), and tick "Add python.exe to PATH" |
| Node.js 18 or newer | `brew install node` | [nodejs.org](https://nodejs.org/) (LTS) |
| Git (optional) | `xcode-select --install` | [git-scm.com](https://git-scm.com/downloads) |

### 2. Get the code

The repository is private, so your GitHub account needs access to it. Then either clone it:

```bash
git clone https://github.com/mahi262004/bcfl-ids.git
cd bcfl-ids
```

or, while signed in to GitHub, [download the ZIP](https://github.com/mahi262004/bcfl-ids/archive/HEAD.zip) and unzip it.

### 3. Run setup once

This needs internet. It installs everything into the project folder, then runs the test suites to confirm the install works.

| macOS or Linux | Windows |
|---|---|
| In a terminal, in the project folder: `bash setup.sh` | Double-click `setup.bat` |

## Run the demo

| macOS or Linux | Windows |
|---|---|
| `bash start.sh` | Double-click `start.bat` |

The dashboard opens at <http://127.0.0.1:8000>. No internet is needed from here on. Keep the window open while you use it, and press Ctrl-C there to stop.

To check everything end to end, leave the demo running and run this in a second terminal. Every line should say PASS.

```bash
.venv/bin/python scripts/preflight.py        # macOS or Linux
.venv\Scripts\python scripts\preflight.py    # Windows
```

To present it, follow [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md). It has the walkthrough, likely questions and what to do if something goes wrong.

### If setup or the demo has a problem

| What you see | What to do |
|---|---|
| "Python 3.11, 3.12 or 3.13 is required" | Install one of those versions, open a new terminal and run setup again. |
| "Node.js 18 or newer is required" | Install the Node.js LTS, open a new terminal and run setup again. |
| "Run setup first" | Setup has not finished on this device. Run it again. |
| The dashboard says it is using the local ledger | The demo still works, but without the blockchain. Stop it, start it again, and check `runtime/chain.log`. |
| Port 8000 is busy | Nothing to do: the terminal prints the address it used instead. |

## How it works

1. **Robust scale:** devices share feature statistics, not traffic, and the server takes a trimmed mean of them, so no single device can distort the scale.
2. **Local training:** each device trains the same small neural network on its own traffic and sends back an update.
3. **Trust gate:** each update is checked for direction and size against the typical update, and each device carries a reputation score.
4. **Aggregation:** the updates that pass are combined with a trimmed mean.
5. **Audit:** every verdict is written to a smart contract, so later tampering with the off-chain records can be detected.
6. **Explanation:** Integrated Gradients shows which traffic features drove each alert.

## Results

Measured on 1,412,554 later traffic records never used in training, over three seeds.

| No attackers | |
|---|---|
| Attacks detected | 99.95% |
| Benign traffic correctly passed | 99.87% |
| Macro F1 across 11 classes | 0.672 |
| Macro F1 with look-alike families merged | 0.870 |

| Macro F1 with 3 of 9 devices attacking | No defence | This system |
|---|---|---|
| Sign flip | 0.021 | 0.657 |
| Scaling | 0.001 | 0.657 |
| Random weights | 0.347 | 0.657 |
| Label flip | 0.563 | 0.654 |
| Targeted evasion | 0.557 | 0.650 |
| Little Is Enough | 0.661 | 0.655 |
| Min-Max | 0.647 | 0.658 |
| Fang | 0.546 | 0.637 |
| Adaptive | 0.553 | 0.596 |
| Statistics poisoning | 0.022 | 0.666 |

The full tables, method and limits are in [docs/RESULTS.md](docs/RESULTS.md).

### Limits

- The system is built for up to 3 attackers of 9. With 4, the adaptive, Fang and statistics-poisoning attacks defeat it.
- Attack versus benign is easy on this dataset: a depth-3 decision tree reaches about 99.8%. Performance under attack is the meaningful result.
- The results come from one dataset, captured in 2018. No physical device was tested.
- The chain is a single local node, and the server is the only oracle. Records are tamper-evident, but the scoring is not decentralised.
- Model updates reach the server unencrypted. There is no secure aggregation or differential privacy.
- Zero-knowledge verification is not implemented.

## What is in the repository

| Path | Purpose |
|---|---|
| `setup.sh`, `setup.bat` | One-time setup |
| `start.sh`, `start.bat` | Start the demo |
| `bcfl_ids/` | Data pipeline, model, attacks, defences, simulator, ledger, chain bridge, explanations |
| `blockchain/` | `TrustLayer` Solidity contract, its tests, compiled artifact |
| `backend/` | FastAPI service |
| `frontend/` | Dashboard, with no external dependencies |
| `experiments/` | Scripts that produce every number in `artifacts/results` |
| `artifacts/` | Trained demo model, small data sample and measured results |
| `tests/` | Python test suite |
| `scripts/` | Demo launcher and end-to-end preflight check |

## Reproduce the results

The demo does not need the dataset. To regenerate the results, download [N-BaIoT from UCI](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) to `data/raw/nbaiot.zip` (1.8 GB; about 12 GB once unpacked and converted), then run these with the project's Python:

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

The experiments use an in-process simulator because it is fast and exactly repeatable. The same trust logic also runs as a real server and nine client processes over gRPC:

```bash
python -m venv .venv-flower
.venv-flower/bin/python -m pip install -r requirements-flower.txt
.venv-flower/bin/python -m bcfl_ids.flower_app --rounds 3
```
