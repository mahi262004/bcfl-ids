# Explainable and Trust-Aware Blockchain Federated Learning for Secure Intrusion Detection in Internet of Things

Final year project -- B.Tech CSE (IoT, Cybersecurity & Blockchain)

## Dataset

**N-BaIoT** (Meidan et al., 2018) -- real traffic from 9 commercial
IoT devices (doorbells, thermostat, baby monitor, security cameras,
webcam) infected with Mirai and BASHLITE/Gafgyt botnets.

Download: https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot

Extract into `data/raw/N-BaIoT/` so files sit directly inside, e.g.:
```
data/raw/N-BaIoT/1.benign.csv
data/raw/N-BaIoT/1.gafgyt.combo.csv
data/raw/N-BaIoT/1.mirai.ack.csv
...
```
Each device's data stays separate -- this maps directly onto your
federated learning clients (one real device = one real FL client,
no artificial partitioning needed).

## Pipeline stages (matches your PERT chart)

1. **Dataset Preparation** -- `fl/data_loader.py`
2. **Centralized IDS** (baseline) -- `fl/centralized_baseline.py`
3. **Federated IDS** -- `fl/client.py` + `fl/server.py`
4. Non-IID Evaluation, Poisoning Simulation, Trust Mechanism, Trust-Aware Aggregation -- builds on stage 3
5. **Blockchain trust layer** -- `blockchain/` + `bridge/`
6. **Dual-level XAI** -- `xai/`
7. **ZK Verification** -- future addition, see project notes
8. **Hardware testbed** -- `iot_layer/` (Phase 2)

## Setup

### Python side
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Blockchain side
```bash
cd blockchain
npm install
npx hardhat compile
```

## Running what's built so far (Stages 1-2, verified working)

**Important: all Python commands run as modules from the project root** (not by `cd`-ing into `fl/`), e.g. `python -m fl.data_loader`, not `python fl/data_loader.py` from inside `fl/`. This keeps file paths and imports consistent for all three of you.

1. Download N-BaIoT into `data/raw/N-BaIoT/` (see above)
2. Check it loads correctly:
   ```bash
   python -m fl.data_loader
   ```
   Expected: prints row/class counts for all 9 devices (devices 3 and 7 will show fewer classes -- they have no Mirai data, which is correct/expected, not a bug).
3. Train the centralized baseline:
   ```bash
   python -m fl.centralized_baseline
   ```
   This trains a RandomForest on a sample of the combined dataset and prints accuracy/precision/recall/F1. `SAMPLE_FRAC` in the script controls how much data is used -- start small (0.05), increase once you've confirmed timing is reasonable on your laptop (N-BaIoT has 7M+ rows total).
4. Explain some predictions:
   ```bash
   python -m xai.explain
   ```

## Not yet built (next sessions)

- Federated version of the baseline (`fl/client.py`/`fl/server.py` exist but need testing against real data once downloaded)
- Blockchain trust layer wired into aggregation
- Non-IID evaluation and poisoning simulation
- Physical hardware testbed (Phase 2)

## Team roles
- **FL / ML track:** `fl/`
- **Blockchain track:** `blockchain/`, `bridge/`
- **XAI + IoT track:** `xai/`, `iot_layer/`
