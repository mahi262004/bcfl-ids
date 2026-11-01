# Demo guide

Written for the team presenting the project. It takes about 10 minutes. Exact measured figures are in [RESULTS.md](RESULTS.md); quote from there.

## Before you present

1. Start the app: `.venv/bin/python scripts/start_demo.py` (Windows: `.venv\Scripts\python scripts\start_demo.py`).
2. In a second terminal, run `.venv/bin/python scripts/preflight.py`. Every line should say PASS.
3. Close heavy apps. Keep the terminal running the app open.
4. Run one simulation before anyone arrives, so you know it works on this machine.

## The story in one sentence

Nine IoT devices train one botnet detector without sharing raw traffic. Even when a third of them are malicious, the model stays useful, every trust decision can be audited, and every alert can be explained.

## The walkthrough

### 1. Overview (1 minute)

Point at the five steps of one training round. Say plainly that all attacks are simulated and traffic is replayed from the N-BaIoT dataset, with no real malware.

### 2. Attack lab (3 minutes): the main moment

1. Select **sign flip**, **3 of 9**, keep **Compare with no defence** ticked, and press **Run simulation**.
2. Watch the two lines separate. The orange line (no defence) collapses and the blue line (proposed) holds.
3. Scroll to **Trust decisions**. Show that the attackers are rejected, with the reason written out.
4. Scroll to **Reputation over time**. Attackers fall below the probation line of 50 while honest clients climb.

If you have time, run **scaler poison**. Say that this attack strikes before training starts, by lying about feature statistics, and that one vote per feature stops it.

### 3. Blockchain audit (2 minutes)

1. The run is already selected. Point at the contract address and the transactions. Explain that these are real transactions on a local Hardhat chain.
2. Press **Verify records against chain**. It reports that all records match.
3. Press **Tamper with a record**. This secretly rewrites one rejected attacker's off-chain record to look clean.
4. Press **Verify** again. The tampering is caught, and the record is highlighted.
5. Press **Restore** and verify once more.

### 4. Detection (2 minutes)

1. Choose a device, leave **Mixed traffic**, and press **Replay traffic**.
2. Point at attacks flagged and false alarms. Say that these records were never used in training.
3. Click a record and read the explanation: bars to the right push towards attack.

### 5. Explainability and Evaluation (2 minutes)

- **Explainability:** deleting the top 10 features removes far more evidence than deleting 10 random ones, and the explanations stay stable across seeds.
- **Evaluation:** show the grid of attacks against defences, then the table showing how simple models already do well on attack versus benign. Saying this yourself earns credibility.

## Questions you should expect

**Why is macro F1 not close to 100%?**
The score covers 11 classes on later traffic the model never saw. Two pairs of families (gafgyt TCP and UDP, gafgyt junk and combo) are nearly identical with these features; even a random forest on pooled data cannot split them. Attack versus benign is above 99%.

**Why do these numbers differ from the Dissertation-I report?**
They come from the code in this repository, using a harder setup: 11 classes, a chronological split, more attacks and tighter thresholds. RESULTS.md lists every difference. Use these numbers, not the report's.

**Why not just use the 99% attack versus benign figure?**
A depth-3 decision tree nearly reaches it, so it says little on this dataset. Robustness under attack and family classification are the meaningful results.

**What does the blockchain actually add, if the server decides everything?**
It makes decisions tamper-evident after the fact. It does not decentralise the scoring: the server is still the only oracle, and this is one local node. An honest comparison is a signed log, which would give much of the same benefit here. A real deployment would need several independent nodes.

**Is it private?**
Raw traffic stays on the device. Feature statistics and plain model updates still reach the server, and those can leak information. There is no secure aggregation or differential privacy.

**Does the trust gate matter, given the trimmed mean?**
It depends on the attack, and the Evaluation grid has a column for each layer. Against loud attacks (sign flip, scaling) the gate clearly beats trimmed mean alone. Against Fang and the adaptive attacker, the trimmed mean does the work, because those updates pass the gate. Reputation adds no measurable accuracy; its value is the audit trail. The defence that nobody else's layer covers is the robust feature scale: without it, statistics poisoning takes macro F1 to about 0.02.

**Where does it break?**
Say this before you are asked. It is built for up to 3 attackers of 9. With 4, adaptive, Fang and statistics poisoning all break it. At 3, the adaptive attacker is the hardest case: macro F1 falls to about 0.60, and a plain median does better there.

**Did you find any flaws in your own defence?**
Yes. The first full evaluation showed Min-Max attackers could get 43% of honest updates rejected, leaving the defence worse than none. The size check was comparing updates with a vector the attackers could shrink. It was fixed, chosen on validation data, and the runs were repeated; RESULTS.md records this.

**Would it work on real devices?**
Not shown. It is one dataset from 2018, with no physical device tested.

**Why was the average of all traffic not used as the explanation baseline?**
The model classifies that average as an attack, so it is a poor reference for "normal". Average benign traffic is used instead.

## If something goes wrong

| What you see | What to do |
|---|---|
| "Chain: offline, using local ledger" | The demo still works. The launcher restarts the chain; run the simulation again. To be sure, stop the app with Ctrl-C and start it again. |
| "the local blockchain stopped during this run" | Wait a few seconds and run it again. |
| "this run's contract no longer exists" | The chain restarted. Run a new simulation. |
| Page does not load | Check the terminal. If port 8000 was busy, it prints a different address. |
| Everything is stuck | Ctrl-C, start again, and run the preflight check. |
