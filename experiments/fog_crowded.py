"""Fog nodes when attackers crowd onto one node, which fog_eval's seeds never produced.

Attackers are placed by hand: two on fog node A (1, 2, 4), or all three on it (1, 2, 3). With only three
devices on a node, two attackers control its median. Run `python -m experiments.fog_crowded`.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from bcfl_ids.config import RESULTS_DIR, SimConfig
from bcfl_ids.data import load_clients
from bcfl_ids.simulator import run_simulation

ATTACKS = ("sign_flip", "label_flip", "targeted_evasion", "lie", "fang_trim", "adaptive_gate")
PLACEMENTS = {"two on one node": [1, 2, 4], "three on one node": [1, 2, 3]}
SEEDS = (1, 2, 3)
METRICS = ("macro_f1", "macro_f1_merged", "attack_recall", "benign_recall", "worst_device_benign_recall")
_clients = None


def _init() -> None:
    global _clients
    torch.set_num_threads(2)
    _clients = load_clients(frac=0.05, split="test")


def _run(job: dict) -> dict:
    cfg = SimConfig(attack=job["attack"], n_attackers=3, attacker_ids=PLACEMENTS[job["placement"]], defence=job["defence"], seed=job["seed"], rounds=20)
    r = run_simulation(_clients, cfg, eval_every=20)
    return {**job, **{m: r["final"][m] for m in METRICS}, "false_reject_rate": r["false_reject_rate"], "attacker_reject_rate": r["attacker_reject_rate"]}


def main() -> None:
    jobs = [{"attack": a, "placement": p, "defence": d, "seed": s} for a in ATTACKS for p in PLACEMENTS for d in ("proposed", "fog") for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as pool:
        for future in as_completed([pool.submit(_run, j) for j in jobs]):
            rows.append(future.result())
            print(len(rows), "of", len(jobs), flush=True)
    cells = []
    for a in ATTACKS:
        for p in PLACEMENTS:
            for d in ("proposed", "fog"):
                group = [r for r in rows if (r["attack"], r["placement"], r["defence"]) == (a, p, d)]
                cell = {"attack": a, "placement": p, "defence": d, "macro_f1_by_seed": [r["macro_f1"] for r in sorted(group, key=lambda x: x["seed"])]}
                for m in (*METRICS, "false_reject_rate", "attacker_reject_rate"):
                    cell[m] = float(np.mean([r[m] for r in group]))
                cells.append(cell)
                print(a, p, d, {k: round(cell[k], 3) for k in ("macro_f1", "benign_recall", "worst_device_benign_recall", "false_reject_rate", "attacker_reject_rate")}, flush=True)
    (RESULTS_DIR / "fog_crowded.json").write_text(json.dumps({"seeds": list(SEEDS), "train_fraction": 0.05, "rounds": 20, "placements": PLACEMENTS, "cells": cells}, indent=1))


if __name__ == "__main__":
    main()
