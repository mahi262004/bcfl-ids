"""Tune training settings and gate thresholds on the VALIDATION split and calibration seeds only.

The test split and the evaluation seeds stay untouched until the final grid.
"""
from __future__ import annotations

import argparse

import numpy as np

from bcfl_ids.config import SimConfig
from bcfl_ids.data import load_clients
from bcfl_ids.simulator import run_simulation

from .common import compact, save_json

CALIBRATION_SEEDS = (101, 102)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05)
    args = p.parse_args()
    clients = load_clients(frac=args.frac, split="val", test_frac=0.25)
    out = {"frac": args.frac, "seeds": CALIBRATION_SEEDS}

    # 1. How many rounds does honest FedAvg need?
    curves = []
    for seed in CALIBRATION_SEEDS:
        res = run_simulation(clients, SimConfig(defence="fedavg", rounds=25, seed=seed, cos_min=-2, ratio_max=1e9))
        curves.append(compact(res)["curve"])
        print("fedavg", seed, "elapsed", round(res["elapsed_s"], 1), [round(c["macro_f1"], 3) for c in curves[-1]], flush=True)
    out["fedavg_curves"] = curves

    # 2. What do honest updates look like to the gate, when nothing is being rejected?
    cos, ratio = [], []
    for seed in CALIBRATION_SEEDS:
        res = run_simulation(clients, SimConfig(defence="fedavg", rounds=15, seed=seed))
        cos += [r["cosine"] for r in res["records"]]
        ratio += [r["norm_ratio"] for r in res["records"]]
    pct = [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100]
    out["honest_cosine_percentiles"] = dict(zip(map(str, pct), np.percentile(cos, pct).tolist()))
    out["honest_ratio_percentiles"] = dict(zip(map(str, pct), np.percentile(ratio, pct).tolist()))
    out["honest_client_rounds"] = len(cos)
    print("cos", out["honest_cosine_percentiles"], flush=True)
    print("ratio", out["honest_ratio_percentiles"], flush=True)

    # 3. And what attacked updates look like, to see whether the two populations separate.
    out["attacked"] = {}
    for attack in ("sign_flip", "scaling", "label_flip", "random_weights", "lie", "minmax", "fang_trim", "adaptive_gate"):
        res = run_simulation(clients, SimConfig(attack=attack, n_attackers=3, defence="fedavg", rounds=8, seed=101))
        bad = [r for r in res["records"] if r["malicious"]]
        out["attacked"][attack] = {
            "cos_median": float(np.median([r["cosine"] for r in bad])),
            "ratio_median": float(np.median([r["norm_ratio"] for r in bad])),
            "macro_f1_fedavg": res["final"]["macro_f1"],
        }
        print(attack, out["attacked"][attack], flush=True)
    save_json("calibration.json", out)


if __name__ == "__main__":
    main()
