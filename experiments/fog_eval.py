"""Does gating on three fog nodes hold up as well as one central gate?

Same settings as the main grid: 5% training sample, 3 seeds, every attack with 3 of 9 attackers, and no
attacker. Attackers are chosen from the seed, so some fog nodes can hold two or three of them. The attacks
are the existing ones; none is tailored to fog nodes. Run `python -m experiments.fog_eval`.
"""
from __future__ import annotations

import numpy as np

from bcfl_ids.config import ATTACKS, FOG_OF, RESULTS_DIR, SimConfig
from bcfl_ids.data import load_clients
from bcfl_ids.simulator import choose_attackers, run_simulation

from .common import save_json

METRICS = ("macro_f1", "macro_f1_merged", "attack_recall", "benign_recall", "worst_device_benign_recall")


def main() -> None:
    clients = load_clients(frac=0.05, split="test")
    out: dict = {"seeds": [1, 2, 3], "train_fraction": 0.05, "rounds": 20, "cells": []}
    for attack in ATTACKS:
        n = 0 if attack == "none" else 3
        for defence in ("proposed", "fog"):
            rows = []
            for seed in out["seeds"]:
                cfg = SimConfig(attack=attack, n_attackers=n, defence=defence, seed=seed, rounds=20)
                r = run_simulation(clients, cfg, eval_every=20)
                attackers = choose_attackers(cfg)
                groups = [sum(FOG_OF[a] == g for a in attackers) for g in "ABC"]
                rows.append({**{m: r["final"][m] for m in METRICS}, "false_reject_rate": r["false_reject_rate"], "attacker_reject_rate": r["attacker_reject_rate"], "most_attackers_in_one_fog": max(groups), "attackers": attackers})
            cell = {"attack": attack, "defence": defence, "runs": rows}
            for m in (*METRICS, "false_reject_rate", "attacker_reject_rate"):
                values = [x[m] for x in rows if x[m] is not None]
                cell[m] = float(np.mean(values)) if values else None
            out["cells"].append(cell)
            print(attack, defence, {k: (round(cell[k], 3) if cell[k] is not None else None) for k in ("macro_f1", "benign_recall", "worst_device_benign_recall", "false_reject_rate", "attacker_reject_rate")}, [x["most_attackers_in_one_fog"] for x in rows], flush=True)
            save_json("fog_eval.json", out)
    print("saved", RESULTS_DIR / "fog_eval.json")


if __name__ == "__main__":
    main()
