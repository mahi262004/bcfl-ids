"""Collapse grid.jsonl into the summary the dashboard and the results document read."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from bcfl_ids.config import ATTACKS, DEFENCE_LABELS, DEFENCES, RESULTS_DIR

from .common import save_json

METRICS = ("macro_f1", "macro_f1_merged", "accuracy", "binary_accuracy", "attack_recall", "benign_recall", "mcc", "worst_device_benign_recall", "target_family_evasion")


def _stat(values) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "min": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values))}


def main() -> None:
    runs = [json.loads(line) for line in (RESULTS_DIR / "grid.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for r in runs:
        c = r["config"]
        groups[(c["attack"], c["n_attackers"], c["defence"])].append(r)

    def cell(key, rs):
        attack, n, defence = key
        out = {"attack": attack, "n_attackers": n, "defence": defence, "runs": len(rs)}
        for m in METRICS:
            out[m] = _stat([r["final"].get(m) for r in rs])
        out["false_reject_rate"] = _stat([r["false_reject_rate"] for r in rs])
        out["attacker_reject_rate"] = _stat([r["attacker_reject_rate"] for r in rs])
        out["elapsed_s"] = float(np.mean([r["elapsed_s"] for r in rs]))
        return out

    all_cells = [cell(k, v) for k, v in groups.items()]
    # The matrix on the dashboard: attack-free row, then every attack with 3 of 9 clients malicious.
    matrix = [c for c in all_cells if c["n_attackers"] == 3 or c["attack"] == "none"]
    clean = groups[("none", 0, "proposed")]
    for c in matrix:
        if c["attack"] == "none" and c["defence"] == "proposed":
            mean_conf = np.mean([r["final"]["confusion"] for r in clean], axis=0)
            c["confusion"] = mean_conf.tolist()
            devices = {}
            for dev in clean[0]["final"]["devices"]:
                entry = clean[0]["final"]["devices"][dev]
                # Pool the rows over seeds for the interval on this device.
                recalls = [r["final"]["devices"][dev]["benign_recall"] for r in clean]
                intervals = [r["final"]["devices"][dev]["benign_recall_ci"] for r in clean]
                devices[dev] = {
                    **entry,
                    "benign_recall": float(np.mean(recalls)),
                    "benign_recall_ci": [float(np.mean([i[0] for i in intervals])), float(np.mean([i[1] for i in intervals]))],
                }
            c["devices"] = devices
            c["family"] = clean[0]["final"]["per_class"]

    def worst(defence):
        rows = [c for c in matrix if c["defence"] == defence and c["attack"] != "none"]
        w = min(rows, key=lambda c: c["macro_f1"]["mean"])
        return {"attack": w["attack"], "value": w["macro_f1"]["mean"]}

    conf_total = int(np.sum(clean[0]["final"]["confusion"]))
    payload = {
        "meta": {
            "seeds": len({r["config"]["seed"] for r in runs}),
            "train_fraction": runs[0]["frac"],
            "rounds": runs[0]["config"]["rounds"],
            "test_rows": conf_total,
            "split": "chronological test",
            "total_runs": len(runs),
        },
        "attacks": [a for a in ATTACKS if any(c["attack"] == a for c in matrix)],
        "defences": [d for d in DEFENCES if any(c["defence"] == d for c in matrix)],
        "defence_labels": DEFENCE_LABELS,
        "cells": matrix,
        "scaling_attackers": [c for c in all_cells if c["defence"] in ("fedavg", "proposed") and c["attack"] != "none"],
        "headline": {"fedavg_worst": worst("fedavg"), "proposed_worst": worst("proposed")},
        "limits": [],
    }
    existing = RESULTS_DIR / "limits.json"
    if existing.exists():
        payload["limits"] = json.loads(existing.read_text())
    save_json("grid_summary.json", payload)
    print(json.dumps(payload["headline"], indent=1))


if __name__ == "__main__":
    main()
