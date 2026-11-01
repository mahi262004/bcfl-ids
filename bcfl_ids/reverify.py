"""Re-check a run's decisions from its stored files, without trusting anything the server reported.

Comparing records with the chain only shows that nothing changed after the event. A server that lied
when it wrote to the chain passes that test. This fetches every update by its on-chain hash, reruns the
gate and the aggregation, and compares the outcome with what the chain says happened.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import DEFENCES, SCORE_PASS
from .defences import apply_round, gate_round, is_included
from .model import initial_weights
from .storage import FileMismatch, FileMissing, cid_from_hash

MODEL_TOLERANCE = 1e-5  # floating-point sums may differ slightly between machines


def _load(store, hex_hash: str) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(store.get(cid_from_hash(hex_hash)), dtype=np.float32).copy())


def recheck_run(config: dict, entries: list[dict], models: list[dict], sizes: dict, store) -> dict:
    defence = DEFENCES[config["defence"]]
    by_round: dict[int, list[dict]] = {}
    for entry in entries:
        by_round.setdefault(entry["round"], []).append(entry)
    anchored = {m["round"]: m["model_hash"] for m in models}

    problems: list[dict] = []
    rows: list[dict] = []
    files_checked = 0

    def problem(round_no: int, client, text: str) -> None:
        problems.append({"round": round_no, "client": client, "problem": text})

    previous = initial_weights(config["seed"])
    for round_no in sorted(by_round):
        round_entries = by_round[round_no]
        devices = [e["client"] for e in round_entries]
        row = {"round": round_no, "files": 0, "files_expected": len(devices), "decisions_differ": None, "model": "not checked"}
        rows.append(row)
        updates = []
        for entry in round_entries:
            try:
                updates.append(_load(store, entry["update_hash"]))
                row["files"] += 1
                files_checked += 1
            except FileMismatch:
                problem(round_no, entry["client"], "the stored update does not match its on-chain hash")
            except FileMissing:
                problem(round_no, entry["client"], "the stored update is missing")

        expected = anchored.get(round_no)
        model = None
        if expected is None:
            problem(round_no, None, "no model hash on the chain for this round")
        else:
            try:
                model = _load(store, expected)
                files_checked += 1
            except FileMismatch:
                problem(round_no, None, "the stored global model does not match its on-chain hash")
            except FileMissing:
                problem(round_no, None, "the stored global model is missing")

        if len(updates) == len(devices) and devices:
            stack = torch.stack(updates)
            verdicts = gate_round(stack, devices, defence, config["cos_min"], config["ratio_max"])
            differ = 0
            keep = []
            for i, (entry, verdict) in enumerate(zip(round_entries, verdicts)):
                if is_included(verdict, defence, entry["rep_before"]):
                    keep.append(i)
                recorded_pass = entry["score"] >= SCORE_PASS
                if entry["score"] != verdict.score or recorded_pass != verdict.passed:
                    differ += 1
                    says = "accepted" if recorded_pass else "rejected"
                    finds = "accepted" if verdict.passed else "rejected"
                    problem(round_no, entry["client"], f"the chain says score {entry['score']} ({says}), but the stored update gives {verdict.score} ({finds})")
            row["decisions_differ"] = differ
            if model is not None:
                weights = torch.tensor([float(sizes[str(d)] if str(d) in sizes else sizes[d]) for d in devices])
                recomputed = apply_round(previous, stack, keep, weights, defence, config["trim"], config["krum_f"])
                gap = float((recomputed - model).abs().max())
                row["model"] = "matches" if gap <= MODEL_TOLERANCE else "differs"
                row["model_gap"] = gap
                if gap > MODEL_TOLERANCE:
                    problem(round_no, None, f"rerunning the aggregation does not give the recorded global model (largest weight difference {gap:.3g})")
        if model is not None:
            previous = model  # So one bad round does not make every later round look wrong.
    return {
        "valid": not problems and bool(rows), "rounds_checked": len(rows), "files_checked": files_checked,
        "decisions_differ": sum(r["decisions_differ"] or 0 for r in rows), "problems": problems, "rounds": rows,
    }
