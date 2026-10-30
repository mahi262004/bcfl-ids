"""Helpers shared by the experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bcfl_ids.config import RESULTS_DIR


def save_json(name: str, payload) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=1, default=_plain))
    return path


def _plain(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def compact(result: dict) -> dict:
    """One run as stored in results: no weights, no per-round client records."""
    final = {k: v for k, v in result["final"].items()}
    return {
        "config": result["config"],
        "final": final,
        "curve": [{"round": h["round"], **h["metrics"]} for h in result["history"] if "metrics" in h],
        "false_reject_rate": result["false_reject_rate"],
        "attacker_reject_rate": result["attacker_reject_rate"],
        "reputation": result["reputation"],
        "elapsed_s": result["elapsed_s"],
    }
