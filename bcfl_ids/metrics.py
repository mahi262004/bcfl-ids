"""Metrics, all derived from one confusion cube: (device, true class, predicted class)."""
from __future__ import annotations

import math

import numpy as np

from .config import BENIGN, CLASSES, DEVICE_IDS, TARGET_FAMILY


def _binary(c: np.ndarray) -> dict:
    tn = int(c[BENIGN, BENIGN])
    fp = int(c[BENIGN].sum() - tn)
    fn = int(c[1:, BENIGN].sum())
    tp = int(c[1:].sum() - fn)
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "attack_recall": tp / (tp + fn) if tp + fn else None,
        "benign_recall": tn / (tn + fp) if tn + fp else None,
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "binary_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _multiclass(c: np.ndarray) -> dict:
    support = c.sum(1)
    predicted = c.sum(0)
    correct = np.diag(c)
    per_class = {}
    f1s = []
    for i, name in enumerate(CLASSES):
        if support[i] == 0:
            continue
        precision = correct[i] / predicted[i] if predicted[i] else 0.0
        recall = correct[i] / support[i]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(support[i])}
    return {
        "accuracy": float(correct.sum() / max(1, c.sum())),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
    }


def merge_lookalikes(c: np.ndarray) -> np.ndarray:
    """Fold gafgyt_junk into gafgyt_combo and gafgyt_tcp into gafgyt_udp.

    The data audit shows even a pooled random forest cannot split these pairs on later traffic.
    """
    m = c.copy()
    for src, dst in ((CLASSES.index("gafgyt_junk"), CLASSES.index("gafgyt_combo")), (CLASSES.index("gafgyt_tcp"), CLASSES.index("gafgyt_udp"))):
        m[dst] += m[src]
        m[:, dst] += m[:, src]
        m[src] = 0
        m[:, src] = 0
    return m


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(cube: np.ndarray, attacker_ids: list[int] | None = None) -> dict:
    """Headline numbers on honest devices, with every device also reported separately."""
    attackers = set(attacker_ids or [])
    honest_idx = [i for i, d in enumerate(DEVICE_IDS) if d not in attackers]
    honest = cube[honest_idx].sum(0)
    out = {**_multiclass(honest), **_binary(honest)}
    out["confusion"] = honest.tolist()
    out["macro_f1_merged"] = _multiclass(merge_lookalikes(honest))["macro_f1"]
    # How often traffic of the targeted family gets through as benign.
    target = honest[TARGET_FAMILY]
    out["target_family_evasion"] = float(target[BENIGN] / target.sum()) if target.sum() else None
    devices = {}
    for i, d in enumerate(DEVICE_IDS):
        multi = _multiclass(cube[i])
        b = _binary(cube[i])
        devices[str(d)] = {
            "macro_f1": multi["macro_f1"],
            "accuracy": multi["accuracy"],
            "benign_recall": b["benign_recall"],
            "benign_recall_ci": wilson(b["tn"], b["tn"] + b["fp"]),
            "attack_recall": b["attack_recall"],
            "benign_test_rows": b["tn"] + b["fp"],
            "malicious": d in attackers,
        }
    out["devices"] = devices
    honest_recall = [v["benign_recall"] for v in devices.values() if not v["malicious"] and v["benign_recall"] is not None]
    out["worst_device_benign_recall"] = min(honest_recall) if honest_recall else None
    return out
