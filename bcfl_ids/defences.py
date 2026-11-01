"""Server-side trust gate and aggregation rules, all operating on flat update vectors."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .config import SCORE_PASS


@dataclass
class GateVerdict:
    cosine: float
    norm_ratio: float
    passed: bool
    score: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def quality_score(cos: float, ratio: float, cos_min: float, ratio_max: float) -> tuple[int, bool, str]:
    """Map the two checks to 0-100: a pass is always at or above SCORE_PASS, a fail always below it."""
    cos_ok = cos >= cos_min
    ratio_ok = ratio <= ratio_max
    if cos_ok and ratio_ok:
        cos_margin = (cos - cos_min) / max(1e-9, 1.0 - cos_min)
        ratio_margin = (ratio_max - ratio) / max(1e-9, ratio_max)
        margin = max(0.0, min(1.0, cos_margin, ratio_margin))
        return SCORE_PASS + int(round(margin * (100 - SCORE_PASS))), True, "within limits"
    reasons = []
    if not cos_ok:
        reasons.append(f"direction disagrees with the majority (cosine {cos:.3f} < {cos_min})")
    if not ratio_ok:
        reasons.append(f"update too large (norm ratio {ratio:.2f} > {ratio_max})")
    cos_miss = max(0.0, (cos_min - cos) / max(1e-9, 1.0 + cos_min))
    ratio_miss = max(0.0, (ratio - ratio_max) / max(1e-9, ratio_max))
    miss = min(1.0, max(cos_miss, ratio_miss))
    return int(round((SCORE_PASS - 1) * (1.0 - miss))), False, "; ".join(reasons)


def score_updates(updates: torch.Tensor, cos_min: float, ratio_max: float) -> list[GateVerdict]:
    """Check each update's direction against the median update, and its size against the median size."""
    median = updates.median(dim=0).values
    norms = updates.norm(dim=1).clamp_min(1e-12)
    cosines = (updates @ median) / (norms * median.norm().clamp_min(1e-12))
    # Relative to the typical update size. Using the norm of the median vector instead lets
    # colluding attackers shrink that vector, which makes honest updates look too large.
    ratios = norms / norms.median().clamp_min(1e-12)
    verdicts = []
    for cos, ratio in zip(cosines.tolist(), ratios.tolist()):
        score, passed, reason = quality_score(cos, ratio, cos_min, ratio_max)
        verdicts.append(GateVerdict(round(cos, 6), round(ratio, 6), passed, score, reason))
    return verdicts


def weighted_mean(updates: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (updates * (weights / weights.sum()).unsqueeze(1)).sum(0)


def trimmed_mean(updates: torch.Tensor, trim: int) -> torch.Tensor:
    """Drop `trim` values from each end per coordinate. Trim shrinks if few updates were accepted."""
    n = len(updates)
    k = min(trim, (n - 1) // 2)
    ordered = updates.sort(dim=0).values
    return ordered[k : n - k].mean(0)


def coordinate_median(updates: torch.Tensor) -> torch.Tensor:
    return updates.median(dim=0).values


def multi_krum(updates: torch.Tensor, f: int) -> torch.Tensor:
    """Multi-Krum (Blanchard et al., 2017): average the updates closest to their neighbours."""
    n = len(updates)
    f = min(f, max(0, (n - 3) // 2))
    neighbours = max(1, n - f - 2)
    dist = torch.cdist(updates, updates) ** 2
    dist.fill_diagonal_(float("inf"))
    scores = dist.sort(dim=1).values[:, :neighbours].sum(1)
    keep = scores.argsort()[: max(1, n - f)]
    return updates[keep].mean(0)


def aggregate(updates: torch.Tensor, weights: torch.Tensor, agg: str, trim: int, krum_f: int) -> torch.Tensor:
    if agg == "mean":
        return weighted_mean(updates, weights)
    if agg == "median":
        return coordinate_median(updates)
    if agg == "trimmed":
        return trimmed_mean(updates, trim)
    if agg == "multikrum":
        return multi_krum(updates, krum_f)
    raise ValueError(f"unknown aggregator {agg}")
