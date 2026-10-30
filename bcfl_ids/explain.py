"""Integrated Gradients for the detector, plus the checks that show whether an explanation holds up."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import BENIGN


def attack_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """How strongly the model prefers `target` over benign. Positive means flagged as `target`."""
    idx = torch.arange(len(logits))
    return logits[idx, target] - logits[idx, BENIGN]


def integrated_gradients(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor, steps: int = 128) -> torch.Tensor:
    """Sundararajan et al. (2017), midpoint rule. x: (n, d), baseline: (d,), target: (n,)."""
    total = torch.zeros_like(x)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        point = (baseline + alpha * (x - baseline)).requires_grad_(True)
        score = attack_score(model(point), target).sum()
        total += torch.autograd.grad(score, point)[0]
    return (x - baseline) * total / steps


def completeness_error(model: nn.Module, x, baseline, target, attributions) -> np.ndarray:
    """|sum(IG) - (F(x) - F(baseline))|, relative. This checks the integration, not faithfulness."""
    with torch.no_grad():
        fx = attack_score(model(x), target)
        fb = attack_score(model(baseline.expand_as(x)), target)
    gap = fx - fb
    return ((attributions.sum(1) - gap).abs() / gap.abs().clamp_min(1e-6)).numpy()


def deletion_curve(model, x, baseline, target, attributions, ks, seed: int = 0) -> list[dict]:
    """Replace k features with the baseline: most-attributed first, versus at random."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        fx = attack_score(model(x), target)
        fb = attack_score(model(baseline.expand_as(x)), target)
        gap = (fx - fb).clamp_min(1e-6)
        order = attributions.argsort(dim=1, descending=True)
        rows = []
        for k in ks:
            top = x.clone()
            rnd = x.clone()
            top.scatter_(1, order[:, :k], baseline[order[:, :k]])
            rand_idx = torch.rand(x.shape, generator=generator).argsort(dim=1)[:, :k]
            rnd.scatter_(1, rand_idx, baseline[rand_idx])
            ft, fr = attack_score(model(top), target), attack_score(model(rnd), target)
            rows.append({
                "k": k,
                "evidence_removed_top": float(((fx - ft) / gap).mean()),
                "evidence_removed_random": float(((fx - fr) / gap).mean()),
                "flipped_top": float((ft < 0).float().mean()),
                "flipped_random": float((fr < 0).float().mean()),
            })
    return rows


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def parse_feature(name: str) -> tuple[str, str, str]:
    """E.g. 'HH_jit_L0.1_mean' -> ('HH_jit', 'L0.1', 'mean')."""
    parts = name.split("_")
    idx = next(i for i, p in enumerate(parts) if p.startswith("L") and p[1:2].isdigit())
    return "_".join(parts[:idx]), parts[idx], "_".join(parts[idx + 1 :])


def group_attribution(mean_abs: np.ndarray, names: list[str]) -> dict:
    """Share of attribution per traffic stream, per time window and per statistic."""
    stream: dict[str, float] = {}
    window: dict[str, float] = {}
    stat: dict[str, float] = {}
    for value, name in zip(mean_abs.tolist(), names):
        s, w, st = parse_feature(name)
        stream[s] = stream.get(s, 0.0) + value
        window[w] = window.get(w, 0.0) + value
        stat[st] = stat.get(st, 0.0) + value
    total = float(mean_abs.sum()) or 1.0

    def share(d: dict[str, float]) -> dict[str, float]:
        return {k: v / total for k, v in sorted(d.items(), key=lambda kv: -kv[1])}

    return {"stream": share(stream), "window": share(window), "statistic": share(stat)}
