"""Federated feature scaling from aggregate statistics only; raw records never leave a client."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientStats:
    n: float
    s: np.ndarray  # per-feature sum
    q: np.ndarray  # per-feature sum of squares


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -10.0, 10.0).astype(np.float32)


def client_stats(x: np.ndarray, chunk: int = 200_000) -> ClientStats:
    s = np.zeros(x.shape[1], dtype=np.float64)
    q = np.zeros(x.shape[1], dtype=np.float64)
    for i in range(0, len(x), chunk):
        b = np.asarray(x[i : i + chunk], dtype=np.float64)
        s += b.sum(0)
        q += (b * b).sum(0)
    return ClientStats(float(len(x)), s, q)


def _safe_std(var: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(var, 0.0))
    std[std < 1e-8] = 1.0  # constant features are left unscaled
    return std


def aggregate_pooled(stats: list[ClientStats]) -> Scale:
    """Exactly what a StandardScaler on pooled data would fit, but one lying client can break it."""
    n = sum(c.n for c in stats)
    mean = sum(c.s for c in stats) / n
    var = sum(c.q for c in stats) / n - mean**2
    return Scale(mean, _safe_std(var))


def _trimmed(values: np.ndarray, trim: int) -> np.ndarray:
    """Coordinate-wise trimmed mean over clients (axis 0)."""
    ordered = np.sort(values, axis=0)
    return ordered[trim : len(values) - trim].mean(axis=0)


def aggregate_robust(stats: list[ClientStats], trim: int = 3) -> Scale:
    """Each client gets one vote per feature, so up to `trim` liars cannot leave the honest range."""
    means = np.stack([c.s / c.n for c in stats])
    second = np.stack([c.q / c.n for c in stats])
    mean = _trimmed(means, trim)
    var = _trimmed(second, trim) - mean**2
    return Scale(mean, _safe_std(var))


def poison_stats(c: ClientStats, factor: float = 1e4) -> ClientStats:
    """What a lying client reports: inflated sums, so the shared mean and deviation are corrupted."""
    return ClientStats(c.n, c.s * factor, c.q * factor * factor)
