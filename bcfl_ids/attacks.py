"""Simulated malicious clients. No real malware is used: attacks act on labels, updates or statistics.

Crafted-update attacks are white-box: they see every honest update in the round. That is
stronger than a real attacker, so a defence that holds here has been tested conservatively.
"""
from __future__ import annotations

from statistics import NormalDist

import torch

from .config import BENIGN, TARGET_FAMILY
from .defences import score_updates

DATA_ATTACKS = {"label_flip", "scaling", "targeted_evasion"}
CRAFTED_ATTACKS = {"lie", "minmax", "fang_trim", "adaptive_gate"}


def poison_labels(y: torch.Tensor, attack: str) -> torch.Tensor:
    y = y.clone()
    if attack in ("label_flip", "scaling"):
        benign = y == BENIGN
        y[~benign] = BENIGN
        y[benign] = TARGET_FAMILY
    elif attack == "targeted_evasion":
        y[y == TARGET_FAMILY] = BENIGN
    return y


def lie_z(n: int, m: int) -> float:
    """Baruch et al. (2019): how far a coordinate may shift while staying with the majority."""
    s = n // 2 + 1 - m
    p = min(0.999, max(0.5, (n - m - s) / (n - m)))
    return NormalDist().inv_cdf(p)


def craft_lie(honest: torch.Tensor, n: int, m: int) -> torch.Tensor:
    return honest.mean(0) - max(lie_z(n, m), 0.25) * honest.std(0, unbiased=False)


def craft_minmax(honest: torch.Tensor) -> torch.Tensor:
    """Shejwalkar & Houmansadr (2021): largest reversed shift no farther than honest clients are apart."""
    mean = honest.mean(0)
    direction = -mean / mean.norm().clamp_min(1e-12)
    limit = torch.cdist(honest, honest).max()
    low, high = 0.0, 50.0
    for _ in range(30):
        gamma = (low + high) / 2
        if torch.cdist((mean + gamma * direction).unsqueeze(0), honest).max() <= limit:
            low = gamma
        else:
            high = gamma
    return mean + low * direction


def craft_fang(honest: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Fang et al. (2020): push each coordinate against its trend, beyond the honest extreme."""
    trend = honest.mean(0).sign()
    low, high = honest.min(0).values, honest.max(0).values
    span = (high - low).clamp_min(1e-9)
    noise = torch.rand(honest.shape[1], generator=generator)
    return torch.where(trend > 0, low - noise * span, high + noise * span)


def craft_adaptive(honest: torch.Tensor, poison: torch.Tensor, cos_min: float, ratio_max: float, m: int) -> torch.Tensor:
    """Blend honest and poisoned updates, keeping the strongest mix that still passes the gate."""
    mean = honest.mean(0)
    direction = poison / poison.norm().clamp_min(1e-12)
    strength = 2.0 * mean.norm()
    best = mean
    for alpha in [i / 20 for i in range(1, 21)]:
        candidate = (1 - alpha) * mean + alpha * strength * direction
        trial = torch.cat([honest, candidate.unsqueeze(0).repeat(m, 1)])
        if score_updates(trial, cos_min, ratio_max)[-1].passed:
            best = candidate
    return best


def craft_update(attack: str, honest, m: int, n: int, poison, cos_min, ratio_max, generator) -> torch.Tensor:
    if attack == "lie":
        return craft_lie(honest, n, m)
    if attack == "minmax":
        return craft_minmax(honest)
    if attack == "fang_trim":
        return craft_fang(honest, generator)
    if attack == "adaptive_gate":
        return craft_adaptive(honest, poison, cos_min, ratio_max, m)
    raise ValueError(f"not a crafted attack: {attack}")


def post_train_update(attack: str, update: torch.Tensor, global_w: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Attacks that alter one client's update after its local training."""
    if attack == "random_weights":
        return torch.normal(0.0, 0.5, size=global_w.shape, generator=generator) - global_w
    if attack == "sign_flip":
        return -3.0 * update
    if attack == "scaling":
        return 10.0 * update
    if attack == "targeted_evasion":
        return 3.0 * update
    return update
