"""Federated training loop with attacks, trust gate, reputation ledger and explanation records.

All nine clients run in one process. The server logic is the same logic a networked deployment
would run (see flower_app.py), but this form is fast and exactly repeatable for experiments.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from . import attacks as atk
from .config import DEFENCES, DEVICE_IDS, FOG_OF, NUM_CLASSES, REP_PROBATION, SimConfig
from .defences import GateVerdict, aggregate, apply_round, gate_round, is_included, score_updates
from .ledger import InMemoryLedger, canonical_hash, sha256_hex
from .metrics import summarise
from .model import Detector, initial_weights, load_flat, to_flat
from .scaler import ClientStats, Scale, aggregate_pooled, aggregate_robust, client_stats, poison_stats


@dataclass
class ClientData:
    device: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray  # may be a memory-mapped array, so tests stay out of RAM
    y_test: np.ndarray


def choose_attackers(cfg: SimConfig) -> list[int]:
    if cfg.attacker_ids:
        return list(cfg.attacker_ids)[: cfg.n_attackers]
    if cfg.n_attackers == 0 or cfg.attack == "none":
        return []
    return sorted(random.Random(10_000 + cfg.seed).sample(list(DEVICE_IDS), cfg.n_attackers))


def shared_scale(clients: list[ClientData], cfg: SimConfig, attackers: list[int]) -> tuple[Scale, list[ClientStats]]:
    stats = []
    for c in clients:
        s = client_stats(c.x_train)
        if cfg.attack == "scaler_poison" and c.device in attackers:
            s = poison_stats(s)
        stats.append(s)
    if DEFENCES[cfg.defence].robust_scaler:
        return aggregate_robust(stats, cfg.trim), stats
    return aggregate_pooled(stats), stats


def class_weights(y: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency loss weights, from this client's own labels. They are never shared."""
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    weights = torch.zeros(NUM_CLASSES)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights.clamp_max(50.0)


def local_train(model: nn.Module, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: SimConfig, generator: torch.Generator) -> torch.Tensor:
    load_flat(model, w.clone())
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y))
    for _ in range(cfg.local_epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss_fn(model(x[idx]), y[idx]).backward()
            optimiser.step()
    return to_flat(model) - w


@torch.no_grad()
def evaluate_cube(model: nn.Module, clients: list[ClientData], scale: Scale, chunk: int = 200_000) -> np.ndarray:
    cube = np.zeros((len(DEVICE_IDS), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for c in clients:
        i = DEVICE_IDS.index(c.device)
        for s in range(0, len(c.x_test), chunk):
            x = torch.from_numpy(scale.apply(np.asarray(c.x_test[s : s + chunk])))
            pred = model(x).argmax(1).numpy()
            np.add.at(cube[i], (np.asarray(c.y_test[s : s + chunk]).astype(np.int64), pred), 1)
    return cube


def update_bytes(w: torch.Tensor) -> bytes:
    return w.numpy().astype(np.float32).tobytes()


def weights_hash(w: torch.Tensor) -> str:
    return sha256_hex(update_bytes(w))


def falsify(verdicts: list[GateVerdict], devices: list[int], attackers: list[int]) -> list[GateVerdict]:
    """Demo of a lying server: attackers' failed updates are recorded and used as if they were clean."""
    return [GateVerdict(0.97, 1.02, True, 90, "within limits") if d in attackers and not v.passed else v for v, d in zip(verdicts, devices)]


def run_simulation(
    clients: list[ClientData],
    cfg: SimConfig,
    ledger=None,
    store=None,
    on_round: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    eval_every: int = 1,
) -> dict:
    """Train under `cfg` and return history, final metrics, trust records and the final weights."""
    started = time.time()
    defence = DEFENCES[cfg.defence]
    attackers = choose_attackers(cfg)
    ledger = ledger if ledger is not None else InMemoryLedger()
    for c in clients:
        ledger.register(c.device)

    scale, _ = shared_scale(clients, cfg, attackers)
    train_x = {c.device: torch.from_numpy(scale.apply(np.asarray(c.x_train))) for c in clients}
    train_y = {c.device: torch.from_numpy(np.asarray(c.y_train).astype(np.int64)) for c in clients}

    torch.manual_seed(cfg.seed)
    model = Detector()
    w = initial_weights(cfg.seed)
    sizes = torch.tensor([float(len(train_y[c.device])) for c in clients])
    history: list[dict] = []
    records: list[dict] = []
    false_rejects = attacker_rejects = attacker_rounds = honest_rounds = 0

    for round_no in range(1, cfg.rounds + 1):
        if should_stop and should_stop():
            break
        attack_gen = torch.Generator().manual_seed(cfg.seed * 7919 + round_no)
        updates = {}
        poison_updates = []
        for c in clients:
            gen = torch.Generator().manual_seed(cfg.seed * 1000 + round_no * 10 + c.device)
            malicious = c.device in attackers
            y = train_y[c.device]
            if malicious and cfg.attack in atk.DATA_ATTACKS:
                y = atk.poison_labels(y, cfg.attack)
            update = local_train(model, w, train_x[c.device], y, cfg, gen)
            if malicious and cfg.attack not in atk.CRAFTED_ATTACKS:
                update = atk.post_train_update(cfg.attack, update, w, attack_gen)
            updates[c.device] = update
            if malicious and cfg.attack == "adaptive_gate":
                flipped = atk.poison_labels(train_y[c.device], "label_flip")
                poison_updates.append(local_train(model, w, train_x[c.device], flipped, cfg, gen))

        if attackers and cfg.attack in atk.CRAFTED_ATTACKS:
            honest = torch.stack([updates[c.device] for c in clients if c.device not in attackers])
            poison = torch.stack(poison_updates).mean(0) if poison_updates else honest.mean(0)
            crafted = atk.craft_update(cfg.attack, honest, len(attackers), len(clients), poison, cfg.cos_min, cfg.ratio_max, attack_gen)
            for a in attackers:
                updates[a] = crafted

        stack = torch.stack([updates[c.device] for c in clients])
        devices = [c.device for c in clients]
        verdicts = gate_round(stack, devices, defence, cfg.cos_min, cfg.ratio_max)
        if cfg.dishonest_server:
            verdicts = falsify(verdicts, devices, attackers)

        keep = []
        round_records = []
        for i, c in enumerate(clients):
            v = verdicts[i]
            rep_before = ledger.reputation(c.device)
            on_probation = defence.reputation and rep_before < REP_PROBATION
            included = is_included(v, defence, rep_before)
            record = {
                "round": round_no,
                "client": c.device,
                "fog": FOG_OF[c.device] if defence.fog else None,
                "cosine": v.cosine,
                "norm_ratio": v.norm_ratio,
                "score": v.score,
                "gate_passed": v.passed,
                "on_probation": on_probation,
                "included": included,
                "reason": v.reason if not on_probation or not v.passed else "passed the gate, but on probation",
                "update_hash": weights_hash(updates[c.device]),
            }
            record_hash = canonical_hash(record)
            if store is not None:
                store.put(update_bytes(updates[c.device]))
            entry = ledger.submit(round_no, c.device, v.score, record["update_hash"], record_hash)
            record = {**record, "record_hash": record_hash, "rep_before": entry["rep_before"], "rep_after": entry["rep_after"], "malicious": c.device in attackers}
            round_records.append(record)
            if included:
                keep.append(i)
            if c.device in attackers:
                attacker_rounds += 1
                attacker_rejects += not v.passed
            else:
                honest_rounds += 1
                false_rejects += not v.passed

        if not keep:  # Every update failed: leave the global model unchanged this round.
            step = torch.zeros_like(w)
        else:
            step = aggregate(stack[keep], sizes[keep], defence.agg, cfg.trim, cfg.krum_f)
        w = w + step
        if not torch.isfinite(w).all():  # A diverged model is a failed defence, not a crash.
            w = torch.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        model_hash = weights_hash(w)
        ledger.anchor_model(round_no, model_hash)
        records.extend(round_records)

        row = {"round": round_no, "accepted": len(keep), "model_hash": model_hash, "clients": round_records, "attackers": attackers}
        if round_no % eval_every == 0 or round_no == cfg.rounds:
            load_flat(model, w)
            metrics = summarise(evaluate_cube(model, clients, scale), attackers)
            row["metrics"] = {k: metrics[k] for k in ("accuracy", "macro_f1", "mcc", "attack_recall", "benign_recall", "false_positive_rate", "target_family_evasion")}
        history.append(row)
        if on_round:
            on_round(row)

    load_flat(model, w)
    final = summarise(evaluate_cube(model, clients, scale), attackers)
    return {
        "config": {**cfg.to_dict(), "attacker_ids": attackers},
        "final": final,
        "history": history,
        "records": records,
        "reputation": {c.device: ledger.reputation(c.device) for c in clients},
        "false_reject_rate": false_rejects / max(1, honest_rounds),
        "attacker_reject_rate": attacker_rejects / attacker_rounds if attacker_rounds else None,
        "weights": w,
        "scale": scale,
        "elapsed_s": time.time() - started,
        "ledger_backend": ledger.backend,
    }
