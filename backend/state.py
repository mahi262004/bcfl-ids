"""Loads the demo bundle and runs federated simulations on one background thread."""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch

from bcfl_ids.chain import ChainLedger, audit, chain_available
from bcfl_ids.config import CLASSES, DEFENCE_LABELS, DEMO_DIR, RESULTS_DIR, RUNTIME_DIR, SimConfig
from bcfl_ids.data import signed_log
from bcfl_ids.explain import attack_score, integrated_gradients, parse_feature
from bcfl_ids.ledger import InMemoryLedger
from bcfl_ids.model import Detector, load_flat
from bcfl_ids.scaler import Scale
from bcfl_ids.simulator import ClientData, run_simulation
from bcfl_ids.storage import DEFAULT_IPFS_API, IpfsClient, UpdateStore


class Bundle:
    """Everything the demo needs, with no dependency on the raw dataset."""

    def __init__(self) -> None:
        data = np.load(DEMO_DIR / "demo_data.npz")
        self.clients = [
            ClientData(int(d), data[f"x_train_{d}"], data[f"y_train_{d}"], data[f"x_test_{d}"], data[f"y_test_{d}"])
            for d in data["devices"]
        ]
        self.by_device = {c.device: c for c in self.clients}
        model = np.load(DEMO_DIR / "model.npz")
        self.scale = Scale(model["mean"], model["std"])
        self.baseline = torch.from_numpy(model["benign_baseline"].astype(np.float32))
        self.model = Detector()
        load_flat(self.model, torch.from_numpy(model["weights"].astype(np.float32)))
        self.model.eval()
        self.features: list[str] = json.loads((DEMO_DIR / "features.json").read_text())
        self.meta: dict = json.loads((DEMO_DIR / "meta.json").read_text())

    @torch.no_grad()
    def predict_scaled(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = torch.softmax(self.model(torch.from_numpy(x)), dim=1)
        conf, pred = probs.max(1)
        return pred.numpy(), conf.numpy()

    def predict_raw(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_scaled(self.scale.apply(signed_log(raw.astype(np.float32))))

    def explain(self, device: int, index: int, top: int = 10) -> dict:
        c = self.by_device[device]
        x = torch.from_numpy(self.scale.apply(c.x_test[index : index + 1]))
        return self.explain_scaled(x, int(c.y_test[index]), top) | {"device": device, "index": index}

    def explain_scaled(self, x: torch.Tensor, true_class: int | None, top: int = 10) -> dict:
        with torch.no_grad():
            logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        # Explain the prediction against normal traffic. For a benign verdict, explain the runner-up
        # attack family, which shows what kept this record from being flagged.
        target_cls = pred if pred != 0 else int(probs[1:].argmax()) + 1
        target = torch.tensor([target_cls])
        attr = integrated_gradients(self.model, x, self.baseline, target, steps=128)[0]
        with torch.no_grad():
            score = float(attack_score(self.model(x), target))
            base = float(attack_score(self.model(self.baseline.unsqueeze(0)), target))
        order = attr.abs().argsort(descending=True)[:top].tolist()
        rows = []
        for i in order:
            stream, window, stat = parse_feature(self.features[i])
            rows.append({
                "feature": self.features[i], "stream": stream, "window": window, "statistic": stat,
                "attribution": float(attr[i]), "value": float(x[0, i]), "normal": float(self.baseline[i]),
            })
        return {
            "predicted": CLASSES[pred],
            "confidence": float(probs[pred]),
            "true_class": CLASSES[true_class] if true_class is not None else None,
            "explained_against": CLASSES[target_cls],
            "score": score,
            "baseline_score": base,
            "completeness_error": abs(float(attr.sum()) - (score - base)) / max(1e-6, abs(score - base)),
            "top": rows,
            "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(probs.tolist())},
        }


@dataclass
class Run:
    id: str
    config: dict
    label: str
    status: str = "queued"  # queued | running | finished | failed | stopped
    error: str | None = None
    rounds: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    original_records: list[dict] = field(default_factory=list)
    final: dict | None = None
    summary: dict | None = None
    ledger: object | None = None
    backend: str | None = None
    tampered: bool = False
    started: float | None = None
    elapsed: float | None = None


class Manager:
    def __init__(self, bundle: Bundle, rpc_url: str) -> None:
        self.bundle = bundle
        self.rpc_url = rpc_url
        self.runs: dict[str, Run] = {}
        self.order: list[str] = []
        self.active_job: dict | None = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.store = UpdateStore(RUNTIME_DIR / "updates", IpfsClient(os.environ.get("BCFL_IPFS_API", DEFAULT_IPFS_API)))

    def busy(self) -> bool:
        return self.active_job is not None and self.active_job["status"] == "running"

    def start(self, attack: str, n_attackers: int, defence: str, rounds: int, seed: int, compare: bool) -> dict:
        with self.lock:
            if self.busy():
                raise RuntimeError("A simulation is already running")
            defences = ["fedavg", "proposed"] if compare else [defence]
            runs = []
            for name in defences:
                run_id = uuid.uuid4().hex[:10]
                cfg = SimConfig(attack=attack, n_attackers=n_attackers, defence=name, rounds=rounds, seed=seed, batch_size=128)
                run = Run(run_id, cfg.to_dict(), DEFENCE_LABELS[name])
                self.runs[run_id] = run
                self.order.append(run_id)
                runs.append(run)
            job = {"id": uuid.uuid4().hex[:10], "status": "running", "run_ids": [r.id for r in runs]}
            self.active_job = job
            self.stop_flag.clear()
            threading.Thread(target=self._work, args=(job, runs), daemon=True).start()
            return job

    def _new_ledger(self):
        """Use the real contract when the chain is up. Otherwise say so and use the in-memory ledger."""
        if chain_available(self.rpc_url):
            return ChainLedger(self.rpc_url)
        return InMemoryLedger()

    def _work(self, job: dict, runs: list[Run]) -> None:
        for run in runs:
            if self.stop_flag.is_set():
                run.status = "stopped"
                continue
            self._execute(run)
        job["status"] = "finished"

    def _execute(self, run: Run) -> None:
        run.status = "running"
        run.started = time.time()
        try:
            ledger = self._new_ledger()
            run.ledger = ledger
            run.backend = ledger.backend
            cfg = SimConfig(**run.config)

            def on_round(row: dict) -> None:
                run.rounds.append(row)

            result = run_simulation(self.bundle.clients, cfg, ledger=ledger, on_round=on_round, should_stop=self.stop_flag.is_set)
            run.records = result["records"]
            run.original_records = copy.deepcopy(result["records"])
            run.final = result["final"]
            run.summary = {
                "false_reject_rate": result["false_reject_rate"],
                "attacker_reject_rate": result["attacker_reject_rate"],
                "reputation": result["reputation"],
                "attackers": result["config"]["attacker_ids"],
            }
            run.status = "stopped" if self.stop_flag.is_set() else "finished"
        except Exception as exc:  # A failed run must leave the server usable and say what went wrong.
            run.status = "failed"
            if isinstance(run.ledger, ChainLedger) and not chain_available(self.rpc_url):
                run.error = "the local blockchain stopped during this run. It restarts by itself, so please run it again"
            else:
                run.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        run.elapsed = time.time() - run.started

    def audit_run(self, run: Run) -> dict:
        return audit(run.records, run.ledger.entries)


def load_results() -> dict:
    """Pre-computed evaluation results produced by experiments/, shown as they were measured."""
    out = {}
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.glob("*.json"):
            out[path.stem] = json.loads(path.read_text())
    return out
