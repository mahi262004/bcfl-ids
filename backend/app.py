"""REST API and static hosting for the dashboard. Binds to localhost only; see scripts/start_demo.py."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from bcfl_ids.chain import DEFAULT_RPC, chain_available
from bcfl_ids.config import ATTACK_INFO, CLASSES, DEFENCE_LABELS, DEFENCES, DEVICES, FOG_GROUPS, FOG_OF, NUM_FEATURES, ROOT
from bcfl_ids.ledger import sha256_hex
from bcfl_ids.reverify import recheck_run
from bcfl_ids.storage import FileMismatch, FileMissing, cid_from_hash, cid_for

from .state import Bundle, Manager, Run, load_results

RPC_URL = os.environ.get("BCFL_RPC_URL", DEFAULT_RPC)
FRONTEND = ROOT / "frontend"

torch.set_num_threads(min(4, os.cpu_count() or 1))
bundle = Bundle()
manager = Manager(bundle, RPC_URL)
results = load_results()

app = FastAPI(title="BCFL-IDS", version="1.0")


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


class SimulationRequest(BaseModel):
    attack: str = "none"
    n_attackers: int = Field(0, ge=0, le=4)
    defence: str = "proposed"
    rounds: int = Field(10, ge=1, le=30)
    seed: int = Field(1, ge=0, le=1_000_000)
    compare: bool = False
    dishonest_server: bool = False

    @field_validator("attack")
    @classmethod
    def known_attack(cls, v: str) -> str:
        if v not in ATTACK_INFO:
            raise ValueError(f"unknown attack: {v}")
        return v

    @field_validator("defence")
    @classmethod
    def known_defence(cls, v: str) -> str:
        if v not in DEFENCES:
            raise ValueError(f"unknown defence: {v}")
        return v


class ReplayRequest(BaseModel):
    device: int
    family: str = "any"
    count: int = Field(40, ge=1, le=500)
    seed: int = Field(0, ge=0, le=1_000_000)

    @field_validator("device")
    @classmethod
    def known_device(cls, v: int) -> int:
        if v not in DEVICES:
            raise ValueError(f"unknown device: {v}")
        return v


class ExplainRequest(BaseModel):
    device: int
    index: int = Field(ge=0)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=NUM_FEATURES, max_length=NUM_FEATURES)


def _get_run(run_id: str) -> Run:
    run = manager.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "No such run")
    return run


def _finished_run(run_id: str) -> Run:
    run = _get_run(run_id)
    if not run.ledger or not run.records:
        raise HTTPException(409, "This run has no finished trust records yet")
    return run


def _read_ledger(fn):
    """A restarted local chain forgets old contracts. Say that plainly rather than returning a 500."""
    try:
        return fn()
    except Exception as exc:
        raise HTTPException(409, "The local chain was restarted, so this run's contract no longer exists. Run a new simulation.") from exc


def _chain_status() -> dict:
    return {"connected": chain_available(RPC_URL), "rpc_url": RPC_URL}


@app.get("/api/status")
def status() -> dict:
    return {
        "ok": True,
        "chain": _chain_status(),
        "storage": manager.store.status(),
        "model": bundle.meta,
        "simulation_running": manager.busy(),
        "runs": len(manager.runs),
    }


@app.get("/api/options")
def options() -> dict:
    return {
        "attacks": [{"id": k, "description": v} for k, v in ATTACK_INFO.items()],
        "defences": [{"id": k, "label": DEFENCE_LABELS[k]} for k in DEFENCES],
        "devices": [{"id": k, "name": v[0]} for k, v in DEVICES.items()],
        "classes": list(CLASSES),
        "fog_groups": FOG_GROUPS,
    }


@app.post("/api/simulations", status_code=202)
def start_simulation(req: SimulationRequest) -> dict:
    n = req.n_attackers if req.attack != "none" else 0
    if req.attack != "none" and n == 0:
        raise HTTPException(422, "Choose at least one attacker, or pick 'none' as the attack")
    try:
        return manager.start(req.attack, n, req.defence, req.rounds, req.seed, req.compare, req.dishonest_server)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def _run_view(run: Run, since: int = 0) -> dict:
    return {
        "id": run.id, "label": run.label, "config": run.config, "status": run.status, "error": run.error,
        "backend": run.backend, "rounds": [r for r in list(run.rounds) if r["round"] > since],
        "final": run.final, "summary": run.summary, "elapsed": run.elapsed, "tampered": run.tampered,
    }


@app.get("/api/jobs/current")
def current_job() -> dict:
    job = manager.active_job
    return job or {"status": "idle", "run_ids": []}


@app.post("/api/simulations/stop")
def stop_simulation() -> dict:
    manager.stop_flag.set()
    return {"stopping": True}


@app.get("/api/simulations/{run_id}")
def get_run(run_id: str, since: int = 0) -> dict:
    return _run_view(_get_run(run_id), since)


@app.get("/api/simulations")
def list_runs() -> list[dict]:
    rows = []
    for run_id in reversed(manager.order):
        run = manager.runs[run_id]
        rows.append({"id": run.id, "label": run.label, "status": run.status, "attack": run.config["attack"], "n_attackers": run.config["n_attackers"], "backend": run.backend, "macro_f1": run.final["macro_f1"] if run.final else None})
    return rows


@app.post("/api/detect/replay")
def replay(req: ReplayRequest) -> dict:
    """Replay recorded dataset traffic through the trained detector. No live malware is involved."""
    client = bundle.by_device[req.device]
    y = np.asarray(client.y_test)
    if req.family == "any":
        pool = np.arange(len(y))
    else:
        if req.family not in CLASSES:
            raise HTTPException(422, "Unknown family")
        pool = np.flatnonzero(y == CLASSES.index(req.family))
        if len(pool) == 0:
            raise HTTPException(422, f"{DEVICES[req.device][0]} has no recorded {req.family} traffic")
    rng = np.random.default_rng(req.seed)
    idx = np.sort(rng.choice(pool, size=min(req.count, len(pool)), replace=False))
    pred, conf = bundle.predict_scaled(bundle.scale.apply(client.x_test[idx]))
    rows = [
        {"index": int(i), "true": CLASSES[int(y[i])], "predicted": CLASSES[int(p)], "confidence": float(c), "correct": int(y[i]) == int(p), "flagged": int(p) != 0}
        for i, p, c in zip(idx, pred, conf)
    ]
    attack_rows = [r for r in rows if r["true"] != "benign"]
    benign_rows = [r for r in rows if r["true"] == "benign"]
    return {
        "device": req.device, "device_name": DEVICES[req.device][0], "rows": rows,
        "summary": {
            "total": len(rows), "correct_family": sum(r["correct"] for r in rows),
            "attacks": len(attack_rows), "attacks_flagged": sum(r["flagged"] for r in attack_rows),
            "benign": len(benign_rows), "false_alarms": sum(r["flagged"] for r in benign_rows),
        },
    }


@app.post("/api/detect/explain")
def explain(req: ExplainRequest) -> dict:
    client = bundle.by_device.get(req.device)
    if client is None:
        raise HTTPException(422, "Unknown device")
    if req.index >= len(client.y_test):
        raise HTTPException(422, "Record index out of range")
    return bundle.explain(req.device, req.index)


@app.post("/api/detect/predict")
def predict(req: PredictRequest) -> dict:
    """Classify one record given as its 115 raw N-BaIoT feature values."""
    raw = np.array([req.features], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise HTTPException(422, "Features must be finite numbers")
    pred, conf = bundle.predict_raw(raw)
    return {"predicted": CLASSES[int(pred[0])], "confidence": float(conf[0]), "flagged": int(pred[0]) != 0}


@app.get("/api/chain/status")
def chain_status() -> dict:
    return _chain_status()


@app.get("/api/chain/{run_id}")
def chain_entries(run_id: str) -> dict:
    run = _finished_run(run_id)
    ledger = run.ledger
    entries, models, info = _read_ledger(lambda: (ledger.entries, ledger.models, ledger.info() if hasattr(ledger, "info") else None))
    return {
        "backend": ledger.backend,
        "info": info,
        "entries": entries,
        "models": models,
        "records": run.records,
        "tampered": run.tampered,
    }


@app.post("/api/chain/{run_id}/verify")
def chain_verify(run_id: str) -> dict:
    run = _finished_run(run_id)
    return {**_read_ledger(lambda: manager.audit_run(run)), "backend": run.ledger.backend, "tampered": run.tampered}


@app.post("/api/chain/{run_id}/tamper")
def chain_tamper(run_id: str) -> dict:
    """Quietly rewrite one attacker's off-chain record to look clean, as a dishonest operator might."""
    run = _finished_run(run_id)
    victims = [r for r in run.records if not r["gate_passed"]] or run.records
    record = victims[0]
    record.update(cosine=0.97, norm_ratio=1.02, gate_passed=True, included=True, reason="within limits")
    run.tampered = True
    return {"round": record["round"], "client": record["client"], "changed": "marked as a clean, accepted update"}


@app.post("/api/chain/{run_id}/restore")
def chain_restore(run_id: str) -> dict:
    run = _finished_run(run_id)
    run.records = [dict(r) for r in run.original_records]
    run.tampered = False
    return {"restored": True}


def _final_model(run: Run) -> dict:
    """The last anchored model, read from the store and checked against its on-chain hash."""
    models = _read_ledger(lambda: run.ledger.models)
    if not models:
        raise HTTPException(409, "This run recorded no global model")
    latest = models[-1]
    cid = cid_from_hash(latest["model_hash"])
    try:
        data = manager.store.get(cid)
    except FileMismatch as exc:
        raise HTTPException(409, "The stored model does not match its on-chain hash, so it will not be served") from exc
    except FileMissing as exc:
        raise HTTPException(409, "The stored model is gone. The local store is cleared when the demo restarts, so run a new simulation.") from exc
    return {"round": latest["round"], "model_hash": latest["model_hash"], "cid": cid, "data": data}


@app.get("/api/audit/{run_id}")
def audit_overview(run_id: str) -> dict:
    run = _finished_run(run_id)
    entries = _read_ledger(lambda: run.ledger.entries)
    models = _read_ledger(lambda: run.ledger.models)
    files = []
    for entry in entries:
        cid = cid_from_hash(entry["update_hash"])
        try:
            where = manager.store.read(cid)[1]
        except (FileMismatch, FileMissing) as exc:
            where = "swapped" if isinstance(exc, FileMismatch) else "missing"
        files.append({"round": entry["round"], "client": entry["client"], "accepted": entry["accepted"], "cid": cid, "where": where})
    info = None
    try:
        final = _final_model(run)
        info = {"round": final["round"], "model_hash": final["model_hash"], "cid": final["cid"], "size": len(final["data"])}
    except HTTPException as exc:
        info = {"error": exc.detail}
    return {
        "backend": run.ledger.backend, "storage": manager.store.status(), "fog_mode": DEFENCES[run.config["defence"]].fog,
        "dishonest_server": run.config.get("dishonest_server", False), "config": run.config, "attackers": (run.summary or {}).get("attackers", []),
        "files": files, "models": len(models), "model": info, "swapped": len(manager.store.swapped),
        "fog_groups": FOG_GROUPS, "fog_of": {str(k): v for k, v in FOG_OF.items()},
    }


@app.post("/api/audit/{run_id}/recheck")
def audit_recheck(run_id: str) -> dict:
    run = _finished_run(run_id)
    entries, models = _read_ledger(lambda: (run.ledger.entries, run.ledger.models))
    return {**recheck_run(run.config, entries, models, run.sizes, manager.store), "chain_records_check": manager.audit_run(run)["valid"]}


@app.post("/api/audit/{run_id}/swap")
def audit_swap(run_id: str) -> dict:
    """Serve a different file for one update, as a dishonest operator might."""
    run = _finished_run(run_id)
    entries = _read_ledger(lambda: run.ledger.entries)
    entry = entries[len(entries) // 2]
    cid = cid_from_hash(entry["update_hash"])
    data = bytearray(manager.store.get(cid))
    data[:4] = b"\x00\x00\x80\x3f"
    manager.store.swapped[cid] = bytes(data)
    return {"round": entry["round"], "client": entry["client"]}


@app.post("/api/audit/restore")
def audit_restore() -> dict:
    manager.store.swapped.clear()
    return {"restored": True}


@app.get("/api/results")
def get_results() -> dict:
    return results


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(FRONTEND) / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
