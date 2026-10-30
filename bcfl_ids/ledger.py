"""Trust ledger. InMemoryLedger mirrors the TrustLayer contract rule for rule.

The chain-backed ledger lives in chain.py; a test confirms both give identical reputations.
"""
from __future__ import annotations

import hashlib
import json

from .config import REP_CAP, REP_FLOOR, REP_GAIN, REP_LOSS, REP_PROBATION, REP_START, SCORE_PASS


def sha256_hex(data: bytes) -> str:
    return "0x" + hashlib.sha256(data).hexdigest()


def canonical_hash(record: dict) -> str:
    """Hash of a trust record in canonical JSON, so anyone can recompute it."""
    return sha256_hex(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())


class InMemoryLedger:
    backend = "memory"

    def __init__(self) -> None:
        self.rep: dict[int, int] = {}
        self.entries: list[dict] = []
        self.models: list[dict] = []

    def register(self, client_id: int) -> None:
        if client_id in self.rep:
            raise ValueError("client already registered")
        self.rep[client_id] = REP_START

    def submit(self, round_no: int, client_id: int, score: int, update_hash: str, record_hash: str) -> dict:
        if client_id not in self.rep:
            raise ValueError("client not registered")
        if not 0 <= score <= 100:
            raise ValueError("score out of range")
        accepted = score >= SCORE_PASS
        before = self.rep[client_id]
        after = min(REP_CAP, before + REP_GAIN) if accepted else max(REP_FLOOR, before - REP_LOSS)
        self.rep[client_id] = after
        entry = {
            "round": round_no, "client": client_id, "score": score, "accepted": accepted,
            "rep_before": before, "rep_after": after, "update_hash": update_hash, "record_hash": record_hash,
        }
        self.entries.append(entry)
        return entry

    def anchor_model(self, round_no: int, model_hash: str) -> None:
        self.models.append({"round": round_no, "model_hash": model_hash})

    def reputation(self, client_id: int) -> int:
        return self.rep[client_id]

    def on_probation(self, client_id: int) -> bool:
        return self.rep[client_id] < REP_PROBATION
