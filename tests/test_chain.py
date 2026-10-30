"""Run the real contract on a real local Hardhat node, driven from Python."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time

import pytest

from bcfl_ids import config as C
from bcfl_ids.chain import ARTIFACT, ChainLedger, audit
from bcfl_ids.ledger import InMemoryLedger
from bcfl_ids.simulator import run_simulation

PORT = 8599
RPC = f"http://127.0.0.1:{PORT}"
NODE_MODULES = C.BLOCKCHAIN_DIR / "node_modules"
pytestmark = pytest.mark.skipif(not NODE_MODULES.exists() or not ARTIFACT.exists() or not shutil.which("npx"), reason="blockchain tools not installed")


@pytest.fixture(scope="module")
def node():
    env = {**os.environ, "HARDHAT_DISABLE_TELEMETRY_PROMPT": "true"}
    proc = subprocess.Popen([shutil.which("npx"), "hardhat", "node", "--hostname", "127.0.0.1", "--port", str(PORT)], cwd=C.BLOCKCHAIN_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    deadline = time.time() + 60
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("Hardhat node did not start")
    yield RPC
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    proc.wait(timeout=10)


def test_chain_ledger_matches_memory_ledger(node):
    chain, memory = ChainLedger(node), InMemoryLedger()
    chain.register(1)
    memory.register(1)
    hash_ = "0x" + "ab" * 32
    for score in (90, 0, 0, 0, 0, 50, 49, 100):
        a = chain.submit(1, 1, score, hash_, hash_)
        b = memory.submit(1, 1, score, hash_, hash_)
        assert (a["accepted"], a["rep_before"], a["rep_after"]) == (b["accepted"], b["rep_before"], b["rep_after"])
    assert chain.on_probation(1) == memory.on_probation(1)


def test_chain_refuses_duplicate_registration(node):
    chain = ChainLedger(node)
    chain.register(1)
    with pytest.raises(Exception):
        chain.register(1)


def test_full_training_run_on_real_contract(node, clients):
    cfg = C.SimConfig(attack="sign_flip", n_attackers=2, defence="proposed", rounds=4, batch_size=32, seed=3)
    chain = ChainLedger(node)
    on_chain = run_simulation(clients, cfg, ledger=chain)
    in_memory = run_simulation(clients, cfg, ledger=InMemoryLedger())
    assert on_chain["ledger_backend"] == "chain"
    assert on_chain["reputation"] == in_memory["reputation"]
    assert on_chain["weights"].tolist() == in_memory["weights"].tolist()
    entries = chain.entries
    assert len(entries) == 36
    assert len(chain.models) == 4
    assert chain.models[-1]["model_hash"] == on_chain["history"][-1]["model_hash"]
    assert audit(on_chain["records"], entries)["valid"]
    records = [dict(r) for r in on_chain["records"]]
    bad = next(r for r in records if not r["gate_passed"])
    bad.update(gate_passed=True, reason="within limits")
    assert not audit(records, entries)["valid"]
