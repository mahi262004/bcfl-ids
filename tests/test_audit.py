"""Fog gating, the update store and the re-check of a run's decisions."""
from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from bcfl_ids import config as C
from bcfl_ids.defences import gate_round, score_updates
from bcfl_ids.ledger import InMemoryLedger
from bcfl_ids.reverify import recheck_run
from bcfl_ids.simulator import run_simulation, weights_hash
from bcfl_ids.storage import FileMismatch, FileMissing, IpfsClient, UpdateStore, cid_for, cid_from_hash, digest_of

BINARY = C.ROOT / "tools" / "ipfs" / ("ipfs.exe" if os.name == "nt" else "ipfs")


def cfg(**kw):
    return C.SimConfig(rounds=kw.pop("rounds", 3), batch_size=32, **kw)


def run(clients, tmp_path, **kw):
    store = UpdateStore(tmp_path / "updates")
    ledger = InMemoryLedger()
    config = cfg(**kw)
    result = run_simulation(clients, config, ledger=ledger, store=store)
    return result, ledger, store


def check(result, ledger, store):
    return recheck_run(result["config"], ledger.entries, ledger.models, result["sizes"], store)


def test_address_is_the_on_chain_hash_reencoded():
    data = b"an update"
    digest = hashlib.sha256(data).digest()
    assert cid_for(data) == cid_from_hash("0x" + digest.hex())
    assert digest_of(cid_for(data)) == digest
    assert cid_for(data).startswith("bafkrei")  # What a CIDv1 raw sha2-256 block looks like on IPFS.


def test_local_store_round_trip_and_integrity(tmp_path):
    store = UpdateStore(tmp_path)
    placed = store.put(b"abc")
    assert placed["where"] == "local" and store.get(placed["cid"]) == b"abc"
    (tmp_path / placed["cid"]).write_bytes(b"xyz")
    with pytest.raises(FileMismatch):
        store.get(placed["cid"])
    with pytest.raises(FileMissing):
        store.get(cid_for(b"never stored"))


def test_store_falls_back_when_the_node_is_down(tmp_path):
    store = UpdateStore(tmp_path, IpfsClient("http://127.0.0.1:1"))
    placed = store.put(b"abc")
    assert placed["where"] == "local" and store.get(placed["cid"]) == b"abc"
    assert store.status()["connected"] is False and store.status()["local_files"] == 1


def test_fog_gate_compares_each_device_only_with_its_own_group():
    torch.manual_seed(0)
    base = torch.randn(50)
    updates = torch.stack([base + 0.05 * torch.randn(50) for _ in range(9)])
    updates[3:6] = -updates[3:6]  # All of fog B points the other way.
    devices = list(C.DEVICE_IDS)
    fog = gate_round(updates, devices, C.DEFENCES["fog"], 0.1, 2.0)
    central = gate_round(updates, devices, C.DEFENCES["proposed"], 0.1, 2.0)
    assert all(v.passed for v in fog)  # Fog B's devices agree with each other, so its gate cannot see it.
    assert [v.passed for v in central[3:6]] == [False] * 3
    assert [v.cosine for v in fog[:3]] == [v.cosine for v in score_updates(updates[:3], 0.1, 2.0)]


def test_fog_records_name_the_node(clients, tmp_path):
    result, _, _ = run(clients, tmp_path, defence="fog")
    assert {r["client"]: r["fog"] for r in result["records"]} == C.FOG_OF
    central, _, _ = run(clients, tmp_path, defence="proposed")
    assert {r["fog"] for r in central["records"]} == {None}


@pytest.mark.parametrize("defence", list(C.DEFENCES))
def test_honest_run_rechecks_clean_with_every_defence(clients, tmp_path, defence):
    outcome = check(*run(clients, tmp_path, attack="sign_flip", n_attackers=3, defence=defence))
    assert outcome["valid"], outcome["problems"]
    assert outcome["files_checked"] == 3 * 10 and all(r["model"] == "matches" for r in outcome["rounds"])


def test_storing_files_does_not_change_the_result(clients, tmp_path):
    stored, _, _ = run(clients, tmp_path, defence="proposed")
    plain = run_simulation(clients, cfg(defence="proposed"))
    assert torch.equal(stored["weights"], plain["weights"])


def test_lying_server_passes_the_record_check_but_not_the_recheck(clients, tmp_path):
    from bcfl_ids.chain import audit

    result, ledger, store = run(clients, tmp_path, attack="sign_flip", n_attackers=3, defence="proposed", dishonest_server=True)
    assert audit(result["records"], ledger.entries)["valid"]  # The chain faithfully stores the lie.
    outcome = check(result, ledger, store)
    assert not outcome["valid"] and outcome["decisions_differ"] > 0
    assert {p["client"] for p in outcome["problems"] if p["client"]} <= set(result["config"]["attacker_ids"])
    assert any("the chain says" in p["problem"] for p in outcome["problems"])


def test_swapped_and_missing_files_are_reported(clients, tmp_path):
    result, ledger, store = run(clients, tmp_path, rounds=2)
    target = cid_from_hash(ledger.entries[0]["update_hash"])
    store.swapped[target] = b"\x00" * 16
    outcome = check(result, ledger, store)
    assert not outcome["valid"] and "does not match" in outcome["problems"][0]["problem"]
    store.swapped.clear()
    (tmp_path / "updates" / cid_from_hash(ledger.entries[1]["update_hash"])).unlink()
    assert "missing" in check(result, ledger, store)["problems"][0]["problem"]


def test_a_changed_model_file_is_caught_by_rerunning_aggregation(clients, tmp_path):
    result, ledger, store = run(clients, tmp_path, rounds=2)
    wrong = result["weights"] + 0.5
    ledger.models[-1]["model_hash"] = weights_hash(wrong)
    store.put(wrong.numpy().astype(np.float32).tobytes())
    outcome = check(result, ledger, store)
    assert outcome["rounds"][-1]["model"] == "differs" and not outcome["valid"]


@pytest.fixture(scope="module")
def ipfs_node(tmp_path_factory):
    if not BINARY.exists():
        pytest.skip("IPFS node not installed")
    port = 5977
    repo: Path = tmp_path_factory.mktemp("ipfs")
    env = {**os.environ, "IPFS_PATH": str(repo)}
    subprocess.run([str(BINARY), "init", "--profile=test"], env=env, check=True, capture_output=True, timeout=60)
    subprocess.run([str(BINARY), "config", "Addresses.API", f"/ip4/127.0.0.1/tcp/{port}"], env=env, check=True, capture_output=True, timeout=60)
    subprocess.run([str(BINARY), "config", "Addresses.Gateway", "/ip4/127.0.0.1/tcp/5978"], env=env, check=True, capture_output=True, timeout=60)
    proc = subprocess.Popen([str(BINARY), "daemon", "--offline"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("IPFS node did not start")
    yield f"http://127.0.0.1:{port}"
    proc.kill()
    proc.wait(timeout=10)


def test_real_ipfs_gives_the_address_derived_from_the_chain_hash(ipfs_node, tmp_path):
    client = IpfsClient(ipfs_node)
    assert client.version()
    update = torch.randn(23819).numpy().astype(np.float32).tobytes()
    assert client.add(update) == cid_for(update) == cid_from_hash("0x" + hashlib.sha256(update).hexdigest())
    assert client.cat(cid_for(update)) == update


def test_full_run_rechecks_from_a_real_ipfs_node(clients, ipfs_node, tmp_path):
    store = UpdateStore(tmp_path / "unused", IpfsClient(ipfs_node))
    ledger = InMemoryLedger()
    result = run_simulation(clients, cfg(attack="sign_flip", n_attackers=3, defence="fog"), ledger=ledger, store=store)
    assert store.ipfs_files == 3 * 10 and store.local_files == 0
    assert not (tmp_path / "unused").exists()
    outcome = check(result, ledger, store)
    assert outcome["valid"], outcome["problems"]
