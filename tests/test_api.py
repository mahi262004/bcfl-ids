"""API behaviour, including bad input and busy states. Uses the bundled demo model and data."""
from __future__ import annotations

import time

import pytest

from bcfl_ids.config import ATTACKS, DEFENCES, DEMO_DIR

pytestmark = pytest.mark.skipif(not (DEMO_DIR / "model.npz").exists(), reason="demo bundle not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend import app as module

    module.manager.rpc_url = "http://127.0.0.1:1"  # no chain in this suite: the fallback must work
    with TestClient(module.app) as c:
        yield c


def wait(client, job, limit=120):
    deadline = time.time() + limit
    while time.time() < deadline:
        views = [client.get(f"/api/simulations/{i}").json() for i in job["run_ids"]]
        if all(v["status"] in ("finished", "failed", "stopped") for v in views):
            return views
        time.sleep(0.1)
    raise TimeoutError


def test_status_and_page(client):
    assert client.get("/api/status").json()["ok"] is True
    page = client.get("/")
    assert page.status_code == 200 and "BCFL-IDS" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_options(client):
    data = client.get("/api/options").json()
    assert {a["id"] for a in data["attacks"]} == set(ATTACKS)
    assert {d["id"] for d in data["defences"]} == set(DEFENCES)


@pytest.mark.parametrize("payload", [
    {"attack": "nonsense"},
    {"defence": "nonsense"},
    {"attack": "sign_flip", "n_attackers": 9},
    {"attack": "sign_flip", "n_attackers": 0},
    {"rounds": 0},
    {"rounds": 9999},
    {"seed": -1},
])
def test_bad_simulation_requests_rejected(client, payload):
    assert client.post("/api/simulations", json=payload).status_code == 422


def test_full_flow_with_compare_and_tamper(client):
    job = client.post("/api/simulations", json={"attack": "sign_flip", "n_attackers": 3, "rounds": 3, "compare": True, "seed": 2}).json()
    assert client.post("/api/simulations", json={"rounds": 1}).status_code == 409  # one job at a time
    fedavg, proposed = wait(client, job)
    assert fedavg["status"] == proposed["status"] == "finished", (fedavg["error"], proposed["error"])
    assert proposed["backend"] == "memory"
    assert proposed["final"]["macro_f1"] > fedavg["final"]["macro_f1"]
    run = proposed["id"]
    assert client.post(f"/api/chain/{run}/verify").json()["valid"]
    client.post(f"/api/chain/{run}/tamper")
    outcome = client.post(f"/api/chain/{run}/verify").json()
    assert not outcome["valid"] and outcome["mismatches"]
    client.post(f"/api/chain/{run}/restore")
    assert client.post(f"/api/chain/{run}/verify").json()["valid"]
    assert any(r["id"] == run for r in client.get("/api/simulations").json())


def test_lost_chain_contract_gives_a_clear_message(client):
    job = client.post("/api/simulations", json={"rounds": 1, "seed": 4}).json()
    run_id = wait(client, job)[0]["id"]
    from backend import app as module

    class Gone:
        backend = "chain"

        @property
        def entries(self):
            raise RuntimeError("no contract")

    real = module.manager.runs[run_id].ledger
    module.manager.runs[run_id].ledger = Gone()
    for res in (client.get(f"/api/chain/{run_id}"), client.post(f"/api/chain/{run_id}/verify")):
        assert res.status_code == 409 and "restarted" in res.json()["detail"]
    module.manager.runs[run_id].ledger = real


def test_stop_ends_a_running_simulation(client):
    job = client.post("/api/simulations", json={"rounds": 30, "seed": 3}).json()
    client.post("/api/simulations/stop")
    views = wait(client, job)
    assert views[0]["status"] == "stopped"


def test_unknown_run_is_404(client):
    assert client.get("/api/simulations/doesnotexist").status_code == 404
    assert client.post("/api/chain/doesnotexist/verify").status_code == 404


def test_replay_and_explain(client):
    out = client.post("/api/detect/replay", json={"device": 5, "family": "mirai_syn", "count": 10}).json()
    assert len(out["rows"]) == 10 and all(r["true"] == "mirai_syn" for r in out["rows"])
    ex = client.post("/api/detect/explain", json={"device": 5, "index": out["rows"][0]["index"]}).json()
    assert len(ex["top"]) == 10 and ex["completeness_error"] < 0.1


def test_replay_of_family_a_device_never_saw_is_refused(client):
    res = client.post("/api/detect/replay", json={"device": 3, "family": "mirai_syn"})
    assert res.status_code == 422 and "no recorded" in res.json()["detail"]


@pytest.mark.parametrize("payload", [{"device": 99}, {"device": 1, "family": "x"}, {"device": 1, "count": 0}, {"device": 1, "count": 100000}])
def test_bad_replay_rejected(client, payload):
    assert client.post("/api/detect/replay", json=payload).status_code == 422


def test_explain_bad_index(client):
    assert client.post("/api/detect/explain", json={"device": 1, "index": 10**9}).status_code == 422
    assert client.post("/api/detect/explain", json={"device": 99, "index": 0}).status_code == 422


def test_predict_validates_shape_and_values(client):
    assert client.post("/api/detect/predict", json={"features": [0.0] * 3}).status_code == 422
    ok = client.post("/api/detect/predict", json={"features": [1.0] * 115})
    assert ok.status_code == 200 and "predicted" in ok.json()
