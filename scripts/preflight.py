#!/usr/bin/env python3
"""End-to-end check of a RUNNING demo. Run this on the demo laptop before presenting.

usage: python scripts/preflight.py [http://127.0.0.1:8000]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
failures: list[str] = []


def call(path: str, payload: dict | None = None, timeout: float = 120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if "json" in r.headers.get("content-type", "") else body


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def wait_job(job: dict, limit: float = 300) -> list[dict]:
    deadline = time.time() + limit
    while time.time() < deadline:
        views = [call(f"/api/simulations/{i}") for i in job["run_ids"]]
        if all(v["status"] in ("finished", "failed", "stopped") for v in views):
            return views
        time.sleep(0.5)
    raise TimeoutError("simulation did not finish")


def main() -> None:
    try:
        status = call("/api/status")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[FAIL] Cannot reach the demo at {BASE}: {exc}\nStart it first: python scripts/start_demo.py")
        sys.exit(1)
    check("API responds", status["ok"] is True)
    check("Blockchain connected", status["chain"]["connected"], status["chain"]["rpc_url"])
    check("Dashboard page served", b"BCFL-IDS" in call("/"))
    for asset in ("/static/app.js", "/static/charts.js", "/static/style.css"):
        check(f"Asset {asset}", len(call(asset)) > 100)

    opts = call("/api/options")
    check("11 classes and 9 devices listed", len(opts["classes"]) == 11 and len(opts["devices"]) == 9)
    results = call("/api/results")
    for name in ("grid_summary", "explain_eval", "data_audit"):
        check(f"Results file: {name}", name in results)

    started = time.time()
    views = wait_job(call("/api/simulations", {"attack": "sign_flip", "n_attackers": 3, "defence": "proposed", "rounds": 5, "seed": 7, "compare": True}))
    check("Comparison run finishes", all(v["status"] == "finished" for v in views), f"{time.time() - started:.1f} s, errors: {[v['error'] for v in views if v['error']]}")
    fedavg, proposed = views
    if fedavg["final"] and proposed["final"]:
        check("Defence beats no defence", proposed["final"]["macro_f1"] > fedavg["final"]["macro_f1"], f"proposed {proposed['final']['macro_f1']:.3f}, no defence {fedavg['final']['macro_f1']:.3f}")
        check("Attackers rejected", proposed["summary"]["attacker_reject_rate"] > 0.5, f"{proposed['summary']['attacker_reject_rate']:.0%}")
    run_id = proposed["id"]
    chain = call(f"/api/chain/{run_id}")
    check("Run logged on the chain", chain["backend"] == "chain", chain["backend"])
    check("One entry per client per round", len(chain["entries"]) == 45, str(len(chain["entries"])))
    check("Records verify", call(f"/api/chain/{run_id}/verify", {})["valid"])
    call(f"/api/chain/{run_id}/tamper", {})
    check("Tampering detected", call(f"/api/chain/{run_id}/verify", {})["valid"] is False)
    call(f"/api/chain/{run_id}/restore", {})
    check("Restore verifies again", call(f"/api/chain/{run_id}/verify", {})["valid"])

    replay = call("/api/detect/replay", {"device": 5, "family": "mirai_syn", "count": 20, "seed": 1})
    check("Replay flags recorded attacks", replay["summary"]["attacks_flagged"] >= 18, f"{replay['summary']['attacks_flagged']} of {replay['summary']['attacks']}")
    benign = call("/api/detect/replay", {"device": 5, "family": "benign", "count": 20, "seed": 1})
    check("Replay leaves benign traffic alone", benign["summary"]["false_alarms"] <= 2, f"{benign['summary']['false_alarms']} false alarms of {benign['summary']['benign']}")
    explanation = call("/api/detect/explain", {"device": 5, "index": replay["rows"][0]["index"]})
    check("Explanation returned", len(explanation["top"]) == 10 and explanation["completeness_error"] < 0.1, f"integration error {explanation['completeness_error']:.3%}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
