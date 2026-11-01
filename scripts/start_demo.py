#!/usr/bin/env python3
"""Start everything for the demo: local chain, deploys per run, API, dashboard. Stop with Ctrl-C.

Works on macOS, Linux and Windows. If the chain cannot start, the demo still runs on the local
ledger, and the dashboard says so openly rather than pretending the chain is up.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runtime"
RPC_PORT = 8545
IPFS_API_PORT = 5901
IPFS_GATEWAY_PORT = 8901
IPFS_BINARY = ROOT / "tools" / "ipfs" / ("ipfs.exe" if os.name == "nt" else "ipfs")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(start: int) -> int:
    for port in range(start, start + 20):
        if not port_open(port):
            return port
    raise SystemExit(f"No free port between {start} and {start + 19}.")


def wait_for(check, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if check():
            return True
        time.sleep(0.3)
    return False


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_chain(procs: list) -> None:
    """Start Hardhat's local node, unless something already answers on the RPC port."""
    if port_open(RPC_PORT):
        print(f"Chain: already running on port {RPC_PORT}")
        return
    npx = shutil.which("npx")
    if not npx or not (ROOT / "blockchain" / "node_modules").exists():
        print("Chain: NOT started (Node.js or blockchain/node_modules missing). Using the local ledger.")
        return
    env = {**os.environ, "HARDHAT_DISABLE_TELEMETRY_PROMPT": "true"}
    log = open(LOGS / "chain.log", "w")
    proc = subprocess.Popen([npx, "hardhat", "node", "--hostname", "127.0.0.1", "--port", str(RPC_PORT)], cwd=ROOT / "blockchain", stdout=log, stderr=subprocess.STDOUT, env=env)
    procs.append(proc)
    if wait_for(lambda: port_open(RPC_PORT), 45):
        print(f"Chain: Hardhat node on port {RPC_PORT}")
    else:
        print("Chain: did not start in 45 s (see runtime/chain.log). Using the local ledger.")


def stop_all(procs: list) -> None:
    for p in procs:
        if p.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            else:
                p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-chain", action="store_true", help="skip the blockchain and use the local ledger")
    args = parser.parse_args()

    for needed in ("artifacts/demo/model.npz", "artifacts/demo/demo_data.npz", "blockchain/artifacts/contracts/TrustLayer.sol/TrustLayer.json"):
        if not (ROOT / needed).exists():
            raise SystemExit(f"Missing {needed}. The repository looks incomplete.")

    LOGS.mkdir(exist_ok=True)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS: numpy and torch may each bring an OpenMP
    procs: list = []
    try:
        if not args.no_chain:
            start_chain(procs)
        port = free_port(args.port)
        log = open(LOGS / "api.log", "w")
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        procs.append(api)
        url = f"http://127.0.0.1:{port}"
        if not wait_for(lambda: http_ok(f"{url}/api/status"), 90):
            raise SystemExit("The API did not start. See runtime/api.log.")
        print(f"\nDashboard ready: {url}\nPress Ctrl-C to stop.\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        while True:
            time.sleep(1)
            if api.poll() is not None:
                print("The API stopped unexpectedly. See runtime/api.log. Restarting it.", flush=True)
                api = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                procs.append(api)
            if not args.no_chain and len(procs) > 1 and procs[0].poll() is not None and not port_open(RPC_PORT):
                print("The chain stopped. Restarting it.", flush=True)
                procs.pop(0)
                restarted: list = []
                start_chain(restarted)
                procs[:0] = restarted
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_all(procs)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
