"""The evaluation grid: every attack against every defence, on the held-out test split.

Finished runs are appended to a JSONL file, so an interrupted grid resumes where it stopped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time

import torch

from bcfl_ids.config import ATTACKS, DEFENCES, RESULTS_DIR, SimConfig
from bcfl_ids.data import load_clients
from bcfl_ids.simulator import run_simulation

from .common import compact

_clients = None


def _init(frac: float, threads: int) -> None:
    global _clients
    torch.set_num_threads(threads)
    _clients = load_clients(frac=frac, split="test")


def _run(job: dict) -> dict:
    cfg = SimConfig(attack=job["attack"], n_attackers=job["n"], defence=job["defence"], seed=job["seed"], rounds=job["rounds"])
    return compact(run_simulation(_clients, cfg, eval_every=job["rounds"]))


def build_jobs(seeds, rounds) -> list[dict]:
    """3 attackers against everything; 1 and 2 attackers for the two ends of the comparison; 4 beyond budget."""
    jobs = []
    for defence, seed in itertools.product(DEFENCES, seeds):
        jobs.append({"attack": "none", "n": 0, "defence": defence, "seed": seed, "rounds": rounds})
    for attack, defence, seed in itertools.product([a for a in ATTACKS if a != "none"], DEFENCES, seeds):
        jobs.append({"attack": attack, "n": 3, "defence": defence, "seed": seed, "rounds": rounds})
    for attack, n, defence, seed in itertools.product([a for a in ATTACKS if a != "none"], (1, 2, 4), ("fedavg", "proposed"), seeds):
        jobs.append({"attack": attack, "n": n, "defence": defence, "seed": seed, "rounds": rounds})
    return jobs


def _key(j: dict) -> tuple:
    return (j["attack"], j["n"], j["defence"], j["seed"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="grid.jsonl")
    p.add_argument("--limit", type=int, default=0, help="run only this many jobs (for timing)")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / args.out
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            c = json.loads(line)["config"]
            done.add((c["attack"], c["n_attackers"], c["defence"], c["seed"]))
    jobs = [j for j in build_jobs(args.seeds, args.rounds) if _key(j) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(done)} done, {len(jobs)} to run", flush=True)

    start = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.frac, args.threads)) as pool, out.open("a") as f:
        for i, res in enumerate(pool.imap_unordered(_run, jobs), 1):
            f.write(json.dumps({**res, "frac": args.frac}) + "\n")
            f.flush()
            c = res["config"]
            if i % 10 == 0 or i == len(jobs):
                rate = (time.time() - start) / i
                print(f"{i}/{len(jobs)} {c['attack']}/{c['n_attackers']}/{c['defence']}/s{c['seed']} f1={res['final']['macro_f1']:.3f} eta={rate * (len(jobs) - i) / 60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
