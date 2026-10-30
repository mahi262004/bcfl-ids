"""Build artifacts/demo: the trained model and a small data sample, so the demo needs no raw dataset."""
from __future__ import annotations

import argparse
import json

import numpy as np

from bcfl_ids.config import BENIGN, DEMO_DIR, DEVICE_IDS, NUM_CLASSES, PROCESSED_DIR, SimConfig
from bcfl_ids.data import feature_names, load_clients
from bcfl_ids.simulator import run_simulation


def per_class_sample(x, y, per_class: int, rng: np.random.Generator):
    y = np.asarray(y)
    keep = []
    for cls in range(NUM_CLASSES):
        idx = np.flatnonzero(y == cls)
        if len(idx):
            keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return np.asarray(x[keep], dtype=np.float32), y[keep].astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.30)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--demo-train", type=int, default=500)
    p.add_argument("--demo-test", type=int, default=150)
    args = p.parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    clients = load_clients(frac=args.frac, split="test")
    result = run_simulation(clients, SimConfig(defence="proposed", rounds=args.rounds, seed=args.seed), eval_every=args.rounds)
    final = result["final"]
    print("detector:", {k: round(final[k], 4) for k in ("accuracy", "macro_f1", "attack_recall", "benign_recall", "mcc")}, "elapsed", round(result["elapsed_s"]), flush=True)

    scale = result["scale"]
    rng = np.random.default_rng(2026)
    benign_rows = []
    arrays = {"devices": np.array(DEVICE_IDS)}
    for c in clients:
        y_test = np.asarray(c.y_test)
        bi = np.flatnonzero(y_test == BENIGN)
        benign_rows.append(scale.apply(np.asarray(c.x_test[rng.choice(bi, min(500, len(bi)), replace=False)])))
        x_tr, y_tr = per_class_sample(c.x_train, c.y_train, args.demo_train, rng)
        x_te, y_te = per_class_sample(c.x_test, y_test, args.demo_test, rng)
        arrays |= {f"x_train_{c.device}": x_tr, f"y_train_{c.device}": y_tr, f"x_test_{c.device}": x_te, f"y_test_{c.device}": y_te}
    baseline = np.concatenate(benign_rows).mean(0)

    np.savez_compressed(DEMO_DIR / "demo_data.npz", **arrays)
    np.savez_compressed(DEMO_DIR / "model.npz", weights=result["weights"].numpy(), mean=scale.mean, std=scale.std, benign_baseline=baseline)
    (DEMO_DIR / "features.json").write_text(json.dumps(feature_names(PROCESSED_DIR)))
    meta = {
        "description": "Federated detector, trained with the proposed defence and no attacker.",
        "train_fraction": args.frac, "rounds": args.rounds, "seed": args.seed,
        "test_metrics": {k: final[k] for k in ("accuracy", "macro_f1", "attack_recall", "benign_recall", "mcc")},
        "test_records": int(np.sum(final["confusion"])),
    }
    (DEMO_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print("demo bundle written", flush=True)


if __name__ == "__main__":
    main()
