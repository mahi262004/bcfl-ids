"""HH_jit_*_mean holds a Unix timestamp on many rows. Does the detector or its explanation depend on it?

Run `python -m experiments.timestamp_check` (needs data/processed). Results go to timestamp_check.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bcfl_ids.config import BENIGN, CLASSES, DEMO_DIR, NUM_CLASSES, SimConfig
from bcfl_ids.data import load_clients
from bcfl_ids.explain import integrated_gradients
from bcfl_ids.scaler import Scale
from bcfl_ids.simulator import ClientData, run_simulation

from .common import save_json
from .reliability_check import build, load_test, logits_of

LOG_LIMIT = np.log1p(1e8)  # no genuine jitter is this large: 1e8 seconds is three years


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    names = json.loads((DEMO_DIR / "features.json").read_text())
    saved = np.load(DEMO_DIR / "model.npz")
    scale = Scale(saved["mean"], saved["std"])
    model = build(saved["weights"])
    baseline = torch.from_numpy(saved["benign_baseline"].astype(np.float32))
    clients = load_clients(frac=0.05, split="test")
    out: dict = {}

    # Which columns hold clock values, and what do the raw rows look like?
    raw = np.concatenate([np.asarray(c.x_test) for c in clients])
    y = np.concatenate([np.asarray(c.y_test) for c in clients]).astype(np.int64)
    stamped = raw > LOG_LIMIT
    columns = [i for i in range(raw.shape[1]) if stamped[:, i].any()]
    out["columns_with_clock_values"] = {names[i]: float(stamped[:, i].mean()) for i in columns}
    jit = [i for i, n in enumerate(names) if n.startswith("HH_jit") and n.endswith("_mean")]
    out["jitter_mean_columns"] = [names[i] for i in jit]
    any_stamp = stamped[:, jit].any(1)
    all_stamp = stamped[:, jit].all(1)
    values = np.expm1(raw[:, jit[0]][stamped[:, jit[0]]].astype(np.float64))
    out["clock_range_utc"] = {"earliest": utc(float(values.min())), "latest": utc(float(values.max()))}
    out["share_with_clock_value"] = {CLASSES[c]: {"any_window": float(any_stamp[y == c].mean()), "all_windows": float(all_stamp[y == c].mean())} for c in range(NUM_CLASSES)}

    # Input precision left after the pipeline: how many of the seconds can the model see at all?
    scaled_step = float(np.spacing(np.float32(21.13)) / scale.std[jit[0]])
    out["scaling"] = {"std_of_first_jit_mean_column_log": float(scale.std[jit[0]]), "one_float32_step_after_scaling": scaled_step, "one_day_after_log_and_scaling": float(86400 / 1.5059e9 / scale.std[jit[0]])}

    # Does the deployed model react to the clock itself? Move every clock value, keep the rest.
    x_test = np.clip((raw - scale.mean) / scale.std, -10, 10).astype(np.float32)
    base_logits = logits_of(model, x_test)
    pred = base_logits.argmax(1)
    conf = torch.softmax(torch.from_numpy(base_logits), 1).max(1).values.numpy()
    out["clock_moved"] = {}
    for label, seconds in (("one hour later", 1.5059e9 + 3600), ("one month later", 1.5059e9 + 30 * 86400), ("today, 2026", 1.7934e9), ("year 2001", 1.0e9), ("year 2286", 1.0e10)):
        moved = raw.copy()
        moved[:, jit] = np.where(stamped[:, jit], np.log1p(seconds), moved[:, jit])
        x = np.clip((moved - scale.mean) / scale.std, -10, 10).astype(np.float32)
        logits = logits_of(model, x)
        p = logits.argmax(1)
        changed = p != pred
        out["clock_moved"][label] = {
            "verdict_changed": float(changed.mean()), "verdict_changed_on_stamped_rows": float(changed[any_stamp].mean()) if any_stamp.any() else None,
            "attack_vs_benign_changed": float(((p != BENIGN) != (pred != BENIGN)).mean()),
            "accuracy": float((p == y).mean()), "attack_recall": float((p[y != BENIGN] != BENIGN).mean()), "benign_recall": float((p[y == BENIGN] == BENIGN).mean()),
        }
        print(label, out["clock_moved"][label], flush=True)

    # Do explanations lean on these columns? Records sampled evenly across the test split.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 3000, replace=False)
    xs, ps = torch.from_numpy(x_test[idx]), torch.from_numpy(pred[idx])
    flagged = ps != BENIGN
    attr = integrated_gradients(model, xs[flagged], baseline, ps[flagged])
    shares = attr[:, jit].abs().sum(1) / attr.abs().sum(1).clamp_min(1e-12)
    top1 = attr.abs().argmax(1).numpy()
    top10 = attr.abs().argsort(1, descending=True)[:, :10].numpy()
    out["explanations"] = {
        "records": int(flagged.sum()), "evidence_share_on_jit_mean_columns": float(shares.mean()),
        "strongest_signal_is_jit_mean": float(np.isin(top1, jit).mean()), "any_top10_is_jit_mean": float(np.isin(top10, jit).any(1).mean()),
        "per_family_share": {CLASSES[c]: float(shares[ps[flagged] == c].mean()) for c in sorted(set(ps[flagged].tolist()))},
    }

    # Is the column needed? Retrain with the clock removed (flat column), then a binary "clock value present" version.
    def variant(kind: str) -> list[ClientData]:
        made = []
        for c in clients:
            tr, te = np.array(c.x_train), np.array(c.x_test)
            for arr in (tr, te):
                mask = arr[:, jit] > LOG_LIMIT
                if kind == "removed":
                    arr[:, jit] = 0.0
                else:  # keep only "this was a clock value", never the time, and leave real jitter alone
                    arr[:, jit] = np.where(mask, 21.0, arr[:, jit])
            made.append(ClientData(c.device, tr, c.y_train, te, np.asarray(c.y_test)))
        return made

    out["retrained"] = {}
    for kind, data in (("as built", clients), ("clock value fixed to a constant", variant("flag")), ("jit mean columns removed", variant("removed"))):
        rows = []
        for seed in (1, 2, 3):
            f = run_simulation(data, SimConfig(defence="proposed", rounds=20, seed=seed), eval_every=20)["final"]
            rows.append({k: f[k] for k in ("macro_f1", "macro_f1_merged", "accuracy", "attack_recall", "benign_recall", "worst_device_benign_recall")})
        out["retrained"][kind] = {k: {"mean": float(np.mean([r[k] for r in rows])), "values": [r[k] for r in rows]} for k in rows[0]}
        print(kind, {k: round(v["mean"], 4) for k, v in out["retrained"][kind].items()}, flush=True)
        save_json("timestamp_check.json", out)


if __name__ == "__main__":
    main()
