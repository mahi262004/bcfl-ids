"""Compare training settings on the validation split, with a calibration seed."""
from __future__ import annotations

from bcfl_ids.config import SimConfig
from bcfl_ids.data import load_clients
from bcfl_ids.simulator import run_simulation

from .common import save_json


def main() -> None:
    clients = load_clients(frac=0.05, split="val", test_frac=0.25)
    rows = []
    for epochs, batch, lr in ((1, 256, 1e-3), (2, 256, 1e-3), (1, 128, 1e-3), (2, 128, 1e-3), (1, 256, 3e-3), (3, 256, 1e-3)):
        res = run_simulation(clients, SimConfig(defence="fedavg", rounds=15, seed=101, local_epochs=epochs, batch_size=batch, lr=lr), eval_every=15)
        f = res["final"]
        row = {"local_epochs": epochs, "batch_size": batch, "lr": lr, "macro_f1": f["macro_f1"], "macro_f1_merged": f["macro_f1_merged"], "accuracy": f["accuracy"], "attack_recall": f["attack_recall"], "benign_recall": f["benign_recall"], "elapsed_s": res["elapsed_s"]}
        rows.append(row)
        print(row, flush=True)
    print({k: round(v["recall"], 3) for k, v in f["per_class"].items()}, flush=True)
    save_json("tuning.json", rows)


if __name__ == "__main__":
    main()
