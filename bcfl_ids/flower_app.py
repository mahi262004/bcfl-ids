"""The same defended aggregation over a real network protocol, using Flower client and server processes.

Optional: install requirements-flower.txt. The experiments and dashboard use the in-process
simulator because it is exactly repeatable; this module shows the logic runs over gRPC too.

    python -m bcfl_ids.flower_app --rounds 3            # starts a server and 9 client processes
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from .config import DEFENCES, DEMO_DIR, SimConfig
from .defences import aggregate, score_updates
from .ledger import InMemoryLedger, canonical_hash
from .metrics import summarise
from .model import Detector, initial_weights, load_flat
from .scaler import ClientStats, aggregate_robust, client_stats
from .simulator import ClientData, evaluate_cube, local_train, weights_hash

ADDRESS = "127.0.0.1:8088"


def _shapes() -> list[tuple[int, ...]]:
    return [tuple(p.shape) for p in Detector().parameters()]


def to_arrays(w: torch.Tensor) -> list[np.ndarray]:
    out, start = [], 0
    for shape in _shapes():
        size = int(np.prod(shape))
        out.append(w[start : start + size].reshape(shape).numpy())
        start += size
    return out


def to_flat(arrays: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([np.asarray(a).ravel() for a in arrays]).astype(np.float32))


def load_demo_clients() -> dict[int, ClientData]:
    data = np.load(DEMO_DIR / "demo_data.npz")
    return {int(d): ClientData(int(d), data[f"x_train_{d}"], data[f"y_train_{d}"], data[f"x_test_{d}"], data[f"y_test_{d}"]) for d in data["devices"]}


def run_client(device: int) -> None:
    import flwr as fl

    torch.set_num_threads(1)
    client = load_demo_clients()[device]
    cfg = SimConfig(batch_size=128)
    model = Detector()

    class Client(fl.client.NumPyClient):
        scaled = None

        def get_properties(self, config):
            s = client_stats(client.x_train)
            return {"n": s.n, "s": s.s.astype(np.float64).tobytes(), "q": s.q.astype(np.float64).tobytes(), "device": device}

        def fit(self, parameters, config):
            # The server sends the agreed scale with the first round; raw records never move.
            if self.scaled is None:
                mean, std = np.frombuffer(config["mean"], dtype=np.float64), np.frombuffer(config["std"], dtype=np.float64)
                self.scaled = torch.from_numpy(np.clip((np.asarray(client.x_train) - mean) / std, -10, 10).astype(np.float32))
                self.labels = torch.from_numpy(np.asarray(client.y_train).astype(np.int64))
            w = to_flat(parameters)
            gen = torch.Generator().manual_seed(1000 * int(config["round"]) + device)
            update = local_train(model, w, self.scaled, self.labels, cfg, gen)
            return to_arrays(w + update), len(self.labels), {"device": device}

    fl.client.start_client(server_address=ADDRESS, client=Client().to_client())


def run_server(rounds: int, n_clients: int = 9) -> dict:
    import flwr as fl
    from flwr.common import FitIns, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays

    clients = load_demo_clients()
    cfg = SimConfig(defence="proposed", rounds=rounds)
    defence = DEFENCES[cfg.defence]
    ledger = InMemoryLedger()
    outcome: dict = {"rounds": []}

    class TrustStrategy(fl.server.strategy.Strategy):
        scale_sent: dict | None = None
        w = initial_weights(cfg.seed)

        def initialize_parameters(self, client_manager):
            return ndarrays_to_parameters(to_arrays(self.w))

        def _agree_scale(self, proxies):
            stats = []
            for proxy in proxies:
                props = proxy.get_properties(GetPropertiesIns({}), timeout=60, group_id="scale").properties
                stats.append(ClientStats(props["n"], np.frombuffer(props["s"], dtype=np.float64), np.frombuffer(props["q"], dtype=np.float64)))
                ledger.register(int(props["device"]))
            self.scale = aggregate_robust(stats, cfg.trim)
            self.scale_sent = {"mean": self.scale.mean.astype(np.float64).tobytes(), "std": self.scale.std.astype(np.float64).tobytes()}

        def configure_fit(self, server_round, parameters, client_manager):
            proxies = client_manager.sample(num_clients=n_clients, min_num_clients=n_clients)
            if self.scale_sent is None:
                self._agree_scale(proxies)
            return [(p, FitIns(parameters, {"round": server_round, **self.scale_sent})) for p in proxies]

        def aggregate_fit(self, server_round, results, failures):
            results = sorted(results, key=lambda r: r[1].metrics["device"])
            devices = [int(r[1].metrics["device"]) for r in results]
            updates = torch.stack([to_flat(parameters_to_ndarrays(r[1].parameters)) - self.w for r in results])
            verdicts = score_updates(updates, cfg.cos_min, cfg.ratio_max)
            keep = []
            for i, (device, v) in enumerate(zip(devices, verdicts)):
                on_probation = ledger.on_probation(device)
                record = {"round": server_round, "client": device, "cosine": v.cosine, "norm_ratio": v.norm_ratio, "score": v.score, "gate_passed": v.passed, "on_probation": on_probation}
                ledger.submit(server_round, device, v.score, weights_hash(updates[i]), canonical_hash(record))
                if v.passed and not on_probation:
                    keep.append(i)
            sizes = torch.tensor([float(r[1].num_examples) for r in results])
            if keep:
                self.w = self.w + aggregate(updates[keep], sizes[keep], defence.agg, cfg.trim, cfg.krum_f)
            model = Detector()
            load_flat(model, self.w)
            metrics = summarise(evaluate_cube(model, list(clients.values()), self.scale))
            outcome["rounds"].append({"round": server_round, "accepted": len(keep), "macro_f1": metrics["macro_f1"], "failures": len(failures)})
            print(f"round {server_round}: accepted {len(keep)} of {len(results)}, macro F1 {metrics['macro_f1']:.3f}", flush=True)
            return ndarrays_to_parameters(to_arrays(self.w)), {}

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

        def evaluate(self, server_round, parameters):
            return None

    fl.server.start_server(server_address=ADDRESS, config=fl.server.ServerConfig(num_rounds=rounds), strategy=TrustStrategy())
    outcome["reputation"] = dict(ledger.rep)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    workers = [ctx.Process(target=run_client, args=(d,), daemon=True) for d in load_demo_clients()]

    def launch() -> None:
        time.sleep(3)
        for w in workers:
            w.start()

    import threading

    threading.Thread(target=launch, daemon=True).start()
    outcome = run_server(args.rounds)
    print("final:", outcome)
    for w in workers:
        w.join(timeout=10)


if __name__ == "__main__":
    main()
