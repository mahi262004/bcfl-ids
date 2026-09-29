"""
Flower server: orchestrates federated rounds across the (up to 9)
real N-BaIoT device clients and aggregates their updates.

Phase 1 uses standard FedAvg to get the pipeline running end-to-end.
Trust-weighted aggregation (querying bridge/web3_client.py before
accepting each update) gets layered in once blockchain/ is deployed
and tested independently.

Run from the project root:
    python -m fl.server
"""

import flwr as fl

NUM_ROUNDS = 10
MIN_CLIENTS = 3  # match this to how many fl.client processes you actually run


def main():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
