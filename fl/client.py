"""
Flower client: represents ONE REAL N-BaIoT device.

Unlike a generic FL demo, this does not need artificial data
partitioning -- each device_id (1-9) already has its own separate
traffic data in the real dataset, which is exactly what a federated
learning client should have.

Run from the project root, e.g.:
    python -m fl.client 1
    python -m fl.client 2
    ...(one process per device you want to simulate)
"""

import sys

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.data_loader import NON_FEATURE_COLUMNS, load_device_data
from fl.model import get_model


def load_client_data(device_id: int):
    df = load_device_data(device_id)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int64)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=32, shuffle=True), X.shape[1]


class NBaIoTClient(fl.client.NumPyClient):
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.train_loader, input_dim = load_client_data(device_id)
        self.model = get_model(input_dim=input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # increase once the pipeline is verified
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                outputs = self.model(X_batch)
                loss_sum += self.criterion(outputs, y_batch).item()
                correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)
        return loss_sum / len(self.train_loader), total, {"accuracy": correct / total}


if __name__ == "__main__":
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=NBaIoTClient(device_id))
