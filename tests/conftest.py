"""Small synthetic federation, so the suite needs no dataset and runs anywhere in seconds."""
from __future__ import annotations

import numpy as np
import pytest

from bcfl_ids.config import DEVICE_IDS, NUM_CLASSES, NUM_FEATURES, NO_MIRAI_DEVICES
from bcfl_ids.simulator import ClientData


def make_clients(rows_per_class: int = 40, seed: int = 0) -> list[ClientData]:
    rng = np.random.default_rng(seed)
    centres = np.random.default_rng(123).normal(0, 3.0, size=(NUM_CLASSES, NUM_FEATURES)).astype(np.float32)
    clients = []
    for device in DEVICE_IDS:
        classes = [c for c in range(NUM_CLASSES) if not (device in NO_MIRAI_DEVICES and c >= 6)]

        def sample(k):
            xs, ys = [], []
            for c in classes:
                xs.append(centres[c] + rng.normal(0, 0.5, size=(k, NUM_FEATURES)).astype(np.float32) + device * 0.02)
                ys.append(np.full(k, c, dtype=np.uint8))
            return np.concatenate(xs).astype(np.float32), np.concatenate(ys)

        x_tr, y_tr = sample(rows_per_class)
        x_te, y_te = sample(rows_per_class // 2)
        clients.append(ClientData(device, x_tr, y_tr, x_te, y_te))
    return clients


@pytest.fixture(scope="session")
def clients():
    return make_clients()
