"""The intrusion detector shared by all clients."""
from __future__ import annotations

import torch
from torch import nn

from .config import NUM_CLASSES, NUM_FEATURES


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def initial_weights(seed: int) -> torch.Tensor:
    """Server-generated starting weights, as a flat vector, so runs do not depend on client order."""
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    w = nn.utils.parameters_to_vector(Detector().parameters()).detach().clone()
    torch.random.set_rng_state(state)
    return w


def load_flat(model: nn.Module, w: torch.Tensor) -> None:
    nn.utils.vector_to_parameters(w, model.parameters())


def to_flat(model: nn.Module) -> torch.Tensor:
    return nn.utils.parameters_to_vector(model.parameters()).detach().clone()
