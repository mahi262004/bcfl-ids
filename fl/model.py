"""
Anomaly-detection model used by each federated learning client.

A small feedforward network is enough here: N-BaIoT traffic data is
tabular (115 statistical features per row), not images, so a
lightweight architecture keeps training fast on CPU and stays
realistic for eventual deployment on resource-constrained devices.
"""

import torch
import torch.nn as nn


class IntrusionDetectionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def get_model(input_dim: int, num_classes: int = 2) -> IntrusionDetectionModel:
    """Factory function so client.py / server.py build identical
    model architectures without duplicating the class definition."""
    return IntrusionDetectionModel(input_dim=input_dim, num_classes=num_classes)
