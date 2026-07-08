from __future__ import annotations

import torch
from torch import nn


def build_mlp(input_features: int, hidden_sizes: list[int]) -> nn.Module:
    """Build an MLP with the given hidden layer widths.

    Shared by the training notebook and the export script so a checkpoint's
    hidden_sizes (saved alongside it) is always enough to reconstruct the
    exact module that produced it.
    """
    layers = []
    in_features = input_features
    for h in hidden_sizes:
        layers.append(nn.Linear(in_features, h))
        layers.append(nn.ReLU())
        in_features = h
    layers.append(nn.Linear(in_features, 2))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class BaselineMLP(nn.Module):
    """Small MLP used by the Task 1 starter baseline."""

    def __init__(self, input_features: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class OurMLP(nn.Module):
    """Our model"""

    def __init__(self, input_features: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
