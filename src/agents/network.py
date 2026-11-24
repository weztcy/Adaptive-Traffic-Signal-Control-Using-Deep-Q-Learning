"""
Neural Network architecture for Deep Q-Network.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for approximating the action-value function Q(s, a).
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for each action given state."""
        return self.network(state)
