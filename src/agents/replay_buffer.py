"""
Experience Replay Buffer for DQN.
Stores transitions (state, action, reward, next_state, done) and samples uniform mini-batches.
"""

from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size cyclic replay buffer for storing experience tuples.
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.device = device

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a new experience transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions uniformly at random."""
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size
