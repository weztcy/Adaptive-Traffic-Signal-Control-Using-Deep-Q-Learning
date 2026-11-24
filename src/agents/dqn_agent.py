"""
Deep Q-Network (DQN) Agent.
Features:
- Policy Q-network and Target Q-network
- Experience replay buffer
- Epsilon-greedy exploration with linear/exponential decay
- Huber loss (SmoothL1Loss) optimization with gradient clipping
- Periodic target network updates
- Checkpoint save and load
"""

import os
import json
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.network import QNetwork
from src.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN Agent for traffic signal control."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        replay_buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_interval: int = 400,
        tau: float = 0.005,
        use_soft_update: bool = False,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.use_soft_update = use_soft_update

        # Q-Network and Target Network
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_capacity,
            state_dim=state_dim,
            device=self.device,
        )

        self.total_steps = 0
        self.update_count = 0

    def select_action(
        self, state: np.ndarray, evaluate: bool = False, legal_actions: Optional[List[int]] = None
    ) -> int:
        """
        Select action using epsilon-greedy exploration.
        If evaluate=True, always exploit (greedy action).
        """
        if legal_actions is None:
            legal_actions = list(range(self.action_dim))

        if not evaluate and np.random.random() < self.epsilon:
            return int(np.random.choice(legal_actions))

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        self.q_network.train()

        # If restricted legal actions, mask illegal actions with -infinity
        masked_q = np.full(self.action_dim, -np.inf)
        for a in legal_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store experience and perform optimization step if sufficient samples exist."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Decay epsilon linearly
        fraction = min(1.0, float(self.total_steps) / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

        # Optimize policy network if buffer has enough samples
        loss_val = None
        if len(self.replay_buffer) >= self.batch_size:
            loss_val = self.learn()

        # Target network update
        if self.use_soft_update:
            self._soft_update(self.tau)
        elif self.total_steps % self.target_update_interval == 0:
            self._hard_update()

        return loss_val

    def learn(self) -> float:
        """Sample batch, compute Bellman loss, and update network parameters."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current Q-values: Q(s, a)
        q_eval = self.q_network(states).gather(1, actions)

        # Target Q-values: y_t = r + gamma * (1 - done) * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_network(next_states).max(1, keepdim=True)[0]
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        # Compute Huber loss
        loss = self.loss_fn(q_eval, q_target)

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to ensure numerical stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_count += 1
        return float(loss.item())

    def _hard_update(self) -> None:
        """Copy weights from Q-network to Target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _soft_update(self, tau: float) -> None:
        """Soft update target network parameters: theta_target = tau * theta_local + (1 - tau) * theta_target."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights and metadata."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load model weights and return metadata."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        return checkpoint.get("metadata", {})
