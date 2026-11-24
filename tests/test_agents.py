"""
Tests for QNetwork, ReplayBuffer, and DQNAgent.
"""

import os
import tempfile
import numpy as np
import torch
import pytest

from src.agents.network import QNetwork
from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_agent import DQNAgent


def test_q_network():
    net = QNetwork(state_dim=12, action_dim=2, hidden_dims=[64, 64])
    x = torch.randn(4, 12)
    out = net(x)
    assert out.shape == (4, 2)
    assert not torch.isnan(out).any()


def test_replay_buffer():
    buf = ReplayBuffer(capacity=100, state_dim=12, device=torch.device("cpu"))
    assert len(buf) == 0

    s = np.zeros(12, dtype=np.float32)
    s_next = np.ones(12, dtype=np.float32)

    for i in range(25):
        buf.push(s, action=0, reward=1.0, next_state=s_next, done=False)

    assert len(buf) == 25
    states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
    assert states.shape == (8, 12)
    assert actions.shape == (8, 1)
    assert rewards.shape == (8, 1)
    assert next_states.shape == (8, 12)
    assert dones.shape == (8, 1)


def test_dqn_agent_learning_and_save_load():
    agent = DQNAgent(
        state_dim=12,
        action_dim=2,
        hidden_dims=[32, 32],
        learning_rate=0.001,
        batch_size=8,
        replay_buffer_capacity=100,
        epsilon_start=0.5,
        seed=42,
    )

    s = np.random.randn(12).astype(np.float32)
    action = agent.select_action(s, evaluate=False)
    assert action in (0, 1)

    action_eval = agent.select_action(s, evaluate=True)
    assert action_eval in (0, 1)

    # Push enough steps to trigger learn
    losses = []
    for _ in range(15):
        s_next = np.random.randn(12).astype(np.float32)
        r = float(np.random.randn())
        loss = agent.step(s, action, r, s_next, False)
        if loss is not None:
            losses.append(loss)
        s = s_next

    assert len(losses) > 0
    assert all(loss >= 0 for loss in losses)

    # Save and reload
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_dqn.pt")
        agent.save(model_path, metadata={"test_metric": 42.0})
        assert os.path.exists(model_path)

        new_agent = DQNAgent(
            state_dim=12,
            action_dim=2,
            hidden_dims=[32, 32],
        )
        meta = new_agent.load(model_path)
        assert meta.get("test_metric") == 42.0

        # Action from reloaded model
        act_loaded = new_agent.select_action(s, evaluate=True)
        assert act_loaded == action_eval
