import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_dqn import DQNAgent
import config

def test_dqn_agent_initialization():
    state_dim = 2
    action_dim = 4
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=config.GAMMA,
        lr=config.LEARNING_RATE,
        batch_size=config.DQN_BATCH_SIZE,
        buffer_capacity=config.DQN_BUFFER_CAPACITY,
        hidden_dim=config.DQN_HIDDEN_DIM,
        target_update_freq=config.DQN_TARGET_UPDATE_FREQ,
        device="cpu"
    )
    
    assert agent.state_dim == state_dim
    assert agent.action_dim == action_dim
    assert agent.gamma == config.GAMMA
    assert agent.batch_size == config.DQN_BATCH_SIZE
    assert agent.target_update_freq == config.DQN_TARGET_UPDATE_FREQ
    
    # Check if QNetwork was initialized with correct dimensions
    # First layer should be (state_dim, hidden_dim)
    first_layer = agent.policy_net.net[0]
    assert first_layer.in_features == state_dim
    assert first_layer.out_features == config.DQN_HIDDEN_DIM

def test_dqn_agent_get_action():
    state_dim = 2
    action_dim = 4
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu"
    )
    
    state = np.random.rand(state_dim).astype(np.float32)
    action = agent.get_action(state, epsilon=0.0)
    assert 0 <= action < action_dim

def test_dqn_agent_remember_and_update():
    state_dim = 2
    action_dim = 4
    # Set small buffer and batch for quick testing
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=2,
        device="cpu"
    )
    
    state = np.random.rand(state_dim).astype(np.float32)
    next_state = np.random.rand(state_dim).astype(np.float32)
    agent.remember(state, 0, 1.0, next_state, False)
    
    # Update should return None because len(memory) < batch_size
    loss = agent.update()
    assert loss is None
    
    agent.remember(state, 1, 0.5, next_state, True)
    # Now it should update
    loss = agent.update()
    assert isinstance(loss, float)
