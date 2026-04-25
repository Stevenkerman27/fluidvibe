import time
import numpy as np
import torch
from tqdm import tqdm
from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from agent_dqn import DQNAgent
import config

def profile_dqn():
    env = TaylorGreenContinuousEnvironment(
        dt=config.DT,
        swimmer_speed=config.SWIMMER_SPEED,
        flow_speed=config.FLOW_SPEED,
        alignment_timescale=config.ALIGNMENT_TIMESCALE,
        seed=config.SEED,
        action_type="discrete"
    )

    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        gamma=config.GAMMA,
        lr=config.LEARNING_RATE,
        batch_size=config.DQN_BATCH_SIZE,
        buffer_capacity=config.DQN_BUFFER_CAPACITY,
        hidden_dim=config.DQN_HIDDEN_DIM,
        target_update_freq=config.DQN_TARGET_UPDATE_FREQ,
        device='cpu',
        seed=config.SEED
    )

    n_episodes = 5
    n_steps = config.N_STEPS
    epsilon = 0.1

    times = {
        "env_step": 0,
        "agent_get_action": 0,
        "agent_remember": 0,
        "agent_update": 0,
        "sampling": 0,
        "transfer": 0,
        "compute": 0
    }

    # Monkey patch update to measure internal times
    original_update = agent.update
    def timed_update():
        if len(agent.memory) < agent.batch_size:
            return original_update()
        
        t0 = time.time()
        states, actions, rewards, next_states, dones = agent.memory.sample(agent.batch_size)
        times["sampling"] += time.time() - t0

        t1 = time.time()
        states = torch.FloatTensor(states).to(agent.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(agent.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(agent.device)
        next_states = torch.FloatTensor(next_states).to(agent.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(agent.device)
        times["transfer"] += time.time() - t1

        t2 = time.time()
        current_q = agent.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = agent.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * agent.gamma * next_q
        loss = torch.nn.MSELoss()(current_q, target_q)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        agent.steps_done += 1
        if agent.steps_done % agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        times["compute"] += time.time() - t2
        
        return loss.item()

    agent.update = timed_update

    print(f"Profiling {n_episodes} episodes...")
    start_total = time.time()
    for episode in range(n_episodes):
        state = env.reset()
        for step in range(n_steps):
            t = time.time()
            action = agent.get_action(state, epsilon)
            times["agent_get_action"] += time.time() - t

            t = time.time()
            next_state, reward = env.step(action)
            times["env_step"] += time.time() - t

            t = time.time()
            agent.remember(state, action, reward, next_state, False)
            times["agent_remember"] += time.time() - t

            t = time.time()
            agent.update()
            times["agent_update"] += time.time() - t

            state = next_state
    
    total_time = time.time() - start_total
    print(f"\nTotal time: {total_time:.2f}s")
    for k, v in times.items():
        if k in ["sampling", "transfer", "compute"]:
             print(f"  {k:20}: {v:.4f}s ({v/total_time*100:5.2f}%) [part of agent_update]")
        else:
            print(f"  {k:20}: {v:.4f}s ({v/total_time*100:5.2f}%)")

if __name__ == "__main__":
    profile_dqn()
