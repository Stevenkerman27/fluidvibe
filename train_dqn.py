import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from agent_dqn import DQNAgent
import config

import itertools

def train_dqn(phi, psi):
    # 1. 初始化环境 (使用连续观测和离散动作)
    env = TaylorGreenContinuousEnvironment(
        dt=config.DT,
        swimmer_speed=phi,
        flow_speed=config.FLOW_SPEED,
        alignment_timescale=psi,
        seed=config.SEED,
        action_type="discrete"
    )

    # 2. 初始化 DQN 智能体
    agent = DQNAgent(
        state_dim=2,      # [vorticity, orientation]
        action_dim=4,     # 0, 90, 180, 270 degrees
        gamma=config.DQN_GAMMA,
        lr=config.DQN_LEARNING_RATE,
        batch_size=config.DQN_BATCH_SIZE,
        buffer_capacity=config.DQN_BUFFER_CAPACITY,
        hidden_dim=config.DQN_HIDDEN_DIM,
        target_update_freq=config.DQN_TARGET_UPDATE_FREQ,
        device=config.DQN_DEVICE,
        seed=config.SEED
    )

    episode_returns = []
    epsilon = config.DQN_EPSILON_START
    # 使用 config 中的 epsilon 衰减策略
    epsilon_decay = config.DQN_EPSILON_DECAY
    min_epsilon = config.DQN_MIN_EPSILON

    print(f"\nStarting DQN training for phi={phi}, psi={psi} on {agent.device}...")

    # 3. 训练循环
    pbar = tqdm(range(config.DQN_N_EPISODES_TRAIN))
    for episode in pbar:
        state = env.reset()
        cumulative_reward = 0
        
        for step in range(config.N_STEPS):
            # 选择动作
            action = agent.get_action(state, epsilon)
            
            # 执行步进
            next_state, reward = env.step(action)
            
            # 记录经验
            # 在这个任务中，没有明确的 "done" 状态（固定步数），所以设为 False
            agent.remember(state, action, reward, next_state, False)
            
            # 更新网络
            loss = agent.update()
            
            state = next_state
            cumulative_reward += reward

        episode_returns.append(cumulative_reward)
        
        # 衰减 epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # 更新进度条信息
        pbar.set_description(f"Ep {episode+1} | Return: {cumulative_reward:.2f} | Eps: {epsilon:.2f}")

    # 4. 保存模型和结果
    model_path = f"{config.SAVE_FOLDER}dqn_phi{phi}_psi{psi}_{config.DQN_N_EPISODES_TRAIN}.pth"
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # 绘制回报曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward (Vertical Displacement)")
    plt.title(f"DQN Training Performance (phi={phi}, psi={psi})")
    plot_path = f"{config.SAVE_FOLDER}returns_dqn_phi{phi}_psi{psi}_{config.DQN_N_EPISODES_TRAIN}.png"
    plt.savefig(plot_path)
    plt.close() # Close figure to free memory
    print(f"Plot saved to {plot_path}")

    return episode_returns

if __name__ == "__main__":
    # 遍历所有配置组合
    phis = config.SWIMMER_SPEED if isinstance(config.SWIMMER_SPEED, list) else [config.SWIMMER_SPEED]
    psis = config.ALIGNMENT_TIMESCALE if isinstance(config.ALIGNMENT_TIMESCALE, list) else [config.ALIGNMENT_TIMESCALE]
    
    for phi, psi in itertools.product(phis, psis):
        train_dqn(phi, psi)
