import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from agent_dqn import DQNAgent
import config

def train_dqn():
    # 1. 初始化环境 (使用连续观测和离散动作)
    env = TaylorGreenContinuousEnvironment(
        dt=config.DT,
        swimmer_speed=config.SWIMMER_SPEED,
        flow_speed=config.FLOW_SPEED,
        alignment_timescale=config.ALIGNMENT_TIMESCALE,
        seed=config.SEED,
        action_type="discrete"
    )

    # 2. 初始化 DQN 智能体
    agent = DQNAgent(
        state_dim=2,      # [vorticity, orientation]
        action_dim=4,     # 0, 90, 180, 270 degrees
        gamma=config.GAMMA,
        lr=config.LEARNING_RATE,
        batch_size=config.DQN_BATCH_SIZE,
        buffer_capacity=config.DQN_BUFFER_CAPACITY,
        hidden_dim=config.DQN_HIDDEN_DIM,
        target_update_freq=config.DQN_TARGET_UPDATE_FREQ,
        device=config.DQN_DEVICE,
        seed=config.SEED
    )

    episode_returns = []
    epsilon = config.EPSILON_START
    # 使用 config 中的 epsilon 衰减策略
    epsilon_decay = config.DQN_EPSILON_DECAY
    min_epsilon = config.DQN_MIN_EPSILON

    print(f"Starting DQN training on {agent.device}...")

    # 3. 训练循环
    pbar = tqdm(range(config.N_EPISODES_TRAIN))
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
    model_path = config.Q_TABLE_PATH.replace(".npy", ".pth")
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # 绘制回报曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward (Vertical Displacement)")
    plt.title("DQN Training Performance")
    plot_path = config.RETURNS_PLOT_PATH.replace(".png", "_dqn.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    return episode_returns

if __name__ == "__main__":
    train_dqn()
