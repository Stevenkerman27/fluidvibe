import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict
import os

from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from agent_dqn import DQNAgent
import config

def plot_dqn_policy(
    n_episodes: int,
    positions: np.ndarray,
    positions_naive: np.ndarray,
    actions_taken: np.ndarray,
    plot_params: Dict[str, float],
    show_arrows: bool = True,
):
    """可视化 DQN 轨迹，逻辑参考 eval.py"""
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    delta_border = np.pi / 4

    x_min, x_max = -np.pi, 10.0
    y_min = -np.pi
    y_max = np.max([positions[:, 1, :], positions_naive[:, 1, :]]) + delta_border

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # 绘制涡量背景
    c = ax.pcolormesh(
        X,
        Y,
        np.cos(X) * np.cos(Y),
        cmap="coolwarm",
        shading="auto",
        alpha=0.3,
        rasterized=True,
    )
    plt.colorbar(c, ax=ax, shrink=0.5, label="vorticity")

    for episode in range(n_episodes):
        # 绘制训练后的轨迹
        plt.plot(
            positions[:, 0, episode],
            positions[:, 1, episode],
            color="xkcd:rich purple",
            alpha=0.3,
            label="DQN" if episode == 0 else ""
        )
        # 绘制朴素轨迹
        plt.plot(
            positions_naive[:, 0, episode],
            positions_naive[:, 1, episode],
            color="xkcd:medium grey",
            alpha=0.3,
            label="Naïve" if episode == 0 else ""
        )

        if show_arrows:
            # 每隔一定步数绘制动作箭头
            arrow_step = config.N_STEPS // 10
            u_vectors = np.array([1, 0, -1, 0])
            v_vectors = np.array([0, 1, 0, -1])
            
            for i in range(0, config.N_STEPS, arrow_step):
                action = actions_taken[i, episode]
                ax.quiver(
                    positions[i, 0, episode],
                    positions[i, 1, episode],
                    u_vectors[action],
                    v_vectors[action],
                    color="xkcd:rich purple",
                    alpha=0.6,
                    scale=25,
                    width=0.005,
                )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.gca().set_aspect("equal")
    plt.legend(loc="upper right")
    plt.title(f"phi={plot_params['phi']}, psi={plot_params['psi']}")
    plt.xlabel("x")
    plt.ylabel("y")
    
    save_path = f"{config.SAVE_FOLDER}eval_dqn_phi{plot_params['phi']}_psi{plot_params['psi']}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

import itertools

def eval_dqn(
    phi: float,
    psi: float,
    model_path: str,
    n_episodes: int = config.N_EPISODES_EVAL,
    n_steps: int = config.N_STEPS,
    logging: bool = True,
    make_plot: bool = True,
    show_arrows: bool = False,
):
    # 1. 环境初始化
    env = TaylorGreenContinuousEnvironment(
        dt=config.DT,
        swimmer_speed=phi,
        flow_speed=config.FLOW_SPEED,
        alignment_timescale=psi,
        seed=config.SEED,
        action_type="discrete"
    )
    
    env_naive = TaylorGreenContinuousEnvironment(
        dt=config.DT,
        swimmer_speed=phi,
        flow_speed=config.FLOW_SPEED,
        alignment_timescale=psi,
        seed=config.SEED,
        action_type="discrete"
    )

    # 2. 智能体初始化并加载模型
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        hidden_dim=config.DQN_HIDDEN_DIM,
        device=config.DQN_DEVICE
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Skipping evaluation for phi={phi}, psi={psi}.")
        return

    # 3. 评估循环
    rng = np.random.default_rng(seed=config.SEED)
    total_return = 0
    total_return_naive = 0
    
    positions = np.zeros([n_steps, 2, n_episodes])
    positions_naive = np.zeros([n_steps, 2, n_episodes])
    actions_taken = np.zeros([n_steps, n_episodes], dtype=int)

    for episode in range(n_episodes):
        # 统一初始状态
        pos_init = np.array([rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)])
        ori_init = rng.uniform(0, 2*np.pi)
        
        state = env.reset(pos_init.copy(), ori_init)
        _ = env_naive.reset(pos_init.copy(), ori_init)
        
        ep_ret = 0
        ep_ret_naive = 0
        
        for i in range(n_steps):
            # DQN 动作选择 (epsilon=0)
            action = agent.get_action(state, epsilon=0.0)
            actions_taken[i, episode] = action
            
            # 环境步进
            next_state, reward = env.step(action)
            # 朴素动作 (假设 action=1 是向上/优选方向)
            _, reward_naive = env_naive.step(1)
            
            state = next_state
            ep_ret += reward
            ep_ret_naive += reward_naive
            
            positions[i, :, episode] = env.swimmer_position
            positions_naive[i, :, episode] = env_naive.swimmer_position
            
        total_return += ep_ret
        total_return_naive += ep_ret_naive
        
        if logging and (episode + 1) % 10 == 0:
            print(f"Ep {episode+1} | DQN Return: {ep_ret:.2f} | Naive Return: {ep_ret_naive:.2f}")

    # 4. 统计结果
    mean_ret = total_return / n_episodes
    mean_ret_naive = total_return_naive / n_episodes
    print(f"\n[phi={phi}, psi={psi}]")
    print(f"Mean DQN Return: {mean_ret:.2f}")
    print(f"Mean Naive Return: {mean_ret_naive:.2f}")
    if mean_ret_naive != 0:
        print(f"Gain: {(mean_ret/mean_ret_naive - 1)*100:.2f}%")

    # 5. 绘图
    if make_plot:
        plot_params = {"phi": phi, "psi": psi}
        plot_dqn_policy(
            n_episodes, 
            positions, 
            positions_naive, 
            actions_taken, 
            plot_params,
            show_arrows=show_arrows
        )

if __name__ == "__main__":
    # 遍历所有配置组合
    phis = config.SWIMMER_SPEED if isinstance(config.SWIMMER_SPEED, list) else [config.SWIMMER_SPEED]
    psis = config.ALIGNMENT_TIMESCALE if isinstance(config.ALIGNMENT_TIMESCALE, list) else [config.ALIGNMENT_TIMESCALE]
    
    for phi, psi in itertools.product(phis, psis):
        model_path = f"{config.SAVE_FOLDER}dqn_phi{phi}_psi{psi}_{config.DQN_N_EPISODES_TRAIN}.pth"
        eval_dqn(phi=phi, psi=psi, model_path=model_path)
