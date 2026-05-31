import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional
from environments.taylor_green import TaylorGreenEnvironment
import config


class QLearningAgent:
    def __init__(
        self, q: np.ndarray, gamma: float = 0.999, seed: Optional[int] = None
    ) -> None:
        """
        Args:
            q: Q-table for estimated value
            gamma: discount factor
            seed: Optional int for specifying random number generation
        """
        self.q = q
        self.gamma = gamma
        _, self._n_actions = np.shape(self.q)
        assert self.gamma > 0
        assert self.gamma <= 1.0
        self.rng = np.random.default_rng(seed=seed)

    def update_q(
        self,
        observation: int,
        action_chosen: int,
        reward: float,
        next_observation: int,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Function to update the Q function
        Args:
            observation: current observation
            action_chosen: action chosen
            reward: reward for observation, action pair
            next_observation: the next observation
            learning_rate: learning rate for Q-function update rule
        """

        self.q[observation, action_chosen] = (1 - learning_rate) * self.q[
            observation, action_chosen
        ] + learning_rate * (reward + self.gamma * np.max(self.q[next_observation]))

    def get_action(self, observation: int, epsilon: float) -> int:
        """
        Gets the action based on the current policy and observation.
        Args:
            observation: observation, provided by environment
            epsilon: decides the fraction of times the action is chosen randomly
        Returns:
            action
        """
        q_obs = self.q[observation]
        rand = self.rng.uniform(0.0, 1.0)

        # is_greedy decides whether action is greedy or is chosen randomly
        is_greedy = 1  # choose greedy action

        if rand < epsilon:
            is_greedy = 0  # choose random action

        if is_greedy:
            # greedy algorithm
            action = np.argmax(q_obs)  # choose the action with the highest q value
        else:
            # compute action randomly from all the actions
            actions_rand = self.rng.permutation(
                self._n_actions
            )  # create vector of actions, shuffled
            action = actions_rand[0]  # take the first index

        return action


if __name__ == "__main__":
    # Local Overrides
    SWIMMER_SPEED = 0.3
    ALIGNMENT_TIMESCALE = 1.0
    N_EPISODES = 1000
    N_CHECKPOINTS = 5
    
    # Checkpoint interval calculation
    save_at_episodes = np.linspace(0, N_EPISODES, N_CHECKPOINTS, dtype=int)
    
    # Paths
    CHECKPOINT_DIR = os.path.join(config.SAVE_FOLDER, "checkpoints")
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Initialize environment and agent
    env = TaylorGreenEnvironment(
        dt=config.DT,
        swimmer_speed=SWIMMER_SPEED,
        alignment_timescale=ALIGNMENT_TIMESCALE,
        seed=config.SEED,
    )
    
    q_initial = config.INITIAL_Q_VALUE * np.ones((12, 4))
    agent = QLearningAgent(q=q_initial, gamma=config.GAMMA, seed=config.SEED)
    
    print(f"Starting standalone training: phi={SWIMMER_SPEED}, psi={ALIGNMENT_TIMESCALE}")
    print(f"Episodes: {N_EPISODES}, Checkpoints: {N_CHECKPOINTS}")
    
    episode_returns = []
    episode_y_dists = []
    for episode in tqdm(range(N_EPISODES + 1)):
        # Checkpoint saving
        if episode in save_at_episodes:
            filename = os.path.join(
                CHECKPOINT_DIR, 
                f"q_table_phi{SWIMMER_SPEED}_psi{ALIGNMENT_TIMESCALE}_ep{episode}.npy"
            )
            np.save(filename, agent.q)

        if episode == N_EPISODES:
            break
            
        obs = env.reset()
        initial_y = env.swimmer_position[1]
        episode_return = 0
        for step in range(config.N_STEPS):
            epsilon = config.EPSILON_START * (1 - episode / N_EPISODES)
            action = agent.get_action(obs, epsilon=epsilon)
            next_obs, reward = env.step(action)
            agent.update_q(
                obs, action, reward, next_obs, learning_rate=config.LEARNING_RATE
            )
            obs = next_obs
            episode_return += reward
            
        episode_y_dist = env.swimmer_position[1] - initial_y
        episode_returns.append(episode_return)
        episode_y_dists.append(episode_y_dist)

    print("Training complete. Generating plot...")
    
    # Plotting
    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 1. Gray line "ticks" (raw returns)
    ax1.plot(episode_returns, color='gray', alpha=0.5, linewidth=0.5)
    
    # 2. Blue line (smoothed reward)
    window = int(N_EPISODES/10)
    if len(episode_returns) >= window:
        smoothed_returns = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, N_EPISODES), smoothed_returns, color='blue', label=f'Average Reward')
    else:
        ax1.plot(episode_returns, color='blue', label='Reward')
    
    # 3. Black dashed vertical lines (checkpoints)
    for i, cp_ep in enumerate(save_at_episodes):
        ax1.axvline(x=cp_ep, color='black', linestyle='--', alpha=0.7, label='Checkpoint' if i == 0 else "")

    ax1.set_xlabel('Episode', fontsize=16)
    ax1.set_ylabel('Return', color='blue', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)

    # Twin axis for Y displacement
    ax2 = ax1.twinx()
    ax2.plot(episode_y_dists, color='lightgreen', alpha=0.5, linewidth=0.5)
    if len(episode_y_dists) >= window:
        smoothed_y = np.convolve(episode_y_dists, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, N_EPISODES), smoothed_y, color='green', label=f'Average Y Dist')
    else:
        ax2.plot(episode_y_dists, color='green', label='Y Dist')
        
    ax2.set_ylabel('Y Displacement', color='green', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

    plt.title(f'Training Progress (phi={SWIMMER_SPEED}, psi={ALIGNMENT_TIMESCALE})', fontsize=18)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plot_filename = os.path.join(CHECKPOINT_DIR, f"returns_phi{SWIMMER_SPEED}_psi{ALIGNMENT_TIMESCALE}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()
