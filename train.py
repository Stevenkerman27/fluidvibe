import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional
from agent_qlearning import QLearningAgent
from environments.base import Environment
import config


def save_q_table(q: np.ndarray, n_episodes: int) -> None:
    filename = config.Q_TABLE_PATH
    np.save(filename, q)


def train(
    env: Environment,
    n_episodes: int = config.N_EPISODES_TRAIN,
    n_steps: int = config.N_STEPS,
    save: bool = False,
    logging: bool = True,
    seed: Optional[int] = config.SEED,
) -> None:
    # Initialize agent with optimistic initialization
    q_initial = config.INITIAL_Q_VALUE * np.ones((12, 4))
    agent = QLearningAgent(q=q_initial, gamma=config.GAMMA, seed=seed)
    episode_returns = []
    mean_max_q_values = []  # Track mean max Q-values

    # Initialize plotting
    if logging:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        (line_return,) = ax1.plot([], [], label="Episode Return")
        (line_q,) = ax2.plot([], [], label="Mean Max Q-Value", color="orange")

        ax1.set_ylabel("Return")
        ax1.set_title("Training Progress")
        ax1.legend()

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Mean Max Q")
        ax2.legend()

    for episode in tqdm(range(n_episodes)):
        obs = env.reset()
        episode_return = 0
        for step in range(n_steps):
            action = agent.get_action(
                obs, epsilon=config.EPSILON_START * (1 - episode / n_episodes)
            )
            next_obs, reward = env.step(action)
            agent.update_q(
                obs, action, reward, next_obs, learning_rate=config.LEARNING_RATE
            )  # update based on experience
            obs = next_obs
            episode_return += reward

        episode_returns.append(episode_return)
        current_mean_max_q = np.mean(np.max(agent.q, axis=1))
        mean_max_q_values.append(current_mean_max_q)

        if logging:
            # Update plot every 50 episodes
            if episode % 50 == 0 or episode == n_episodes - 1:
                line_return.set_xdata(range(len(episode_returns)))
                line_return.set_ydata(episode_returns)
                line_q.set_xdata(range(len(mean_max_q_values)))
                line_q.set_ydata(mean_max_q_values)

                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()

                fig.canvas.draw()
                fig.canvas.flush_events()

            if episode % 100 == 0:
                print(f"Episode {episode} return: \t {episode_return}")
                print(f"Policy: \t {np.argmax(agent.q, axis=1)}.")
            elif episode == n_episodes - 1:
                print(f"Last episode return: \t {episode_return}")
                print(f"Policy: \t {np.argmax(agent.q, axis=1)}.")

    if save:
        save_q_table(agent.q, n_episodes)

    if logging:
        plt.ioff()
        if save:
            filename = config.RETURNS_PLOT_PATH
            plt.savefig(filename)
            print(f"Training progress plot saved to {filename}")
        plt.show()
