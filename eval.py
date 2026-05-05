import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from typing import Dict
from environments.taylor_green import TaylorGreenEnvironment
import config
import os


def load_policy(filename: str) -> np.ndarray:
    q_table = np.load(filename)
    policy = np.argmax(q_table, axis=1)
    return policy


def plot_policy(
    n_episodes: int,
    positions: np.ndarray,
    positions_naive: np.ndarray,
    actions_taken: np.ndarray,
    plot_params: Dict[str, float],
    show_arrows: bool = True,
    filename_prefix: str = ""
):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5, 8))
    ax2 = plt.subplot(111)
    
    # Set tick spacing
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    delta_border = np.pi / 4

    x_min, x_max = -np.pi, 10.0
    y_min = -np.pi
    y_max = np.max([positions[:, 1, :], positions_naive[:, 1, :]]) + delta_border

    x = np.linspace(x_min, x_max, int(100 * (x_max - x_min)))
    y = np.linspace(y_min, y_max, int(100 * (y_max - y_min)))
    X, Y = np.meshgrid(x, y)

    vorticity = np.cos(X) * np.cos(Y)
    # Discretize based on the threshold used in agent's observation
    vorticity_discrete = np.zeros_like(vorticity)
    vorticity_discrete[vorticity < -config.VORTICITY_THRESHOLD] = -1
    vorticity_discrete[vorticity > config.VORTICITY_THRESHOLD] = 1

    # Use a discrete colormap: [Neg (blue), Zero (white), Pos (red)]
    discrete_cmap = ListedColormap(['#67a9cf', '#f7f7f7', '#ef8a62'])

    c = ax2.pcolormesh(
        X,
        Y,
        vorticity_discrete,
        cmap=discrete_cmap,
        shading="auto",
        alpha=0.3,
        rasterized=True,
        vmin=-1, 
        vmax=1
    )
    cb = plt.colorbar(c, ax=ax2, shrink=0.25, label="vorticity (perceived)", anchor=(0, 1.0))
    cb.set_ticks([-0.66, 0, 0.66]) # Align ticks with the centers of the discrete bands
    cb.set_ticklabels(["Neg", "Zero", "Pos"])


    for episode in range(n_episodes):
        plt.plot(
            positions[:, 0, episode],
            positions[:, 1, episode],
            color="xkcd:rich purple",
            alpha=0.2,
        )
        plt.plot(
            positions_naive[:, 0, episode],
            positions_naive[:, 1, episode],
            color="xkcd:medium grey",
            alpha=0.2,
        )

        if show_arrows:
            # Plot arrows for actions every 10 steps
            arrow_step = 15
            # Action mapping to vectors: 0: Right, 1: Up, 2: Left, 3: Down
            u = np.array([1, 0, -1, 0])
            v = np.array([0, 1, 0, -1])
            
            for i in range(0, len(positions[:, 0, episode]), arrow_step):
                action = actions_taken[i, episode]
                ax2.quiver(
                    positions[i, 0, episode],
                    positions[i, 1, episode],
                    u[action],
                    v[action],
                    color="xkcd:rich purple",
                    alpha=0.6,
                    scale=17,
                    width=0.01,
                )

        plt.plot(
            positions[-1, 0, episode],
            positions[-1, 1, episode],
            "o",
            markersize=2,
            markeredgecolor="xkcd:rich purple",
            markerfacecolor="none",
            alpha=0.7,
            label="trained" if episode == 0 else "",
        )
        plt.plot(
            positions_naive[-1, 0, episode],
            positions_naive[-1, 1, episode],
            "o",
            markersize=2,
            markeredgecolor="xkcd:medium grey",
            markerfacecolor="none",
            alpha=0.7,
            label="naïve" if episode == 0 else "",
        )

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    plt.gca().set_aspect("equal")
    plt.legend(bbox_to_anchor=(1.05, 0.15), loc="lower left")
    
    plt.title(rf"$\phi={plot_params['phi']}, \psi={plot_params['psi']}$")
    plt.tight_layout()
    
    # Ensure pics directory exists
    pics_dir = "pics"
    if not os.path.exists(pics_dir):
        os.makedirs(pics_dir)
        
    save_name = f"phi{plot_params['phi']}_psi{plot_params['psi']}_{plot_params['mean_y']:.2f}({plot_params['mean_y_naive']:.2f}).png"
    if filename_prefix:
        save_name = f"{filename_prefix}_{save_name}"
    
    save_path = os.path.join(pics_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()


def eval(
    policy: list,
    swimmer_speed: float = config.SWIMMER_SPEED,
    alignment_timescale: float = config.ALIGNMENT_TIMESCALE,
    n_episodes: int = config.N_EPISODES_EVAL,
    n_steps: int = config.N_STEPS,
    logging: bool = True,
    make_plot: bool = False,
    show_arrows: bool = True,
    filename_prefix: str = ""
) -> None:
    rng = np.random.default_rng(seed=config.SEED)
    
    # Use config.DT to ensure consistency with training
    env = TaylorGreenEnvironment(
        dt=config.DT,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=config.SEED,
    )  # instantiate environment

    env_naive = TaylorGreenEnvironment(
        dt=config.DT,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=config.SEED,
    )  # instantiate environment for naïve swimmer

    print(f"The trained policy is {policy}.")
    total_episode_return = 0
    total_episode_return_naive = 0
    total_y_dist = 0
    total_y_dist_naive = 0
    
    positions = np.zeros([n_steps, 2, n_episodes])
    positions_naive = np.zeros([n_steps, 2, n_episodes])
    actions_taken = np.zeros([n_steps, n_episodes], dtype=int)

    for episode in range(n_episodes):
        episode_return = 0
        episode_return_naive = 0

        position_initial = np.array(
            [rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)]
        )
        orientation_initial = rng.uniform(0, 2 * np.pi)

        obs = env.reset(position_initial.copy(), orientation_initial)
        _ = env_naive.reset(position_initial.copy(), orientation_initial)

        for i in range(n_steps):

            action = policy[obs]
            actions_taken[i, episode] = action
            next_obs, reward = env.step(action)
            _, reward_naive = env_naive.step(1)

            obs = next_obs

            episode_return += reward
            episode_return_naive += reward_naive

            positions[i, :, episode] = env.swimmer_position
            positions_naive[i, :, episode] = env_naive.swimmer_position

        total_episode_return += episode_return
        total_episode_return_naive += episode_return_naive
        
        # Calculate actual y displacement for this episode
        y_dist = env.swimmer_position[1] - position_initial[1]
        y_dist_naive = env_naive.swimmer_position[1] - position_initial[1]
        total_y_dist += y_dist
        total_y_dist_naive += y_dist_naive

        if logging:
            print(
                f"Episode {episode+1} \t return: {episode_return:.2f} \t naive return: {episode_return_naive:.2f} \t y-dist: {y_dist:.2f}"
            )

    mean_y_dist = total_y_dist / n_episodes
    mean_y_dist_naive = total_y_dist_naive / n_episodes

    if logging:
        print(f"\nMean return (trained): {total_episode_return/n_episodes:.2f}")
        print(f"Mean return (naive):   {total_episode_return_naive/n_episodes:.2f}")
        print(f"Mean y-distance (trained): {mean_y_dist:.2f}")
        print(f"Mean y-distance (naive):   {mean_y_dist_naive:.2f}")

    if total_episode_return_naive != 0:
        print(f"Return gain: {total_episode_return/total_episode_return_naive-1:.2%}")
    if mean_y_dist_naive != 0:
        print(f"y-distance gain: {mean_y_dist/mean_y_dist_naive-1:.2%}")

    if make_plot:
        plot_params = {
            "phi": env.swimmer_speed,
            "psi": env.alignment_timescale,
            "mean_y": mean_y_dist,
            "mean_y_naive": mean_y_dist_naive,
        }
        plot_policy(
            n_episodes,
            positions,
            positions_naive,
            actions_taken,
            plot_params,
            show_arrows=show_arrows,
            filename_prefix=filename_prefix
        )


if __name__ == "__main__":
    # Define parameters for standalone evaluation
    phi = 0.2
    psi = 1.2
    q_table_path = f"{config.SAVE_FOLDER}q_table_phi{phi}_psi{psi}_{config.N_EPISODES_TRAIN}.npy"
    
    print(f"Loading policy from {q_table_path}...")
    try:
        policy = load_policy(q_table_path)
        print(f"Loaded policy: {policy}")
    except FileNotFoundError:
        print(f"Warning: {q_table_path} not found. Using default dummy policy.")
        policy = [2, 2, 1, 2, 1, 1, 1, 3, 1, 0, 3, 0]

    eval(
        policy=policy,
        swimmer_speed=phi,
        alignment_timescale=psi,
        make_plot=True,
    )
