# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

# Custom Environment Registration
import environments.taylor_green_continuous
import config
import matplotlib.pyplot as plt


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = config.SEED
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = config.DQN_SAVE_MODEL
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "TaylorGreen-v0"
    """the id of the environment"""
    total_timesteps: int = config.DQN_N_EPISODES_TRAIN * config.N_STEPS
    """total timesteps of the experiments"""
    learning_rate: float = config.DQN_LEARNING_RATE
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = config.DQN_BUFFER_CAPACITY
    """the replay memory buffer size"""
    gamma: float = config.DQN_GAMMA
    """the discount factor gamma"""
    tau: float = config.DQN_TAU
    """the target network update rate"""
    target_network_frequency: int = config.DQN_TARGET_UPDATE_FREQ
    """the timesteps it takes to update the target network"""
    batch_size: int = config.DQN_BATCH_SIZE
    """the batch size of sample from the reply memory"""
    start_e: float = config.DQN_EPSILON_START
    """the starting epsilon for exploration"""
    end_e: float = config.DQN_EPSILON_END
    """the ending epsilon for exploration"""
    exploration_fraction: float = config.DQN_EPSILON_DECAY_DURATION / config.DQN_N_EPISODES_TRAIN
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = config.DQN_LEARNING_STARTS
    """timestep to start learning"""
    train_frequency: int = config.DQN_TRAIN_FREQ
    """the frequency of training"""

    # Custom environment parameters
    phi: float = config.SWIMMER_SPEED[0] if isinstance(config.SWIMMER_SPEED, list) else config.SWIMMER_SPEED
    psi: float = config.ALIGNMENT_TIMESCALE[0] if isinstance(config.ALIGNMENT_TIMESCALE, list) else config.ALIGNMENT_TIMESCALE
    hidden_dim: int = config.DQN_HIDDEN_DIM
    num_checkpoints: int = 5
    """number of checkpoints to save during training"""


def make_env(env_id, seed, idx, capture_video, run_name, phi=None, psi=None):
    if phi is None:
        phi = config.SWIMMER_SPEED[0] if isinstance(config.SWIMMER_SPEED, list) else config.SWIMMER_SPEED
    if psi is None:
        psi = config.ALIGNMENT_TIMESCALE[0] if isinstance(config.ALIGNMENT_TIMESCALE, list) else config.ALIGNMENT_TIMESCALE
        
    def thunk():
        env = gym.make(env_id, swimmer_speed=phi, alignment_timescale=psi)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int
    hidden_dim: int = config.DQN_HIDDEN_DIM

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__phi{args.phi}_psi{args.psi}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, phi=args.phi, psi=args.psi) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    q_network = QNetwork(action_dim=envs.single_action_space.n, hidden_dim=args.hidden_dim)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()
    episodic_returns = []
    episodic_y_dists = []

    if args.save_model:
        checkpoint_dir = os.path.join(config.SAVE_FOLDER, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_at_steps = np.linspace(0, args.total_timesteps, args.num_checkpoints, dtype=int)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    from tqdm import tqdm
    pbar = tqdm(range(args.total_timesteps))
    for global_step in pbar:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # Robustly handle different Gymnasium info formats
        finished_episodes = []
        
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if info and "episode" in info:
                    r = float(info["episode"]["r"])
                    l = int(info["episode"]["l"])
                    y = info.get("y_dist")
                    finished_episodes.append((r, l, y))
        
        # Backup if final_info is not used or only episode is present
        if not finished_episodes and "episode" in infos:
            if isinstance(infos["episode"], dict):
                # Handle RecordEpisodeStatistics in VectorEnv format
                if "_episode" in infos:
                    for i, finished in enumerate(infos["_episode"]):
                        if finished:
                            r = float(infos["episode"]["r"][i])
                            l = int(infos["episode"]["l"][i])
                            # Check top-level or final_info for y_dist
                            y = None
                            if "y_dist" in infos: y = infos["y_dist"][i]
                            elif "final_info" in infos and infos["final_info"][i]:
                                y = infos["final_info"][i].get("y_dist")
                            finished_episodes.append((r, l, y))
            else:
                # Older format
                for info in infos["episode"]:
                    if info and isinstance(info, dict) and "r" in info:
                        r, l = float(info["r"]), int(info["l"])
                        y = info.get("y_dist")
                        finished_episodes.append((r, l, y))

        for r, l, y in finished_episodes:
            pbar.set_description(f"Step {global_step} | Return: {r:.2f}")
            writer.add_scalar("charts/episodic_return", r, global_step)
            writer.add_scalar("charts/episodic_length", l, global_step)
            episodic_returns.append(r)
            if y is not None:
                # y is already the cumulative distance over the episode from the wrapper
                writer.add_scalar("charts/episodic_y_dist", y, global_step)
                episodic_y_dists.append(y)
            else:
                # If y_dist is missing, append 0 or handle it
                episodic_y_dists.append(0.0)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 5000 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

        if args.save_model and global_step in save_at_steps:
            ep = global_step // config.N_STEPS
            model_path = os.path.join(checkpoint_dir, f"dqn_jax_phi{args.phi}_psi{args.psi}_ep{ep}.cleanrl_model")
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))

    if args.save_model:
        model_path = f"{config.SAVE_FOLDER}dqn_jax_phi{args.phi}_psi{args.psi}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

        # Also save final model to checkpoints folder
        final_checkpoint_path = os.path.join(checkpoint_dir, f"dqn_jax_phi{args.phi}_psi{args.psi}_final.cleanrl_model")
        with open(final_checkpoint_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"final model checkpoint saved to {final_checkpoint_path}")
        
        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        window_size = 50
        def moving_average(x, w):
            if len(x) < w: return []
            return np.convolve(x, np.ones(w), 'valid') / w
        
        # Plot Return
        ax1.plot(episodic_returns, color='blue', alpha=0.3)
        returns_ma = moving_average(episodic_returns, window_size)
        if len(returns_ma) > 0:
            x_ma = range(window_size - 1, len(episodic_returns))
            ax1.plot(x_ma, returns_ma, color='blue', label='Episodic Return')
            
        # Draw vertical lines for checkpoints
        for i, step in enumerate(save_at_steps):
            cp_ep = step // config.N_STEPS
            ax1.axvline(x=cp_ep, color='black', linestyle='--', alpha=0.5, label='Checkpoint' if i == 0 else "")

        ax1.set_xlabel("Episode", fontsize=20)
        ax1.set_ylabel("Episodic Return", color='blue', fontsize=20)
        ax1.tick_params(axis='y', labelcolor='blue', labelsize=18)
        ax1.tick_params(axis='x', labelsize=18)
        
        if len(episodic_returns) > 0:
            ax1.set_ylim(min(episodic_returns), max(episodic_returns))
        
        # Plot Y Average Distance
        if len(episodic_y_dists) > 0:
            ax2 = ax1.twinx()
            ax2.plot(episodic_y_dists, color='green', alpha=0.3)
            y_dists_ma = moving_average(episodic_y_dists, window_size)
            if len(y_dists_ma) > 0:
                x_ma_y = range(window_size - 1, len(episodic_y_dists))
                ax2.plot(x_ma_y, y_dists_ma, color='green', label='Y dist travelled')
            ax2.set_ylabel("Total Y distance", color='green', fontsize=20)
            ax2.tick_params(axis='y', labelcolor='green', labelsize=18)
            ax2.set_ylim(min(episodic_y_dists), max(episodic_y_dists))

        plt.title(f"DQN Training Performance (phi={args.phi}, psi={args.psi})", fontsize=20)
        plot_path = f"{config.SAVE_FOLDER}returns_dqn_jax_phi{args.phi}_psi{args.psi}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

        eval_episodes = 50
        print(f"Running fast evaluation ({eval_episodes} episodes)...")
        eval_returns = []
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + 1000 + i, i, False, f"{run_name}-eval", phi=args.phi, psi=args.psi) for i in range(args.num_envs)]
        )
        eval_obs, _ = eval_envs.reset()
        eval_pbar = tqdm(total=eval_episodes, desc="Evaluating (Episodes)")
        
        eval_steps = 0
        while len(eval_returns) < eval_episodes:
            eval_steps += 1
            if eval_steps % 100 == 0:
                eval_pbar.set_postfix(steps=eval_steps)
            
            q_values = q_network.apply(q_state.params, eval_obs)
            eval_actions = q_values.argmax(axis=-1)
            eval_actions = jax.device_get(eval_actions)
            
            eval_obs, _, _, _, eval_infos = eval_envs.step(eval_actions)
            
            # Robust info parsing for both Gymnasium versions and wrappers
            if "final_info" in eval_infos:
                for info in eval_infos["final_info"]:
                    if info and "episode" in info:
                        eval_returns.append(info['episode']['r'])
                        eval_pbar.update(1)
            elif "episode" in eval_infos:
                if isinstance(eval_infos["episode"], dict):
                    if "_episode" in eval_infos:
                        for i, finished in enumerate(eval_infos["_episode"]):
                            if finished:
                                eval_returns.append(eval_infos["episode"]["r"][i])
                                eval_pbar.update(1)
                else:
                    for info in eval_infos["episode"]:
                        if info is not None and isinstance(info, dict) and "r" in info:
                            eval_returns.append(info["r"])
                            eval_pbar.update(1)
            
            if eval_steps > (eval_episodes + 2) * config.N_STEPS:
                print(f"\nSafety break: Evaluation exceeded expected steps ({eval_steps}).")
                break
        
        eval_pbar.close()
        
        for idx, episodic_return in enumerate(eval_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        print(f"Evaluation complete. Average return: {np.mean(eval_returns):.2f}")
        eval_envs.close()

    print("Closing environments and flusing logs...")
    envs.close()
    writer.close()
    if args.track:
        import wandb
        wandb.finish()
    print("Done.")
