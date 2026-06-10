import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import glob
import re
import config

# JAX/Flax dependencies
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32

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

def plot_dqn_policy(model_path, save_path=None):
    """
    Visualizes the JAX DQN policy as a 2D map.
    X-axis: Vorticity (-1.0 to 1.0)
    Y-axis: Orientation (-pi/2 to 3pi/2)
    Color: Best Action (Right, Up, Left, Down)
    """
    # Initialize JAX network
    q_network = QNetwork(action_dim=4, hidden_dim=config.DQN_HIDDEN_DIM)
    obs_dummy = jnp.zeros((1, 3))
    variables = q_network.init(jax.random.PRNGKey(0), obs_dummy)
    
    # Load parameters
    with open(model_path, "rb") as f:
        params_bytes = f.read()
    
    # Restore the state dict and extract the 'params' key
    state_dict = flax.serialization.msgpack_restore(params_bytes)
    if 'params' in state_dict:
        # q_state.params in training was a dict containing 'params'
        params = flax.serialization.from_state_dict(variables['params'], state_dict['params'])
    else:
        # Fallback if it was saved differently
        params = flax.serialization.from_state_dict(variables['params'], state_dict)
    
    # JIT the apply function
    @jax.jit
    def get_best_action(obs):
        q_values = q_network.apply({'params': params}, obs)
        return q_values.argmax(axis=-1)

    # Define grid
    n_vort = 150
    n_ori = 150
    vort_range = np.linspace(-1.0, 1.0, n_vort)
    ori_range = np.linspace(-np.pi/2, 1.5 * np.pi, n_ori)
    
    V, O = np.meshgrid(vort_range, ori_range)
    
    # Vectorized computation for the entire grid
    # Grid shape: (n_ori, n_vort), we need to create (n_ori * n_vort, 3)
    # Order of features: [vorticity, sin_orientation, cos_orientation]
    vort_flat = V.flatten()
    ori_flat = O.flatten()
    states = jnp.stack([vort_flat, jnp.sin(ori_flat), jnp.cos(ori_flat)], axis=-1)
    
    # Inference in batches to avoid OOM if grid is huge, but 150x150 is fine
    actions_flat = get_best_action(states)
    policy_grid = jax.device_get(actions_flat).reshape((n_ori, n_vort))

    # Plotting
    plt.figure(figsize=(8, 9))
    
    # Okabe-Ito palette: Blue, Orange, Bluish Green, Reddish Purple
    # Actions: 0: Right (0), 1: Up (90), 2: Left (180), 3: Down (270)
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    cmap = ListedColormap(colors)
    
    plt.pcolormesh(V, O, policy_grid, cmap=cmap, shading='auto', vmin=0, vmax=3)
    
    # Add labels and formatting
    plt.xlabel("Vorticity (scaled)", fontsize=44)
    plt.ylabel("Swimmer Orientation", fontsize=44)
    
    phi_match = re.search(r'phi([\d\.]+)', model_path)
    psi_match = re.search(r'psi([\d\.]+)', model_path)
    ep_match = re.search(r'_ep(\d+)', model_path)
    final_match = re.search(r'_final', model_path)
    
    phi_val = phi_match.group(1) if phi_match else "?"
    psi_val = psi_match.group(1) if psi_match else "?"
    
    ep_suffix = ""
    if ep_match:
        ep_suffix = f", ep={ep_match.group(1)}"
    elif final_match:
        ep_suffix = ", final"
    
    plt.title(f"phi{phi_val}, psi{psi_val}{ep_suffix}", fontsize=32)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Right'),
        Patch(facecolor=colors[1], label='Up'),
        Patch(facecolor=colors[2], label='Left'),
        Patch(facecolor=colors[3], label='Down')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=24)

    # Add reference lines for vorticity threshold
    plt.axvline(x=config.VORTICITY_THRESHOLD, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=-config.VORTICITY_THRESHOLD, color='k', linestyle='--', alpha=0.5)
    
    # Add orientation labels (pi multiples)
    plt.yticks([-np.pi/2, 0, np.pi/2, np.pi, 1.5*np.pi], 
               ['Down', 'Right', 'Up', 'Left', 'Down'])

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 1. Process main models
    model_dir = config.SAVE_FOLDER
    output_dir = os.path.join(model_dir, "plots_dqn_jax")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_files = glob.glob(os.path.join(model_dir, "dqn_jax_*.cleanrl_model"))
    
    if model_files:
        print(f"Found {len(model_files)} main JAX models. Generating plots...")
        for model_file in model_files:
            filename = os.path.basename(model_file).replace(".cleanrl_model", ".png")
            save_path = os.path.join(output_dir, filename)
            plot_dqn_policy(model_file, save_path)

    # 2. Process checkpoint models
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    checkpoint_output_dir = os.path.join(checkpoint_dir, "plots")
    
    if os.path.exists(checkpoint_dir):
        if not os.path.exists(checkpoint_output_dir):
            os.makedirs(checkpoint_output_dir)
            
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "dqn_jax_*.cleanrl_model"))
        if checkpoint_files:
            print(f"Found {len(checkpoint_files)} checkpoint JAX models. Generating plots...")
            for cp_file in checkpoint_files:
                filename = os.path.basename(cp_file).replace(".cleanrl_model", ".png")
                save_path = os.path.join(checkpoint_output_dir, filename)
                plot_dqn_policy(cp_file, save_path)
    
    print("Done.")
