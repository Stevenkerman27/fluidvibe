import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import re
import config
from agent_dqn import DQNAgent

plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

def plot_dqn_policy(model_path, save_path=None):
    """
    Visualizes the DQN policy as a 2D map.
    X-axis: Vorticity (-1.2 to 1.2)
    Y-axis: Orientation (-pi/2 to 3pi/2)
    Color: Best Action (Right, Up, Left, Down)
    """
    # Initialize agent and load model
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        hidden_dim=config.DQN_HIDDEN_DIM,
        device="cpu"
    )
    agent.load(model_path)
    agent.policy_net.eval()

    # Define grid
    n_vort = 100
    n_ori = 100
    vort_range = np.linspace(-1.0, 1.0, n_vort)
    ori_range = np.linspace(-np.pi/2, 1.5 * np.pi, n_ori)
    
    V, O = np.meshgrid(vort_range, ori_range)
    policy_grid = np.zeros((n_ori, n_vort))

    # Compute best action for each point in grid
    for i in range(n_ori):
        for j in range(n_vort):
            state = np.array([vort_range[j], ori_range[i]])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q = agent.policy_net(state_tensor).numpy()[0]
                policy_grid[i, j] = np.argmax(q)

    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Custom discrete colormap for 4 actions
    from matplotlib.colors import ListedColormap
    # Actions: 0: Right, 1: Up, 2: Left, 3: Down
    # Colors: Red (R), Green (U), Blue (L), Yellow (D)
    colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']
    cmap = ListedColormap(colors)
    
    plt.pcolormesh(V, O, policy_grid, cmap=cmap, shading='auto')
    
    # Add labels and formatting
    plt.xlabel("Vorticity (scaled)", fontsize=18)
    plt.ylabel("Swimmer Orientation (radians)", fontsize=18)
    
    phi_match = re.search(r'phi([\d\.]+)', model_path)
    psi_match = re.search(r'psi([\d\.]+)', model_path)
    ep_match = re.search(r'_(\d+)\.pth$', model_path)
    phi_val = phi_match.group(1) if phi_match else "?"
    psi_val = psi_match.group(1) if psi_match else "?"
    ep_val = ep_match.group(1) if ep_match else "?"
    
    plt.title(f"DQN Policy Map (phi={phi_val}, psi={psi_val}, Ep={ep_val})\nColor: 0:R(red), 1:U(green), 2:L(blue), 3:D(yellow)", fontsize=20)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='0: Right (0°)'),
        Patch(facecolor=colors[1], label='1: Up (90°)'),
        Patch(facecolor=colors[2], label='2: Left (180°)'),
        Patch(facecolor=colors[3], label='3: Down (270°)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Add reference lines for vorticity threshold
    plt.axvline(x=config.VORTICITY_THRESHOLD, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=-config.VORTICITY_THRESHOLD, color='k', linestyle='--', alpha=0.5)
    
    # Add orientation labels (pi multiples)
    plt.yticks([-np.pi/2, 0, np.pi/2, np.pi, 1.5*np.pi], 
               [r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    model_dir = config.SAVE_FOLDER
    output_dir = os.path.join(model_dir, "plots_dqn")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_files = glob.glob(os.path.join(model_dir, "dqn_*.pth"))
    model_files.sort()
    
    if not model_files:
        print(f"No DQN model files found in {model_dir}")
    else:
        print(f"Found {len(model_files)} models. Generating plots...")
        for model_file in model_files:
            filename = os.path.basename(model_file).replace(".pth", ".png")
            save_path = os.path.join(output_dir, filename)
            plot_dqn_policy(model_file, save_path)
            print(f"Saved: {save_path}")
        print("All plots generated.")
