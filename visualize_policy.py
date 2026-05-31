import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import config

plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

def plot_policy(q_path, save_path=None):
    """
    Visualizes the Q-table policy.
    Rows of Q-table (States):
    0-3: Vorticity < -T   (Right, Up, Left, Down)
    4-7: |Vorticity| <= T (Right, Up, Left, Down)
    8-11: Vorticity > T   (Right, Up, Left, Down)
    
    Columns of Q-table (Actions):
    0: Right (0 deg), 1: Up (90 deg), 2: Left (180 deg), 3: Down (270 deg)
    """
    q = np.load(q_path)
    n_states, n_actions = q.shape
    
    # Increase base font size globally for this plot
    plt.rcParams.update({'font.size': 14}) 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 14))
    
    # Labels
    vort_labels = ["V<-T", "|V|<=T", "V>T"]
    ori_labels = ["Right", "Up", "Left", "Down"]
    action_labels = ["Right", "Up", "Left", "Down"]
    
    # --- Subplot 1: Policy Grid (Arrows in Boxes) ---
    best_actions = np.argmax(q, axis=1).reshape(3, 4)
    
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(vort_labels)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(ori_labels) 
    
    # Secondary ticks for box borders
    ax1.set_xticks(np.arange(-0.5, 3.5, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, 4.5, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax1.tick_params(which='both', length=0)
    
    for v_idx in range(3):
        for o_idx in range(4):
            action = best_actions[v_idx, o_idx]
            y, x = o_idx, v_idx
            
            # Enlarge arrow parameters
            dx, dy = 0, 0
            head_size = 0.3 # Increased
            arrow_len = 0.8 # Increased
            if action == 0: dx = arrow_len
            elif action == 1: dy = arrow_len
            elif action == 2: dx = -arrow_len
            elif action == 3: dy = -arrow_len
            
            ax1.arrow(x - dx/2, y - dy/2, dx, dy, 
                      head_width=head_size, head_length=head_size, 
                      fc='blue', ec='blue', length_includes_head=True,
                      linewidth=2) # Thicker arrow
            
    import re
    phi_match = re.search(r'phi([\d\.]+)', q_path)
    psi_match = re.search(r'psi([\d\.]+)', q_path)
    ep_match = re.search(r'(?:ep)?(\d+)\.npy$', q_path)
    
    phi_val = phi_match.group(1) if phi_match else "?"
    psi_val = psi_match.group(1) if psi_match else "?"
    title_suffix = f" (Episode {ep_match.group(1)})" if ep_match else ""

    ax1.set_title(f"Greedy Policy{title_suffix}", fontsize=24, pad=15)
    ax1.set_xlabel("Vorticity", fontsize=26)
    ax1.set_ylabel("Swimmer Orientation", fontsize=26)

    # --- Subplot 2: Q-Value Heatmap ---
    im = ax2.imshow(q, cmap="YlGnBu", aspect='auto')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, format='%.0f')
    
    state_labels_short = [f"{v} | {o}" for v in ["Neg", "Zero", "Pos"] for o in ["R", "U", "L", "D"]]
    ax2.set_yticks(range(n_states))
    ax2.set_yticklabels(state_labels_short)
    ax2.set_xticks(range(n_actions))
    ax2.set_xticklabels(action_labels)
    ax2.set_title(f"Q-Table Heatmap (phi={phi_val}, psi={psi_val})", fontsize=24, pad=15)
    ax2.set_xlabel("Actions", fontsize=22)
    ax2.set_ylabel("States", fontsize=22)

    # Add text annotations
    for i in range(n_states):
        for j in range(n_actions):
            text_color = "red" if q[i, j] > np.max(q) * 0.7 else "black"
            ax2.text(j, i, f"{q[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=18)

    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    checkpoint_dir = os.path.join(config.SAVE_FOLDER, "checkpoints")
    output_dir = os.path.join(checkpoint_dir, "plots")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    q_files = glob.glob(os.path.join(checkpoint_dir, "*.npy"))
    q_files.sort()
    
    if not q_files:
        print(f"No Q-table files found in {checkpoint_dir}")
    else:
        print(f"Found {len(q_files)} checkpoints. Generating plots...")
        for q_file in q_files:
            filename = os.path.basename(q_file).replace(".npy", ".png")
            save_path = os.path.join(output_dir, filename)
            plot_policy(q_file, save_path)
            print(f"Saved: {save_path}")
        print("All plots generated.")
