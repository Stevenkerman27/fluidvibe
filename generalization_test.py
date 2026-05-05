import itertools
import os
import argparse
import config
from eval import eval, load_policy

def main():
    parser = argparse.ArgumentParser(description="Generalization test: Evaluate a single Q-table on multiple environments.")
    # Default path for convenience as requested
    default_q_table = f"{config.SAVE_FOLDER}q_table_phi0.3_psi1.0_3000.npy"
    parser.add_argument("--q_table", type=str, default=default_q_table, help="Path to the Q-table (.npy) to evaluate.")
    args = parser.parse_args()

    q_table_path = args.q_table
    if not os.path.exists(q_table_path):
        print(f"Error: Q-table file {q_table_path} not found.")
        return

    print(f"Loading policy from {q_table_path}...")
    policy = load_policy(q_table_path)
    
    # Extract params from filename for prefixing
    # e.g., q_table_phi0.3_psi1.0_3000.npy -> gen_from_phi0.3_psi1.0_3000
    base_name = os.path.basename(q_table_path).replace("q_table_", "").replace(".npy", "")
    prefix = f"gen_from_{base_name}"

    print(f"Starting generalization sweep using policy {base_name}...")
    
    parameters = list(itertools.product(config.SWIMMER_SPEED, config.ALIGNMENT_TIMESCALE))
    
    for phi, psi in parameters:
        print(f"\n" + "="*50)
        print(f"EVALUATING ON: phi={phi}, psi={psi}")
        print("="*50)
        eval(
            policy=policy,
            swimmer_speed=phi,
            alignment_timescale=psi,
            n_episodes=config.N_EPISODES_EVAL,
            n_steps=config.N_STEPS,
            logging=True,
            make_plot=True,
            show_arrows=False,
            filename_prefix=prefix
        )

if __name__ == "__main__":
    main()
