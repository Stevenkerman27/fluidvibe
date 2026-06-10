import itertools
import os
import argparse
import config
from eval_dqn import eval_dqn
from agent_dqn import DQNAgent

def main():
    parser = argparse.ArgumentParser(description="Generalization test: Evaluate a single DQN model on multiple environments.")
    # Default path based on training defaults in config.py
    default_model = f"{config.SAVE_FOLDER}dqn_phi0.3_psi1.0_{config.DQN_N_EPISODES_TRAIN}.pth"
    parser.add_argument("--model", type=str, default=default_model, help="Path to the DQN model (.pth) to evaluate.")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        # Check if there's any dqn model in the folder to suggest
        models = [f for f in os.listdir(config.SAVE_FOLDER) if f.startswith("dqn_") and (f.endswith(".pth") or f.endswith(".cleanrl_model"))]
        if models:
            print(f"Available models in {config.SAVE_FOLDER}:")
            for m in models:
                print(f"  - {m}")
        return

    # Extract params from filename for prefixing
    # Support both .pth and .cleanrl_model
    base_name = os.path.basename(model_path).replace("dqn_", "")
    for ext in [".pth", ".cleanrl_model"]:
        base_name = base_name.replace(ext, "")
    prefix = f"gen_dqn_from_{base_name}"

    print(f"Starting generalization sweep using DQN model {base_name}...")
    
    # Use config parameters for the sweep
    parameters = list(itertools.product(config.SWIMMER_SPEED, config.ALIGNMENT_TIMESCALE))
    
    for phi, psi in parameters:
        print(f"\n" + "="*50)
        print(f"EVALUATING ON: phi={phi}, psi={psi}")
        print("="*50)
        eval_dqn(
            phi=phi,
            psi=psi,
            model_path=model_path,  # Pass path instead of agent object
            n_episodes=config.N_EPISODES_EVAL,
            n_steps=config.N_STEPS,
            logging=True,
            make_plot=True,
            show_arrows=False,
            filename_prefix=prefix
        )

if __name__ == "__main__":
    main()
