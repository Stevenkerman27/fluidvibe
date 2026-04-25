import itertools
import os
import argparse
from environments.taylor_green import TaylorGreenEnvironment
from train import train
from eval import eval, load_policy
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FluidVibe: Train or Evaluate RL agents in Taylor-Green Vortex.")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode: skip training and evaluate existing Q-tables.")
    args = parser.parse_args()

    mode_str = "EVALUATION" if args.eval else "TRAINING + EVALUATION"
    print(f"Starting {mode_str} sweep with dt={config.DT}...")
    
    # Generate all combinations of swimmer_speed and alignment_timescale
    parameters = list(itertools.product(config.SWIMMER_SPEED, config.ALIGNMENT_TIMESCALE))
    
    for phi, psi in parameters:
        print(f"\n" + "="*50)
        print(f"PROCESS: phi={phi}, psi={psi}")
        print("="*50)
        
        # 1. Initialization
        env = TaylorGreenEnvironment(
            dt=config.DT,
            swimmer_speed=phi,
            alignment_timescale=psi,
            seed=config.SEED,
        )

        # 2. Training (Skip if in eval mode)
        q_table_filename = f"{config.SAVE_FOLDER}q_table_phi{phi}_psi{psi}_{config.N_EPISODES_TRAIN}.npy"
        
        if not args.eval:
            print(f"Training...")
            train(
                env=env,
                n_episodes=config.N_EPISODES_TRAIN,
                n_steps=config.N_STEPS,
                save=True,
                logging=False, 
                seed=config.SEED,
            )
        else:
            print(f"Skipping training, looking for {q_table_filename}...")
        
        # 3. Evaluation
        if os.path.exists(q_table_filename):
            print(f"\nEvaluating trained policy from {q_table_filename}...")
            policy = load_policy(q_table_filename)
            
            eval(
                policy=policy,
                swimmer_speed=phi,
                alignment_timescale=psi,
                n_episodes=config.N_EPISODES_EVAL,
                n_steps=config.N_STEPS,
                logging=True,
                make_plot=True, # This will save phi{phi}_psi{psi}.png
                show_arrows=False,
            )
        else:
            print(f"Warning: Q-table file {q_table_filename} not found, skipping evaluation.")
            
    print(f"\n{mode_str} completed successfully.")
