from environments.taylor_green import TaylorGreenEnvironment
from train import train
import config


if __name__ == "__main__":
    print(f"Using closed-form analytical solution with dt={config.DT}...")
    
    env = TaylorGreenEnvironment(
        dt=config.DT,
        swimmer_speed=config.SWIMMER_SPEED,
        alignment_timescale=config.ALIGNMENT_TIMESCALE,
        seed=config.SEED,
    )  # initialise environment

    train(
        env=env,
        n_episodes=config.N_EPISODES_TRAIN,
        n_steps=config.N_STEPS,
        save=True,
        seed=config.SEED,
    )
