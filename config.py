import numpy as np
import os

# --- Environment Parameters (Physics) ---
total_time = 100
basic_dt = 0.1
speed_up = 1
DT = basic_dt*speed_up
SWIMMER_SPEED = [0.3]      # phi
ALIGNMENT_TIMESCALE = [1.0] # psi
FLOW_SPEED = 1.0         # u0
DIFFUSIVITY_ROTATIONAL = 0.0001
DIFFUSIVITY_TRANSLATIONAL = 0.001
LATERAL_PENALTY_WEIGHT = 0.01
LATERAL_X_MIN = -0.5 * np.pi
LATERAL_X_MAX = 2.5 * np.pi
MIN_FLOW_SPEED_THRESHOLD = 1e-8
VORTICITY_THRESHOLD = 1.0 / 3.0
SEED = 42

# --- General Training Parameters (Q-Learning) ---
N_EPISODES_TRAIN = 2000
N_STEPS = int(total_time/DT)           # Steps per episode
GAMMA = 0.999            # Discount factor
EPSILON_START = 0.9     # Initial exploration rate
LEARNING_RATE = 0.05     # Q-learning rate
INITIAL_Q_VALUE = 20.0  # Optimistic initialization

# --- Evaluation Parameters ---
N_EPISODES_EVAL = 80

# --- DQN Specific Parameters ---
DQN_N_EPISODES_TRAIN = 2000
DQN_LEARNING_RATE = 1e-4
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.4
DQN_EPSILON_DECAY_DURATION = int(DQN_N_EPISODES_TRAIN)*0.9  # Number of episodes to decay from START to END
DQN_DEVICE = "cpu"      # "cpu", "cuda", or "auto"
DQN_BATCH_SIZE = 64
DQN_HIDDEN_DIM = 64
DQN_BUFFER_CAPACITY = 20000
DQN_TRAIN_FREQ = 4       # Number of environment steps between each network update
DQN_TARGET_UPDATE_FREQ = 300
DQN_LEARNING_STARTS = 1000  # Timestep to start learning
DQN_TAU = 0.5             # Target network update rate
DQN_SAVE_MODEL = True


# --- Paths & Logging ---
SAVE_FOLDER = "./q_table/"
DQN_LOG_DIR = "./logs/dqn/"
DQN_LOG_INTERVAL = 10  # Log every N update steps

# Create folders if they don't exist
for folder in [SAVE_FOLDER, DQN_LOG_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

