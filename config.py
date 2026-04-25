import numpy as np
import os

# --- Environment Parameters (Physics) ---
total_time = 100
basic_dt = 0.1
speed_up = 1
DT = basic_dt*speed_up
SWIMMER_SPEED = [0.2, 0.3, 0.4]      # phi
ALIGNMENT_TIMESCALE = [0.8, 1.0, 1.2] # psi
FLOW_SPEED = 1.0         # u0
DIFFUSIVITY_ROTATIONAL = 0.0001
DIFFUSIVITY_TRANSLATIONAL = 0.001
LATERAL_PENALTY_WEIGHT = 0.02
LATERAL_PENALTY_THRESHOLD = np.pi  # Absolute distance threshold before penalty
MIN_FLOW_SPEED_THRESHOLD = 1e-8
VORTICITY_THRESHOLD = 1.0 / 3.0
SEED = 42

# --- Training Parameters (RL) ---
N_EPISODES_TRAIN = 3000
N_STEPS = int(total_time/DT)           # Steps per episode
GAMMA = 0.999            # Discount factor
EPSILON_START = 0.1     # Initial exploration rate
LEARNING_RATE = 0.1     # Q-learning rate
INITIAL_Q_VALUE = 10.0  # Optimistic initialization

# --- Evaluation Parameters ---
N_EPISODES_EVAL = 50

# --- DQN Parameters ---
DQN_DEVICE = "cpu"      # "cpu", "cuda", or "auto"
DQN_BATCH_SIZE = 32
DQN_EPSILON_DECAY = 0.99
DQN_MIN_EPSILON = 0.01
DQN_HIDDEN_DIM = 32
DQN_BUFFER_CAPACITY = 10000
DQN_TARGET_UPDATE_FREQ = 100

# --- Paths & Logging ---
SAVE_FOLDER = "./q_table/"
# Create save folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

