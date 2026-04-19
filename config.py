import numpy as np
import os

# --- Environment Parameters (Physics) ---
total_time = 100
basic_dt = 0.01
speed_up = 1
DT = basic_dt*speed_up
SWIMMER_SPEED = 0.3      # phi
ALIGNMENT_TIMESCALE = 1.0 # psi
FLOW_SPEED = 1.0         # u0
SEED = 42

# --- Training Parameters (RL) ---
N_EPISODES_TRAIN = 600
N_STEPS = int(total_time/DT)           # Steps per episode
GAMMA = 0.99            # Discount factor
EPSILON_START = 0.2     # Initial exploration rate
LEARNING_RATE = 0.01     # Q-learning rate
INITIAL_Q_VALUE = 1000.0  # Optimistic initialization

# --- Evaluation Parameters ---
N_EPISODES_EVAL = 10

# --- Paths & Logging ---
SAVE_FOLDER = "./q_table/"
# Create save folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

Q_TABLE_PATH = f"{SAVE_FOLDER}q_table_{N_EPISODES_TRAIN}.npy"
RETURNS_PLOT_PATH = f"{SAVE_FOLDER}episode_returns.png"
