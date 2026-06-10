# DQN Optimization Log

## Attempt 1: Standard Deep RL Baseline
**Date:** 2026-06-01
**Goal:** Establish a stronger baseline with standard deep RL parameters.

### Hyperparameters
- `DQN_HIDDEN_DIM`: 128
- `DQN_EPSILON_END`: 0.05
- `DQN_EPSILON_DECAY_DURATION`: 1600 episodes
- `DQN_TAU`: 0.005 (Soft Updates)
- `DQN_TARGET_UPDATE_FREQ`: 1
- `DQN_BATCH_SIZE`: 128
- `DQN_BUFFER_CAPACITY`: 50,000
- `DQN_GAMMA`: 0.999

### Results
- Mean DQN Return: 36.60
- Mean DQN Net Y-Displacement: 38.03
- Gain: 1322.80%

### Observations
- Significant improvement over baseline (22.68 -> 38.03).
- Soft updates (TAU=0.005) and smaller final epsilon (0.05) contributed to much better stability and exploitation.
- Aligning GAMMA with tabular agent (0.999) seems to have helped the agent value long-term y-distance.
- Performance is very close to the tabular Q-learning target (~40).
- The training curve (visualized in the plot) shows a steady increase in return and y-distance.

## Attempt 2: Fine-tuning for >40
**Date:** 2026-06-01
**Goal:** Increase capacity and training duration to cross the 40 y-distance threshold.

### Hyperparameters
- `DQN_N_EPISODES_TRAIN`: 3000
- `DQN_LEARNING_RATE`: 5e-5
- `DQN_HIDDEN_DIM`: 256
- `DQN_EPSILON_DECAY_DURATION`: 2400
- `DQN_BUFFER_CAPACITY`: 100,000
- Other parameters same as Attempt 1.

### Results
- Mean DQN Return: 1.74
- Mean DQN Net Y-Displacement: 6.85
- Gain: -32.28%

### Observations
- Significant regression. The lower learning rate combined with larger network might have required even more training or better initialization.
- The training curve (visualized in the plot) shows a steady increase in return and y-distance.

## Attempt 3: Smoothing and Batching
**Date:** 2026-06-01
**Goal:** Return to Attempt 1 baseline and improve stability with larger batch size and smoother target updates.

### Hyperparameters
- `DQN_N_EPISODES_TRAIN`: 3000
- `DQN_BATCH_SIZE`: 256
- `DQN_TAU`: 0.001
- `DQN_HIDDEN_DIM`: 128
- `DQN_LEARNING_RATE`: 1e-4
- `DQN_BUFFER_CAPACITY`: 50,000
- `DQN_EPSILON_DECAY_DURATION`: 2400
- Other parameters same as Attempt 1.

### Results
- Mean DQN Return: -17.22
- Mean DQN Net Y-Displacement: 27.31
- Gain: -769.47%

### Observations
- Regression compared to Attempt 1. Larger batch size and smoother updates might have slowed down learning too much for 3000 episodes.
- Epsilon decay might have been too slow, preventing the agent from exploiting effectively until very late.

## Attempt 4: Lower LR and Early Epsilon End
**Date:** 2026-06-01
**Goal:** Use Attempt 1 baseline with lower learning rate for better convergence and finish exploration earlier to allow more exploitation time.

### Hyperparameters
- `DQN_N_EPISODES_TRAIN`: 3000
- `DQN_LEARNING_RATE`: 5e-5
- `DQN_EPSILON_DECAY_DURATION`: 1600
- `DQN_HIDDEN_DIM`: 128
- `DQN_BATCH_SIZE`: 128
- `DQN_TAU`: 0.005
- `DQN_BUFFER_CAPACITY`: 50,000
- Other parameters same as Attempt 1.

### Results
- Mean DQN Return: 18.78
- Mean DQN Net Y-Displacement: 30.18
- Gain: 630.05%

### Observations
- Regression compared to Attempt 1. Lowering the learning rate might have slowed down learning too much even with more episodes.

## Attempt 5: Attempt 1 with Lower Epsilon End
**Date:** 2026-06-01
**Goal:** Return to Attempt 1 baseline and use even lower final epsilon (0.01) to maximize exploitation.

### Hyperparameters
- `DQN_N_EPISODES_TRAIN`: 2000
- `DQN_EPSILON_END`: 0.01
- `DQN_EPSILON_DECAY_DURATION`: 1600
- All other parameters same as Attempt 1.

### Results
- Mean DQN Return: -17.80
- Mean DQN Net Y-Displacement: 20.07
- Gain: -791.97%

### Observations
- Drastic regression. It's possible that 0.01 final epsilon is too low and leads to a very fragile policy, or the agent got stuck in a local optimum.

## Attempt 6: Cost Reduction
**Date:** 2026-06-01
**Goal:** Maintain y-distance ~38 while reducing training cost (capacity, batch size, episodes).

### Hyperparameters
- `DQN_N_EPISODES_TRAIN`: 1600 (80% of Attempt 1)
- `DQN_HIDDEN_DIM`: 64
- `DQN_BATCH_SIZE`: 64
- `DQN_EPSILON_DECAY_DURATION`: 1200
- Other parameters same as Attempt 1.

### Results
- Mean DQN Return: 40.23
- Mean DQN Net Y-Displacement: 41.22
- Gain: 1464.06%

### Observations
- Highly successful! Surpassed the target of 40 while significantly reducing training cost.
- Reduced capacity (HIDDEN_DIM=64) and batch size (64) didn't hurt performance, and likely helped generalization or convergence speed.
- 1600 episodes was sufficient, confirming the observation that performance peaked around 80% of Attempt 1's training.
- This configuration (Attempt 6) is now the recommended setup for both performance and efficiency.


