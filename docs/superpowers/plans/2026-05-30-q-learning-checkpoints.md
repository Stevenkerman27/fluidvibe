# Q-Learning Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone training mode to `agent_qlearning.py` with checkpoint saving functionality.

**Architecture:** Add a `if __name__ == "__main__":` block to `agent_qlearning.py` that implements a training loop with periodic Q-table saves to `q_table/checkpoints/`. It will override specific config parameters (`SWIMMER_SPEED`, `ALIGNMENT_TIMESCALE`) and use the rest from `config.py`.

**Tech Stack:** Python, NumPy, tqdm, TaylorGreenEnvironment.

---

### Task 1: Update agent_qlearning.py with Training Logic

**Files:**
- Modify: `agent_qlearning.py`

- [ ] **Step 1: Add necessary imports**
Add `import config`, `from environments.taylor_green import TaylorGreenEnvironment`, `from tqdm import tqdm`, and `import os`.

- [ ] **Step 2: Add __main__ block with training parameters**
Define local `SWIMMER_SPEED = 0.3`, `ALIGNMENT_TIMESCALE = 1.0`, `N_EPISODES = 6000`, and `N_CHECKPOINTS = 5`.

- [ ] **Step 3: Implement training loop with checkpoint logic**
Initialize the environment and agent. Run the training loop for `N_EPISODES`. Calculate checkpoint intervals: `save_every = N_EPISODES // (N_CHECKPOINTS - 1)` if `N_CHECKPOINTS > 1`. Save at `episode % save_every == 0` and the final episode.

- [ ] **Step 4: Save checkpoints to q_table/checkpoints/**
Use `np.save` to save the Q-table with a filename like `q_table/checkpoints/q_table_phi{phi}_psi{psi}_ep{episode}.npy`.

- [ ] **Step 5: Verify implementation**
Run `python agent_qlearning.py` for a small number of episodes (e.g., 10 episodes, 2 checkpoints) to ensure files are saved correctly.

```python
# Example of the main block structure to be added:
if __name__ == "__main__":
    # Local overrides
    SWIMMER_SPEED = 0.3
    ALIGNMENT_TIMESCALE = 1.0
    N_EPISODES = 6000
    N_CHECKPOINTS = 5
    
    # Calculate intervals (0, 1500, 3000, 4500, 6000)
    save_at_episodes = np.linspace(0, N_EPISODES, N_CHECKPOINTS, dtype=int)
    
    # ... Environment & Agent Init ...
    # ... Training Loop ...
    # ... Checkpoint Saving ...
```

### Task 2: Validation

- [ ] **Step 1: Create a test script or run a short trial**
Run `python agent_qlearning.py` with `N_EPISODES = 4` and `N_CHECKPOINTS = 3` to verify saves at 0, 2, 4.

- [ ] **Step 2: Check for existence of files**
Verify `q_table/checkpoints/q_table_phi0.3_psi1.0_ep0.npy`, etc. exist.
