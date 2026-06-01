# Sin/Cos Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single radian orientation observation with sine and cosine components to resolve angular discontinuities in DQN training.

**Architecture:** Update the environment to output a 3D observation vector `[vorticity, sin_theta, cos_theta]` and adjust visualization/evaluation scripts to handle this new input dimension.

**Tech Stack:** Python, NumPy, Gymnasium, JAX/Flax (CleanRL)

---

### Task 1: Environment Modification

**Files:**
- Modify: `environments/taylor_green_continuous.py`
- Create: `tests/test_environment_sincos.py`

- [ ] **Step 1: Create a test to verify the current (failing) state**
Create `tests/test_environment_sincos.py`:
```python
import numpy as np
import gymnasium as gym
import environments.taylor_green_continuous

def test_observation_shape():
    env = gym.make("TaylorGreen-v0")
    obs, _ = env.reset()
    # This should fail after we change the spec/expectation
    assert obs.shape == (3,), f"Expected shape (3,), got {obs.shape}"
    assert -1.0 <= obs[1] <= 1.0
    assert -1.0 <= obs[2] <= 1.0
    assert np.isclose(obs[1]**2 + obs[2]**2, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_environment_sincos.py`
Expected: FAIL (AssertionError: Expected shape (3,), got (2,))

- [ ] **Step 3: Update `_get_observation` in `environments/taylor_green_continuous.py`**
Replace the orientation logic:
```python
    def _get_observation(self):
        if abs(self.u0) > _MIN_FLOW_SPEED_THRESHOLD:
            vorticity_scaled = self.flow_vorticity / self.u0
        else:
            vorticity_scaled = 0

        orientation_rad = np.arctan2(self.swimming_velocity[1], self.swimming_velocity[0])
        return np.array([vorticity_scaled, np.sin(orientation_rad), np.cos(orientation_rad)])
```

- [ ] **Step 4: Update `TaylorGreenGymWrapper` in `environments/taylor_green_continuous.py`**
Update `observation_space`:
```python
        # State: [vorticity_scaled, sin_theta, cos_theta]
        low = np.array([-np.inf, -1.0, -1.0], dtype=np.float32)
        high = np.array([np.inf, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
```

- [ ] **Step 5: Run test to verify it passes**
Run: `pytest tests/test_environment_sincos.py`
Expected: PASS

---

### Task 2: Update Visualization Script

**Files:**
- Modify: `visualize_dqn_policy.py`

- [ ] **Step 1: Update `obs_dummy` and grid processing**
In `plot_dqn_policy`:
Update `obs_dummy`: `obs_dummy = jnp.zeros((1, 3))`
Update `states` creation:
```python
    # Vectorized computation for the entire grid
    vort_flat = V.flatten()
    ori_flat = O.flatten()
    # Transform radians to sin/cos
    sin_ori = jnp.sin(ori_flat)
    cos_ori = jnp.cos(ori_flat)
    states = jnp.stack([vort_flat, sin_ori, cos_ori], axis=-1)
```

- [ ] **Step 2: Verify with a dummy model run**
Run: `python visualize_dqn_policy.py`
Expected: Script should run (it might fail to load existing 2D models, which is expected. We are verifying the logic for 3D).

---

### Task 3: Update Evaluation Script

**Files:**
- Modify: `eval_dqn.py`

- [ ] **Step 1: Update `JaxQNetwork` and `obs_dummy`**
In `eval_dqn.py`:
Update `JaxQNetwork` default `state_dim` (implied in init) and `obs_dummy`:
```python
        jax_q_net = JaxQNetwork(action_dim=4, hidden_dim=config.DQN_HIDDEN_DIM)
        obs_dummy = jnp.zeros((1, 3))
```

- [ ] **Step 2: Update `get_jax_action` observation transformation**
In the evaluation loop:
```python
            # DQN 动作选择 (epsilon=0)
            if is_jax:
                # state is [vorticity, sin, cos] now, so just pass it
                action = int(get_jax_action(state[np.newaxis, :])[0])
```
(Wait, if `state` is already returned as 3D by the environment, no change is needed in the loop, just the model init.)

- [ ] **Step 3: Verify script runs**
Run: `python eval_dqn.py` (Expect model loading failure for old models, but check for shape errors).

---

### Task 4: (Optional) Cleanup and Final Check

- [ ] **Step 1: Remove temporary test**
Run: `rm tests/test_environment_sincos.py`
- [ ] **Step 2: Final training test**
Run a short training run: `python train_dqn_jax.py --total-timesteps 1000`
Expected: Starts training without shape errors.
