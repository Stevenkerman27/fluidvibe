# Design Spec: Sin/Cos Orientation Observations for DQN

**Date:** 2026-06-01
**Status:** Approved
**Topic:** Transition from radian-based orientation to sine/cosine components in the TaylorGreen environment to resolve angular discontinuity issues.

## 1. Purpose
Currently, the `TaylorGreenContinuousEnvironment` provides orientation as a single radian value. This results in a jump (discontinuity) at the boundary (e.g., from $3\pi/2$ back to $-\pi/2$), which makes it difficult for a neural network to learn a smooth policy. By representing the orientation $\theta$ as $[\sin \theta, \cos \theta]$, we ensure the observation space is continuous throughout the full $2\pi$ rotation.

## 2. Approach
We will replace the single `orientation` scalar in the observation vector with two scalars: `sin_orientation` and `cos_orientation`.

### 2.1 Observation Space Mapping
*   **Old:** `[vorticity_scaled, orientation]` (Shape: 2)
    *   `vorticity_scaled`: $(-\infty, \infty)$
    *   `orientation`: $[-\pi/2, 3\pi/2)$
*   **New:** `[vorticity_scaled, sin_orientation, cos_orientation]` (Shape: 3)
    *   `vorticity_scaled`: $(-\infty, \infty)$
    *   `sin_orientation`: $[-1.0, 1.0]$
    *   `cos_orientation`: $[-1.0, 1.0]$

## 3. Implementation Details

### 3.1 Environment (`environments/taylor_green_continuous.py`)
*   Update `_get_observation()` to return the 3-element array.
*   Update `TaylorGreenGymWrapper` to define the new `observation_space` Box with shape `(3,)`.

### 3.2 Training (`train_dqn_jax.py`)
*   The training script should automatically handle the dimensionality change because it initializes the model and replay buffer based on `envs.single_observation_space`.

### 3.3 Visualization (`visualize_dqn_policy.py`)
*   Update `obs_dummy` to shape `(1, 3)`.
*   In the policy plotting loop, transform the grid's `orientation` (radians) into `sin` and `cos` components before passing them to the model.

## 4. Verification Plan

### 4.1 Unit Testing
*   Modify `tests/test_environment.py` (if it exists) or create a new check to verify that `env.reset()` and `env.step()` return an observation of length 3.
*   Verify that `sin^2 + cos^2` is approximately 1.0 for the orientation components.

### 4.2 Functional Verification
*   Run `visualize_dqn_policy.py` with a dummy model or a newly trained checkpoint to ensure it doesn't crash and correctly maps the 3D input space.
*   Start a short training run of `train_dqn_jax.py` to confirm the loss decreases and no shape mismatches occur.
