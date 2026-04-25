# Design Spec: Generalization Test for RL Agent

## Goal
Create a generalization test script that evaluates a single trained policy (Q-table) across a range of fluid environments with different physical parameters ($\phi$ and $\psi$), as defined in `config.py`.

## Proposed Changes

### 1. Create `generalization_test.py`
This new script will:
- Load a specific Q-table (default path defined in code, also overridable via CLI).
- Iterate through all combinations of `SWIMMER_SPEED` ($\phi$) and `ALIGNMENT_TIMESCALE` ($\psi$) from `config.py`.
- For each combination, instantiate a `TaylorGreenEnvironment` and evaluate the policy using the existing `eval` function.
- Save individual plots for each environment.

### 2. Update `eval.py` (Optional but Recommended)
To prevent filename collisions and clarify which policy was used for each plot, I'll update `plot_policy` to accept an optional `filename_prefix`.

## Architecture & Data Flow
1. **Input:** A `.npy` file containing a Q-table.
2. **Process:**
   - Load Q-table $\rightarrow$ Argmax $\rightarrow$ Policy array.
   - Loop over `(phi, psi)` in `config.SWIMMER_SPEED` $\times$ `config.ALIGNMENT_TIMESCALE`.
   - Call `eval(policy, swimmer_speed=phi, alignment_timescale=psi, ...)`.
3. **Output:** A set of `.png` files showing trajectories in different environments.

## Testing Strategy
- Verify the script runs with a valid Q-table path.
- Verify it handles missing Q-table files gracefully.
- Verify plots are generated with the expected naming convention.
