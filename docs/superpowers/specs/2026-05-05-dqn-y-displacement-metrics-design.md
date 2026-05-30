# Design: DQN Evaluation Y-Displacement Metrics

## Goal
Modify `eval_dqn.py` to calculate and report the mean net y-direction displacement for both the trained DQN agent and the naive agent across all evaluation episodes.

## Metrics
- **Net Y-Displacement (DQN)**: $y_{final} - y_{initial}$ for the DQN agent.
- **Net Y-Displacement (Naive)**: $y_{final} - y_{initial}$ for the naive agent.
- **Mean Net Y-Displacement**: The average of the net displacement across all evaluation episodes.

## Implementation Details
1. **Accumulators**: Initialize `total_y_disp = 0` and `total_y_disp_naive = 0` at the start of `eval_dqn`.
2. **Episode Tracking**: 
   - Store `y_init = pos_init[1]` at the start of each episode.
   - After $N$ steps, compute `ep_y_disp = env.swimmer_position[1] - y_init`.
   - Update accumulators.
3. **Reporting**:
   - Calculate `mean_y_disp = total_y_disp / n_episodes`.
   - Calculate `mean_y_disp_naive = total_y_disp_naive / n_episodes`.
   - Print these values in the summary block of `eval_dqn`.

## Success Criteria
- The script correctly prints the mean net y-displacement for both agents.
- The values are physically plausible given the environment (Taylor-Green flow).
