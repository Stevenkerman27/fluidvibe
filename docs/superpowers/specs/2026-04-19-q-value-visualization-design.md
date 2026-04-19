# Q-Value Visualization Design

## 1. Objective
Add real-time and post-training visualization for Q-values in `train.py` to monitor the learning progress and evaluate the suitability of the learning rate.

## 2. Metrics to Track
- **Episode Return**: The total reward accumulated in each episode (existing).
- **Mean Max Q-Value**: The average of the maximum Q-values across all states at the end of each episode.
  - Formula: $\frac{1}{N} \sum_{s \in S} \max_{a} Q(s, a)$
  - Significance: Indicates the agent's overall value estimation and convergence.

## 3. Visualization Implementation
- **Layout**: Two vertically stacked subplots using `plt.subplots(2, 1, sharex=True)`.
  - **Top Plot**: Episode Return vs. Episode.
  - **Bottom Plot**: Mean Max Q-Value vs. Episode.
- **Interactivity**: 
  - `plt.ion()` for real-time updates.
  - Refresh rate: Every 50 episodes (as currently implemented for returns).
- **Storage**: Maintain a list `mean_max_q_values` to store historical data during training.

## 4. Output & Saving
- The combined plot (Return and Q-Value) will be saved to the path defined in `config.RETURNS_PLOT_PATH` when `save=True`.
- Consolidation: Instead of saving only the returns, the saved image will now provide a comprehensive view of both performance (Return) and learning internal state (Q-Value).

## 5. Implementation Steps
1.  Modify `train.py`'s `train` function to initialize a second list for Q-values.
2.  Update the plotting initialization to create two subplots.
3.  Update the loop to calculate and store the mean max Q-value.
4.  Update the plotting update logic to refresh both subplots.
5.  Ensure the save logic captures the entire figure.
