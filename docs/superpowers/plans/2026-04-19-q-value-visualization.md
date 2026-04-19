# Q-Value Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time and post-training visualization for Mean Max Q-values in `train.py` to monitor learning progress.

**Architecture:** Use `plt.subplots(2, 1, sharex=True)` to create two vertically stacked subplots. Track `np.mean(np.max(agent.q, axis=1))` in each episode and update the plots every 50 episodes.

**Tech Stack:** Python, NumPy, Matplotlib, tqdm.

---

### Task 1: Update `train.py` for Dual Plotting

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Initialize dual plotting and data tracking**

```python
# In train() function, replace the plotting initialization block:
    # Initialize agent with optimistic initialization
    q_initial = config.INITIAL_Q_VALUE * np.ones((12, 4))
    agent = QLearningAgent(q=q_initial, gamma=config.GAMMA, seed=seed)
    episode_returns = []
    mean_max_q_values = [] # Track mean max Q-values

    # Initialize plotting
    if logging:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        (line_return,) = ax1.plot([], [], label="Episode Return")
        (line_q,) = ax2.plot([], [], label="Mean Max Q-Value", color='orange')
        
        ax1.set_ylabel("Return")
        ax1.set_title("Training Progress")
        ax1.legend()
        
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Mean Max Q")
        ax2.legend()
```

- [ ] **Step 2: Update loop to calculate and store Q-values**

```python
# Inside the episode loop, after updating Q-table:
        # agent.update_q(...)
        
        # After the step loop:
        episode_returns.append(episode_return)
        current_mean_max_q = np.mean(np.max(agent.q, axis=1))
        mean_max_q_values.append(current_mean_max_q)
```

- [ ] **Step 3: Update live plotting logic**

```python
# Inside the logging block in the episode loop:
        if logging:
            # Update plot every 50 episodes
            if episode % 50 == 0 or episode == n_episodes - 1:
                line_return.set_xdata(range(len(episode_returns)))
                line_return.set_ydata(episode_returns)
                line_q.set_xdata(range(len(mean_max_q_values)))
                line_q.set_ydata(mean_max_q_values)
                
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                fig.canvas.draw()
                fig.canvas.flush_events()
```

- [ ] **Step 4: Update final display and save logic**

```python
# At the end of train() function:
    if logging:
        plt.ioff()
        if save:
            filename = config.RETURNS_PLOT_PATH
            plt.savefig(filename)
            print(f"Training progress plot saved to {filename}")
        plt.show()
```

- [ ] **Step 5: Run a short training session to verify**

Run: `python train.py` (assuming it has a `if __name__ == "__main__":` block or similar, or I can use a test script).
Actually, I should verify if `train.py` can be run directly.

- [ ] **Step 6: Commit changes**

```bash
git add train.py
git commit -m "feat: add mean max q-value visualization to training"
```
