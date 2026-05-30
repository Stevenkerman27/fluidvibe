# DQN Y-Displacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `eval_dqn.py` to calculate and report the mean net y-direction displacement for both the trained DQN agent and the naive agent.

**Architecture:** Initialize displacement accumulators, capture initial y-positions at the start of each episode, calculate net displacement at the end of each episode, and print the averaged results in the summary section.

**Tech Stack:** Python, NumPy

---

### Task 1: Add Y-Displacement Tracking to `eval_dqn`

**Files:**
- Modify: `eval_dqn.py`

- [ ] **Step 1: Initialize accumulators**

In `eval_dqn`, add `total_y_disp` and `total_y_disp_naive` next to the return accumulators.

```python
    # 3. 评估循环
    rng = np.random.default_rng(seed=config.SEED)
    total_return = 0
    total_return_naive = 0
    total_y_disp = 0          # New
    total_y_disp_naive = 0    # New
```

- [ ] **Step 2: Capture initial Y-position and calculate final displacement**

Update the episode loop to capture the starting Y and update the totals after the step loop.

```python
    for episode in range(n_episodes):
        # 统一初始状态
        pos_init = np.array([rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)])
        y_init = pos_init[1] # Capture initial Y
        ori_init = rng.uniform(0, 2*np.pi)
        
        # ... (rest of reset logic)

        for i in range(n_steps):
            # ... (step logic)
            
        total_return += ep_ret
        total_return_naive += ep_ret_naive
        
        # Calculate net displacement
        total_y_disp += (env.swimmer_position[1] - y_init)
        total_y_disp_naive += (env_naive.swimmer_position[1] - y_init)
```

- [ ] **Step 3: Print averaged Y-displacement metrics**

Update the reporting section to include the mean net y-displacement.

```python
    # 4. 统计结果
    mean_ret = total_return / n_episodes
    mean_ret_naive = total_return_naive / n_episodes
    mean_y_disp = total_y_disp / n_episodes             # New
    mean_y_disp_naive = total_y_disp_naive / n_episodes # New

    print(f"\n[phi={phi}, psi={psi}]")
    print(f"Mean DQN Return: {mean_ret:.2f}")
    print(f"Mean Naive Return: {mean_ret_naive:.2f}")
    print(f"Mean DQN Net Y-Displacement: {mean_y_disp:.2f}")       # New
    print(f"Mean Naive Net Y-Displacement: {mean_y_disp_naive:.2f}") # New
```

- [ ] **Step 4: Run evaluation to verify output**

Run: `python eval_dqn.py`
Expected: The output should now include "Mean DQN Net Y-Displacement" and "Mean Naive Net Y-Displacement" for each configuration.

- [ ] **Step 5: Commit changes**

```bash
git add eval_dqn.py docs/superpowers/specs/2026-05-05-dqn-y-displacement-metrics-design.md
git commit -m "feat: report net y-displacement in dqn evaluation"
```
