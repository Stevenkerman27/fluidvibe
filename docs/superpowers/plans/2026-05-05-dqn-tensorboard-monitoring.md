# DQN TensorBoard Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate TensorBoard monitoring into the DQN training pipeline to provide real-time diagnostic metrics for learning rate optimization.

**Architecture:** A "Sensor-Logger" approach where the `DQNAgent` calculates internal metrics (Loss, Gradient Norm, Mean Q) and `train_dqn.py` uses `SummaryWriter` to log these along with environment rewards.

**Tech Stack:** PyTorch, TensorBoard (torch.utils.tensorboard).

---

### Task 1: Update Configuration

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add logging configurations to `config.py`**

```python
# --- Logging & Monitoring ---
DQN_LOG_DIR = "./logs/dqn/"
DQN_LOG_INTERVAL = 10  # Log every N update steps
```

- [ ] **Step 2: Verify configuration**
Confirm the values are added correctly.

---

### Task 2: Enhance DQNAgent for Diagnostics

**Files:**
- Modify: `agent_dqn.py`

- [ ] **Step 1: Update `DQNAgent.update` to calculate Grad Norm and Avg Q**

Modify `agent_dqn.py` to return a dictionary of metrics:
```python
    def update(self) -> Optional[dict]:
        """从 Buffer 中采样并更新网络参数，返回诊断指标"""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前估计的 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        avg_q = current_q.mean().item()

        # 计算 Target Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算 Loss
        loss = nn.MSELoss()(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数
        grad_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        self.optimizer.step()

        # 定期同步目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "loss": loss.item(),
            "avg_q": avg_q,
            "grad_norm": grad_norm
        }
```

---

### Task 3: Integrate TensorBoard in Training Script

**Files:**
- Modify: `train_dqn.py`

- [ ] **Step 1: Initialize `SummaryWriter` and update logging logic**

Modify `train_dqn.py`:
```python
from torch.utils.tensorboard import SummaryWriter
import datetime

# inside train_dqn(phi, psi) before the loop:
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"{config.DQN_LOG_DIR}phi{phi}_psi{psi}_{timestamp}"
writer = SummaryWriter(log_dir=log_dir)
total_steps = 0

# inside step loop after agent.update():
info = agent.update()
if info and total_steps % config.DQN_LOG_INTERVAL == 0:
    writer.add_scalar("Loss/train", info['loss'], total_steps)
    writer.add_scalar("Diagnostic/Grad_Norm", info['grad_norm'], total_steps)
    writer.add_scalar("Diagnostic/Avg_Q", info['avg_q'], total_steps)
total_steps += 1

# at the end of episode loop:
writer.add_scalar("Reward/episode", cumulative_reward, episode)
writer.add_scalar("Config/Epsilon", epsilon, episode)

# at the end of train_dqn function:
writer.close()
```

---

### Task 4: Verification

- [ ] **Step 1: Run a short training run**
Execute `python train_dqn.py`.

- [ ] **Step 2: Start TensorBoard**
Run `tensorboard --logdir ./logs/dqn/` and verify metrics are being logged.
