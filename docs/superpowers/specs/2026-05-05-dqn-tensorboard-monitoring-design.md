# Design Spec: DQN TensorBoard Monitoring for Learning Rate Optimization

## 1. Purpose
The goal of this design is to integrate **TensorBoard** monitoring into the DQN training pipeline of the `fluidvibe` project. This will provide real-time diagnostic metrics (Loss, Reward, Gradient Norm, and Average Q-values) to scientifically determine if the chosen learning rate is appropriate, too high (causing instability/explosion), or too low (causing slow convergence).

## 2. Architecture & Data Flow
The monitoring system follows a "Sensor-Logger" pattern:
- **Sensors (Agent)**: The `DQNAgent` calculates internal training metrics during each weight update.
- **Collector (Trainer)**: The `train_dqn.py` script manages the `SummaryWriter` and decides when to flush data to disk.
- **Visualizer (TensorBoard)**: A standalone process reads logs and renders curves in a web UI.

### Key Metrics
| Metric | Granularity | Purpose |
| :--- | :--- | :--- |
| `Loss/train` | Per Step | Monitor convergence and check for sudden spikes (too high LR). |
| `Reward/episode` | Per Episode | Track overall performance improvement. |
| `Diagnostic/Grad_Norm` | Per Step | Measure update magnitude. High values suggest LR reduction or clipping. |
| `Diagnostic/Avg_Q` | Per Step | Detect Q-value overestimation or divergence. |
| `Config/Epsilon` | Per Episode | Correlate exploration rate with stability. |

## 3. Component Details

### 3.1. Configuration (`config.py`)
Add parameters to centralize logging control:
- `DQN_LOG_DIR`: Base directory for TensorBoard logs (e.g., `./logs/dqn/`).
- `DQN_LOG_INTERVAL`: Step frequency for logging high-volume metrics (Loss, Grad Norm).

### 3.2. Agent Enhancement (`agent_dqn.py`)
Modify `agent.update()` to return a diagnostic dictionary:
- Calculate **Global Gradient Norm** using `torch.norm` across all model parameters before `optimizer.step()`.
- Extract **Mean Predicted Q-values** from the current batch.
- Return: `{"loss": float, "grad_norm": float, "avg_q": float}`.

### 3.3. Training Loop Integration (`train_dqn.py`)
- Initialize `torch.utils.tensorboard.SummaryWriter` with a run-specific path (e.g., `logs/dqn/phi0.3_psi1.0_YYYYMMDD-HHMMSS`).
- Maintain a `total_steps` counter for consistent X-axis mapping.
- Log Step-based metrics immediately after `agent.update()`.
- Log Episode-based metrics at the end of each `env.reset()` loop.

## 4. Testing & Validation
- **Connectivity Check**: Ensure TensorBoard can open the generated log directory.
- **Metric Integrity**: Verify `Grad_Norm` is non-zero and changes over time.
- **Experiment Comparison**: Validate that multiple training runs (e.g., different phis/psis) appear as separate selectable curves in the UI.

## 5. Success Criteria
- Real-time visualization of Loss and Reward.
- Ability to identify "Gradient Explosion" (Grad Norm > 100) or "Stagnation" (Grad Norm < 1e-6) visually.
- Capability to compare at least two different learning rate settings side-by-side in TensorBoard.
