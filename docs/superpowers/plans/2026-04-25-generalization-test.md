# Generalization Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script to evaluate a single Q-table against multiple fluid parameter environments and save the results as individual plots.

**Architecture:** 
1. Enhance `eval.py` to support custom filename prefixes for plots.
2. Create `generalization_test.py` which loads a Q-table and iterates through parameter combinations from `config.py`, calling `eval` for each.

**Tech Stack:** Python, NumPy, Matplotlib.

---

### Task 1: Update `eval.py` for flexible plotting

**Files:**
- Modify: `eval.py`

- [ ] **Step 1: Modify `plot_policy` to accept `filename_prefix`**

```python
def plot_policy(
    n_episodes: int,
    positions: np.ndarray,
    positions_naive: np.ndarray,
    actions_taken: np.ndarray,
    plot_params: Dict[str, float],
    show_arrows: bool = True,
    filename_prefix: str = ""
):
    # ... existing code ...
    plt.title(rf"$\phi={plot_params['phi']}, \psi={plot_params['psi']}$")
    plt.tight_layout()
    save_name = f"phi{plot_params['phi']}_psi{plot_params['psi']}_{plot_params['mean_y']:.2f}({plot_params['mean_y_naive']:.2f}).png"
    if filename_prefix:
        save_name = f"{filename_prefix}_{save_name}"
    plt.savefig(save_name, dpi=300)
    plt.close()
```

- [ ] **Step 2: Update `eval` function signature and call to `plot_policy`**

```python
def eval(
    policy: list,
    swimmer_speed: float = config.SWIMMER_SPEED,
    alignment_timescale: float = config.ALIGNMENT_TIMESCALE,
    n_episodes: int = config.N_EPISODES_EVAL,
    n_steps: int = config.N_STEPS,
    logging: bool = True,
    make_plot: bool = False,
    show_arrows: bool = True,
    filename_prefix: str = ""
) -> None:
    # ... existing code ...
    if make_plot:
        plot_params = {
            "phi": env.swimmer_speed,
            "psi": env.alignment_timescale,
            "mean_y": mean_y_dist,
            "mean_y_naive": mean_y_dist_naive,
        }
        plot_policy(
            n_episodes,
            positions,
            positions_naive,
            actions_taken,
            plot_params,
            show_arrows=show_arrows,
            filename_prefix=filename_prefix
        )
```

- [ ] **Step 3: Commit**

```bash
git add eval.py
git commit -m "refactor: allow filename prefix in eval plotting"
```

### Task 2: Create `generalization_test.py`

**Files:**
- Create: `generalization_test.py`

- [ ] **Step 1: Write `generalization_test.py`**

```python
import itertools
import os
import argparse
import config
from eval import eval, load_policy

def main():
    parser = argparse.ArgumentParser(description="Generalization test: Evaluate a single Q-table on multiple environments.")
    # Default path for convenience as requested
    default_q_table = f"{config.SAVE_FOLDER}q_table_phi0.3_psi1.0_3000.npy"
    parser.add_argument("--q_table", type=str, default=default_q_table, help="Path to the Q-table (.npy) to evaluate.")
    args = parser.parse_args()

    q_table_path = args.q_table
    if not os.path.exists(q_table_path):
        print(f"Error: Q-table file {q_table_path} not found.")
        return

    print(f"Loading policy from {q_table_path}...")
    policy = load_policy(q_table_path)
    
    # Extract params from filename for prefixing (optional but nice)
    # e.g., q_table_phi0.3_psi1.0_3000.npy -> gen_phi0.3_psi1.0
    base_name = os.path.basename(q_table_path).replace("q_table_", "").replace(".npy", "")
    prefix = f"gen_from_{base_name}"

    print(f"Starting generalization sweep using policy {base_name}...")
    
    parameters = list(itertools.product(config.SWIMMER_SPEED, config.ALIGNMENT_TIMESCALE))
    
    for phi, psi in parameters:
        print(f"\nEvaluating on phi={phi}, psi={psi}...")
        eval(
            policy=policy,
            swimmer_speed=phi,
            alignment_timescale=psi,
            n_episodes=config.N_EPISODES_EVAL,
            n_steps=config.N_STEPS,
            logging=True,
            make_plot=True,
            show_arrows=False,
            filename_prefix=prefix
        )

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs**

Run: `python generalization_test.py --q_table q_table/q_table_phi0.3_psi1.0_3000.npy`
Expected: Processes all combinations and saves PNG files starting with `gen_from_phi0.3_psi1.0_3000_...`

- [ ] **Step 3: Commit**

```bash
git add generalization_test.py
git commit -m "feat: add generalization test script"
```
