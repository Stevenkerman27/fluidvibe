# Taylor-Green Flow Field Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone visualization script `visualize_flow.py` that generates a static snapshot of the Taylor-Green flow field matching the DQN aesthetic.

**Architecture:** A script that calculates vorticity and velocity grids over a $[0, 2\pi]^2$ domain and uses Matplotlib's `pcolormesh` and `quiver` for visualization.

**Tech Stack:** Python, NumPy, Matplotlib.

---

### Task 1: Flow Calculation Module

**Files:**
- Create: `visualize_flow.py`
- Test: `tests/test_visualize_flow.py`

- [ ] **Step 1: Write a test for flow data generation**

```python
import numpy as np
import pytest
from visualize_flow import generate_flow_data

def test_generate_flow_data():
    res = 10
    u0 = 1.0
    X, Y, U, V, Vort = generate_flow_data(res, u0)
    
    # Check shapes
    assert X.shape == (res, res)
    assert Y.shape == (res, res)
    assert U.shape == (res, res)
    assert V.shape == (res, res)
    assert Vort.shape == (res, res)
    
    # Check specific values at origin (x=0, y=0)
    # Vorticity: u0 * cos(0) * cos(0) = u0
    assert np.isclose(Vort[0, 0], u0)
    # U velocity: -0.5 * u0 * cos(0) * sin(0) = 0
    assert np.isclose(U[0, 0], 0)
    # V velocity: 0.5 * u0 * sin(0) * cos(0) = 0
    assert np.isclose(V[0, 0], 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualize_flow.py`
Expected: FAIL (Module not found)

- [ ] **Step 3: Implement `generate_flow_data`**

```python
import numpy as np
import matplotlib.pyplot as plt
import os
import config

def generate_flow_data(res=100, u0=config.FLOW_SPEED):
    x = np.linspace(0, 2 * np.pi, res)
    y = np.linspace(0, 2 * np.pi, res)
    X, Y = np.meshgrid(x, y)
    
    # Taylor-Green formulas
    U = -0.5 * u0 * np.cos(X) * np.sin(Y)
    V = 0.5 * u0 * np.sin(X) * np.cos(Y)
    Vort = u0 * np.cos(X) * np.cos(Y)
    
    return X, Y, U, V, Vort
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualize_flow.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add visualize_flow.py tests/test_visualize_flow.py
git commit -m "feat: add flow data generation with tests"
```

---

### Task 2: Plotting and Main Entry Point

**Files:**
- Modify: `visualize_flow.py`

- [ ] **Step 1: Implement `plot_flow` function**

```python
def plot_flow(res_bg=100, res_vector=20, u0=config.FLOW_SPEED, save_path="pics/taylor_green_flow.png"):
    X, Y, _, _, Vort = generate_flow_data(res_bg, u0)
    X_v, Y_v, U_v, V_v, _ = generate_flow_data(res_vector, u0)
    
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    
    # Background: Vorticity
    c = ax.pcolormesh(
        X, Y, Vort, 
        cmap="coolwarm", 
        shading="auto", 
        alpha=0.6, 
        rasterized=True
    )
    plt.colorbar(c, ax=ax, shrink=0.8, label="Vorticity")
    
    # Vectors: Velocity
    ax.quiver(
        X_v, Y_v, U_v, V_v, 
        color="xkcd:rich purple", 
        scale=u0 * 5,
        width=0.005
    )
    
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(rf"Taylor-Green Flow Field ($U_0={u0}$)")
    plt.tight_layout()
    
    if not os.path.exists("pics"):
        os.makedirs("pics")
        
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Flow visualization saved to {save_path}")
```

- [ ] **Step 2: Add `if __name__ == "__main__":` block**

```python
if __name__ == "__main__":
    plot_flow()
```

- [ ] **Step 3: Run the script and verify output**

Run: `python visualize_flow.py`
Expected: "Flow visualization saved to pics/taylor_green_flow.png" and the file exists.

- [ ] **Step 4: Commit**

```bash
git add visualize_flow.py
git commit -m "feat: implement flow plotting and CLI entry point"
```
