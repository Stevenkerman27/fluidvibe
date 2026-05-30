# Taylor-Green Flow Field Visualization Design

## 1. Objective
Create a standalone visualization script `visualize_flow.py` that generates a high-quality static snapshot of the Taylor-Green flow field. The visual style will strictly mimic the existing DQN evaluation plots (`eval_dqn.py`).

## 2. Visual Specifications
- **Domain**: $[0, 2\pi] \times [0, 2\pi]$ to capture exactly 4 vortices.
- **Background (Vorticity)**:
  - Tool: `ax.pcolormesh`
  - Formula: $\omega = U_0 \cos(x) \cos(y)$
  - Colormap: `coolwarm`
  - Alpha: `0.6`
- **Vectors (Velocity)**:
  - Tool: `ax.quiver`
  - Formula: $u = -0.5 U_0 \cos(x) \sin(y)$, $v = 0.5 U_0 \sin(x) \cos(y)$
  - Color: `xkcd:rich purple` (matching the DQN agent color)
  - Grid: Sparse 20x20 grid for clarity.
- **Aesthetics**:
  - `ax.set_aspect('equal')`
  - Colorbar for vorticity.
  - LaTeX labels for axes and title.
  - Font size matching `eval.py` (14pt).

## 3. Technical Implementation
- **Dependencies**: `numpy`, `matplotlib`.
- **Configuration**: Use parameters from `config.py` (e.g., `FLOW_SPEED`).
- **Output**: Save to `pics/taylor_green_flow.png` with 300 DPI.

## 4. Code Structure
- `generate_flow_data(res)`: Computes X, Y, U, V, and Vorticity grids.
- `plot_flow(data)`: Handles all Matplotlib styling and saving.
- `main()`: Entry point.
