import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import os
import config

def generate_flow_data(res=100, u0=config.FLOW_SPEED, x_range=(-np.pi, 3*np.pi), y_range=(-np.pi, 3*np.pi)):
    x = np.linspace(x_range[0], x_range[1], res)
    y = np.linspace(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(x, y)
    
    # Taylor-Green formulas
    U = -0.5 * u0 * np.cos(X) * np.sin(Y)
    V = 0.5 * u0 * np.sin(X) * np.cos(Y)
    Vort = u0 * np.cos(X) * np.cos(Y)
    
    return X, Y, U, V, Vort

def plot_flow(res_bg=200, res_vector=25, u0=config.FLOW_SPEED, save_path="pics/taylor_green_flow.png"):
    # Define ranges: Center is (pi, pi)
    x_range = (-np.pi, 3 * np.pi)
    y_range = (-0.5 * np.pi, 2.5 * np.pi)
    
    X, Y, _, _, Vort = generate_flow_data(res_bg, u0, x_range, y_range)
    X_v, Y_v, U_v, V_v, _ = generate_flow_data(res_vector, u0, x_range, y_range)
    
    plt.figure(figsize=(10, 7.5))
    plt.rcParams.update({'font.size': 18})
    ax = plt.subplot(111)
    
    # Background: Vorticity
    c = ax.pcolormesh(
        X, Y, Vort, 
        cmap="coolwarm", 
        shading="auto", 
        alpha=0.6, 
        rasterized=True,
        vmin=-u0, vmax=u0  # Ensure range is fixed to flow speed u0
    )
    cbar = plt.colorbar(c, ax=ax, shrink=0.7, label="Vorticity")
    # Ensure -1.0 and 1.0 are shown on colorbar
    cbar.set_ticks([-u0, -0.5*u0, 0, 0.5*u0, u0])
    
    # Vectors: Velocity
    ax.quiver(
        X_v, Y_v, U_v, V_v, 
        color="xkcd:rich purple", 
        scale=u0 * 8, # Increased from 5 to 8 to shorten arrows
        width=0.005
    )
    
    # Draw Initialization Area (0 to 2*pi)
    init_rect = patches.Rectangle(
        (0, 0), 2*np.pi, 2*np.pi, 
        linewidth=3, 
        edgecolor='black', 
        facecolor='none', 
        linestyle='--'
    )
    ax.add_patch(init_rect)
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    
    # Set tick density
    ax.xaxis.set_major_locator(MultipleLocator(np.pi))
    ax.yaxis.set_major_locator(MultipleLocator(np.pi))
    
    # Custom labels for Pi multiples
    def format_func(value, tick_number):
        n = value / np.pi
        if abs(n) < 1e-6: return "0"
        if abs(n - 1) < 1e-6: return r"$\pi$"
        if abs(n + 1) < 1e-6: return r"$-\pi$"
        if n % 1 == 0:
            return rf"${int(n)}\pi$"
        return rf"${n:.1f}\pi$"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()
    
    if not os.path.exists("pics"):
        os.makedirs("pics")
        
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Flow visualization saved to {save_path}")

if __name__ == "__main__":
    plot_flow()
