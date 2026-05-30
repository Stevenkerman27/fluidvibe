import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import config

def generate_flow_data(res=100, u0=config.FLOW_SPEED):
    x = np.linspace(0.5 * np.pi, 2.5 * np.pi, res)
    y = np.linspace(0.5 * np.pi, 2.5 * np.pi, res)
    X, Y = np.meshgrid(x, y)
    
    # Taylor-Green formulas
    U = -0.5 * u0 * np.cos(X) * np.sin(Y)
    V = 0.5 * u0 * np.sin(X) * np.cos(Y)
    Vort = u0 * np.cos(X) * np.cos(Y)
    
    return X, Y, U, V, Vort

def plot_flow(res_bg=100, res_vector=15, u0=config.FLOW_SPEED, save_path="pics/taylor_green_flow.png"):
    X, Y, _, _, Vort = generate_flow_data(res_bg, u0)
    X_v, Y_v, U_v, V_v, _ = generate_flow_data(res_vector, u0)
    
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 22}) # Further enlarged font
    ax = plt.subplot(111)
    
    # Background: Vorticity
    c = ax.pcolormesh(
        X, Y, Vort, 
        cmap="coolwarm", 
        shading="auto", 
        alpha=0.6, 
        rasterized=True
    )
    # Shrinked colorbar height (shrink=0.5)
    plt.colorbar(c, ax=ax, shrink=0.5, label="Vorticity")
    
    # Vectors: Velocity
    ax.quiver(
        X_v, Y_v, U_v, V_v, 
        color="xkcd:rich purple", 
        scale=u0 * 5,
        width=0.008
    )
    
    ax.set_xlim(0.5 * np.pi, 2.5 * np.pi)
    ax.set_ylim(0.5 * np.pi, 2.5 * np.pi)
    ax.set_aspect("equal")
    
    # Set tick density to 1
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    # plt.title(rf"Taylor-Green Flow Field ($U_0={u0}$)") 
    plt.tight_layout()
    
    if not os.path.exists("pics"):
        os.makedirs("pics")
        
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Flow visualization saved to {save_path}")

if __name__ == "__main__":
    plot_flow()
