import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define test functions
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def sphere(x, y):
    return x**2 + y**2

# Function to get optimizer
def get_optimizer(opt_name, params, lr):
    if opt_name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    elif opt_name == "Momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif opt_name == "Adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.95))
    elif opt_name == "Adagrad":
        return torch.optim.Adagrad(params, lr=10*lr)
    elif opt_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9)
    else:
        raise ValueError(f"Optimizer {opt_name} not recognized")

# Function to update optimizer and track path
def optimize(func, optimizer, x, y, steps):
    path = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = func(x, y)
        loss.backward()
        optimizer.step()
        path.append((x.item(), y.item(), loss.item()))
    return path

# --- Main animation function ---
def animate_optimizers(func, opt_names, x_range, y_range, lr=0.01, steps=50, title="Optimization Paths"):
    # Create initial parameters
    x = torch.tensor(x_range[0], requires_grad=True, dtype=torch.float32)
    y = torch.tensor(y_range[0], requires_grad=True, dtype=torch.float32)

    # Initialize optimizers and paths
    optimizers = [get_optimizer(opt_name, [x, y], lr) for opt_name in opt_names]
    paths = [[] for _ in opt_names]

    # --- Setup Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for surface plot
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = func(torch.tensor(X), torch.tensor(Y))

    # Plot surface
    ax.plot_surface(X, Y, Z.numpy(), cmap='viridis', alpha=0.6)

    # Store line objects for updating data later
    lines = [ax.plot([], [], [], label=opt_names[i], lw=2)[0] for i in range(len(opt_names))]
    markers = [ax.plot([], [], [], marker='o', markersize=8, linestyle='None', color=line.get_color())[0] for line in lines]

    # --- Animation update function ---
    def update(frame):
        # Update each optimizer and its path
        for i, optimizer in enumerate(optimizers):
            optimizer.zero_grad()
            loss = func(x, y)
            loss.backward()
            optimizer.step()
            paths[i].append((x.item(), y.item(), loss.item()))

            # Get path and current position
            path_x, path_y, path_z = zip(*paths[i])

            # Update path data
            lines[i].set_data(path_x, path_y)
            lines[i].set_3d_properties(path_z)

            # Update marker position
            markers[i].set_data([path_x[-1]], [path_y[-1]])
            markers[i].set_3d_properties([path_z[-1]])

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Loss')
        ax.set_title(f"{title} - Frame {frame + 1}")
        ax.legend()
        return lines + markers

    # Create animation
    ani = FuncAnimation(fig, update, frames=steps, interval=200, blit=True)
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Example with Rosenbrock function
    animate_optimizers(
        rosenbrock,
        ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"],
        x_range=(-2, 2),
        y_range=(-1, 3),
        lr=0.001,
        steps=50,
        title="Optimization on Rosenbrock Function",
    )

    # Example with Himmelblau function
    animate_optimizers(
        himmelblau,
        ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"],
        x_range=(-5, 5),
        y_range=(-5, 5),
        lr=0.01,
        steps=50,
        title="Optimization on Himmelblau Function",
    )

    # Example with Sphere function
    animate_optimizers(
        sphere,
        ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"],
        x_range=(-5, 5),
        y_range=(-5, 5),
        lr=0.01,
        steps=50,
        title="Optimization on Sphere Function"
    )