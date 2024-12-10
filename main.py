import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
import hashlib

# Define test functions
def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))

def sphere(x, y):
    return x**2 + y**2

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def eggholder(x, y):
    return -(y + 47) * torch.sin(torch.sqrt(torch.abs(x / 2 + (y + 47)))) - x * torch.sin(torch.sqrt(torch.abs(x - (y + 47))))

def beale(x, y):
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

def goldstein_price(x, y):
    part1 = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    part2 = (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return part1 * part2

test_functions = {
    "Rastrigin": rastrigin,
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Himmelblau": himmelblau,
    "Eggholder": eggholder,
    "Beale": beale,
    "Goldstein-Price": goldstein_price
}

# --- Optimization Setup ---
n_steps = 250  # Default number of steps
initial_lr = 0.01 #Default learning rate
initial_step = 100

x_range = (-5, 5)
y_range = (-5, 5)

# Adjust ranges for specific functions
function_ranges = {
    "Rastrigin": (-5, 5),
    "Sphere": (-5, 5),
    "Rosenbrock": (-2, 2),
    "Himmelblau": (-5, 5),
    "Eggholder": (-512, 512),
    "Beale": (-4.5, 4.5),
    "Goldstein-Price": (-2, 2)
}

x = torch.linspace(x_range[0], x_range[1], 100)
y = torch.linspace(y_range[0], y_range[1], 100)
X, Y = torch.meshgrid(x, y, indexing='ij')

# --- Matplotlib Figure Setup ---
fig = plt.figure(figsize=(14, 8)) #Increased width
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2, left=0.1)  # Make more space for sliders and buttons

# --- Slider Setup ---
ax_steps = plt.axes([0.1, 0.05, 0.3, 0.03])
slider_steps = Slider(ax_steps, 'Steps', 10, 1000, valinit=n_steps, valstep=1)

ax_step = plt.axes([0.1, 0.025, 0.3, 0.03]) #Slider for current step
slider_step = Slider(ax_step, 'Current Step', 0, n_steps, valinit=initial_step, valstep=1)

# --- Function Selection ---
function_names = list(test_functions.keys())
current_function_index = 2
ax_function = plt.axes([0.1, 0.075, 0.3, 0.03]) #New slider for function selection
slider_function = Slider(ax_function, 'Function', 0, len(function_names) - 1, valinit=current_function_index, valstep=1)
 
# --- Optimizer Selection and Configuration ---
optimizer_configs = [
    {"name": "SGD", "class": optim.SGD, "params": {"lr": initial_lr, "momentum": 0.9}},
    {"name": "SGD (no momentum)", "class": optim.SGD, "params": {"lr": initial_lr}},
    {"name": "Adam", "class": optim.Adam, "params": {"lr": initial_lr, "betas": (0.9, 0.95)}},
    {"name": "Adam High Grade", "class": optim.Adam, "params": {"lr": initial_lr, "betas": (0.9, 0.99)}},
    {"name": "Adagrad", "class": optim.Adagrad, "params": {"lr": initial_lr}},
    {"name": "RMSprop", "class": optim.RMSprop, "params": {"lr": initial_lr, "alpha": 0.99}}
]

# Create learning rate sliders dynamically for each optimizer configuration
lr_sliders = {}
optimizer_visibility = []
ax_lr_base = 0.025
ax_lr_height = 0.03
lr_slider_space = 0.025
for i, config in enumerate(optimizer_configs):
    opt_name = config["name"]
    ax_lr = plt.axes([0.6, ax_lr_base + i * lr_slider_space, 0.3, ax_lr_height])
    lr_sliders[opt_name] = Slider(
        ax_lr,
        f"{opt_name} LR",
        -6,  # min value (10^-6)
        1,   # max value (10^1 = 10)
        valinit=np.log10(initial_lr),  # Initial value in log scale
        valfmt="%1.1f"  # Display log value on the slider
    )
    optimizer_visibility.append(True)

# Checkbox for optimizer visibility
rax = plt.axes([0.05, 0.6, 0.2, 0.3])
optimizer_checkbox = CheckButtons(rax, [config["name"] for config in optimizer_configs], optimizer_visibility)

# --- Optimization and Plotting ---
paths = {}
markers = {}

def init_optimizers(selected_function, n_steps, lr_dict, current_step):
    paths.clear()
    Z = selected_function(X, Y)
    for config in optimizer_configs:
        opt_name = config["name"]
        opt_class = config["class"]
        params = config["params"].copy()
        params["lr"] = lr_dict[opt_name]
        x_init = torch.tensor([2.0, -2.0], requires_grad=True)
        optimizer = opt_class([x_init], **params)

        path = [x_init.detach().clone().numpy()]
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = selected_function(x_init[0], x_init[1])
            loss.backward()
            optimizer.step()
            path.append(x_init.detach().clone().numpy())

        paths[opt_name] = np.array(path)

def get_color_from_name(name):
    """Hash the name to get a unique color."""
    hash_object = hashlib.sha256(name.encode())
    hex_dig = hash_object.hexdigest()
    # Take the first 6 characters of the hash and convert to an RGB color
    color_code = "#" + hex_dig[:6]
    return color_code

# --- Update Function ---
def update(val):
    global current_function_index
    n_steps = int(slider_steps.val)
    current_step = int(slider_step.val)
    current_function_index = int(slider_function.val)
    selected_function_name = function_names[current_function_index]
    selected_function = test_functions[selected_function_name]
    x_range = function_ranges[selected_function_name]
    y_range = function_ranges[selected_function_name]

    x = torch.linspace(x_range[0], x_range[1], 100)
    y = torch.linspace(y_range[0], y_range[1], 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Convert slider values from log scale to linear scale for learning rates
    lr_dict = {opt_name: 10**slider.val for opt_name, slider in lr_sliders.items()}

    
    init_optimizers(selected_function, n_steps, lr_dict, current_step)
    
    
    ax.cla()  # Clear the previous plot
    
    Z = selected_function(X, Y)
    
    # Plot the surface
    ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', alpha=0.8)

    # Plot optimization paths and mark current position
    for i, (opt_name, path) in enumerate(paths.items()):
        if not optimizer_visibility[i]:
            continue
        # Use the hash of the name to get a unique color
        color = get_color_from_name(opt_name)
        
        # Display the path up to the current step
        ax.plot(path[:current_step+1, 0], path[:current_step+1, 1], selected_function(torch.tensor(path[:current_step+1, 0]), torch.tensor(path[:current_step+1, 1])).numpy(),
                linestyle='-', label=opt_name, color = color, alpha=0.7, zorder=8, linewidth=3)
        
        # Mark the last position with a larger marker
        markers[opt_name] = ax.plot([path[current_step, 0]], [path[current_step, 1]], [selected_function(torch.tensor(path[current_step, 0]), torch.tensor(path[current_step, 1])).numpy()],
                marker='o', color=color, markersize=8, zorder=8)
        
    # Customize plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{selected_function_name} Function Optimization')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend()
    fig.canvas.draw_idle()

def update_current_step_slider(val):
    n_steps = int(slider_steps.val)
    slider_step.valmax = n_steps
    slider_step.ax.set_xlim(slider_step.valmin, slider_step.valmax)
    fig.canvas.draw_idle()
    
def optimizer_checkbox_fun(label):
    index = [config["name"] for config in optimizer_configs].index(label)
    optimizer_visibility[index] = not optimizer_visibility[index]
    update(None)

# --- Connect Sliders and Checkbox ---
slider_steps.on_changed(update_current_step_slider)
slider_step.on_changed(update)
slider_function.on_changed(update)
for slider in lr_sliders.values():
    slider.on_changed(update)
optimizer_checkbox.on_clicked(optimizer_checkbox_fun)

# --- Initialize and Show Plot ---
update(None)  # Initial plot
plt.show()