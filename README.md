# Gradient Descent Vis

This was a project when I was tired of just reading about "momentum" and "adaptive learning rates" and actually wanted to see them fail in real-time.

It’s a single-file Matplotlib mess that visualizes how various PyTorch optimizers (SGD, Adam, RMSprop, etc.) traverse classic optimization test functions like Rastrigin, Rosenbrock, and the dreaded Eggholder.

### Why this exists
Mostly because I wanted to see exactly why Adam "overshoots" or how SGD with momentum can sometimes slingshot out of a local minimum. Also, Matplotlib's 3D widgets are surprisingly cursed to work with, and I think I was trying to prove to myself that I could make a semi-functional UI without leaving the standard library/plotting stack.

### The "I'm not super proud of this" section
Looking back at the code:
- It’s a monolith. Everything is in `main.py`. 
- The UI layout is hardcoded with magic numbers. If you resize the window, god help you.
- I’m using `hashlib` to generate colors for the optimizers because I was too lazy to define a proper color palette.
- It recomputes the entire path every time you move a slider. It’s not "optimized," but it’s fast enough for a 100-step trajectory.
- It's purely 2D... which is very deceptive when the real world applications often have thousands of dimensions.

### How to run it
You’ll need `torch`, `numpy`, and `matplotlib`. 

```bash
pip install torch numpy matplotlib
python main.py
```

### Controls
- **Function Slider:** Switch between different test functions (Himmelblau, Beale, etc.).
- **Steps Slider:** Change the max iterations.
- **Current Step:** Scrub through the optimization timeline to see the "race" happen.
- **LR Sliders:** These are on a log scale. Sliding to `-2` means a learning rate of `0.01`.
- **Checkboxes:** Toggle specific optimizers on/off to declutter the view.

### Optimization Functions Included
- **Rastrigin:** Lots of local minima. Great for watching Adam get confused.
- **Rosenbrock:** The "banana function." Hard to find the narrow valley.
- **Eggholder:** A nightmare landscape. Most optimizers just give up.
- **Beale/Himmelblau:** Classic benchmarks for multi-modal surfaces.

### Future Plans (Maybe)
- [ ] Figure out a way to make it representative of many dimesions while being visible on a 2D screen...
