# JaxOpt: Optimization Algorithms in JAX

JaxOpt is a modular optimization package built on JAX, designed for experimentation with various optimization algorithms. The package features efficient implementations of line search methods and optimization algorithms with visualization capabilities.

## Features

- **Modular Design**: Easily switch between different optimization algorithms and line search methods
- **JAX Acceleration**: Leverage JAX's automatic differentiation, compilation (JIT), and hardware acceleration
- **Visualization Tools**: Included utilities for visualizing optimization trajectories and convergence behavior
- **Extensible Framework**: Built to be easily extendable with new algorithms and test functions

## Installation

### Setting up a Virtual Environment

#### Option 1: Using venv (Python's built-in virtual environment)

```bash
# Create a virtual environment
python -m venv jaxopt-env

# Activate the virtual environment
# On Windows:
jaxopt-env\Scripts\activate
# On macOS/Linux:
source jaxopt-env/bin/activate
```

#### Option 2: Using conda

```bash
# Create a conda environment
conda create -n jaxopt-env python=3.9
conda activate jaxopt-env
```

### Installing JaxOpt

```bash
# Clone the repository
git clone https://github.com/yourusername/jaxopt.git
cd jaxopt

# Install the package in development mode
pip install -e .

# Install additional development dependencies (optional)
pip install -e ".[dev]"
```

### Installing JAX with GPU Support (Optional)

For GPU acceleration, follow the [official JAX installation instructions](https://github.com/google/jax#installation).

```bash
# Example for CUDA 11.8
pip install --upgrade "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Basic Usage

Here's a simple example to get you started:

```python
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Create a test function (ill-conditioned quadratic)
function = jaxopt.functions.IllConditionedQuadratic(condition_number=100.0)

# Define starting point
x0 = jnp.array([1.0, 100.0])

# Create an optimizer (gradient descent with exact line search)
optimizer = jaxopt.create_gradient_descent(line_search_type='exact')

# Run the optimization
results = optimizer.minimize(function, function.gradient, x0)

# Print results
print(f"Minimum found at: {results.x}")
print(f"Function value: {results.f_x}")
print(f"Iterations: {results.iterations}")
print(f"Gradient norm: {jnp.linalg.norm(results.grad_x)}")

# Visualize the optimization path
vis_bounds = function.get_visualization_bounds()
trajectory = [state.x for state in results.trajectory]

plt.figure(figsize=(10, 8))
jaxopt.viz.Visualizer.plot_contour(
    function, 
    vis_bounds, 
    [trajectory], 
    ["Gradient Descent"]
)
plt.title("Optimization Trajectory")
plt.tight_layout()
plt.show()
```

## Comparing Different Optimization Methods

```python
import jaxopt
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create Rosenbrock function
rosenbrock = jaxopt.functions.RosenbrockFunction()
x0 = jnp.array([-1.2, 1.0])

# Create different optimizers
gd = jaxopt.create_gradient_descent(line_search_type='backtracking')
newton = jaxopt.create_newton_method()
cg = jaxopt.create_conjugate_gradient(beta_method='polak_ribiere')

# Run optimization
results_gd = gd.minimize(rosenbrock, rosenbrock.gradient, x0)
results_newton = newton.minimize(rosenbrock, rosenbrock.gradient, x0, 
                                 hessian_fn=rosenbrock.hessian)
results_cg = cg.minimize(rosenbrock, rosenbrock.gradient, x0)

# Extract trajectories
traj_gd = [state.x for state in results_gd.trajectory]
traj_newton = [state.x for state in results_newton.trajectory]
traj_cg = [state.x for state in results_cg.trajectory]

# Visualize
vis_bounds = rosenbrock.get_visualization_bounds()

# Plot trajectories
plt.figure(figsize=(12, 10))
jaxopt.viz.Visualizer.plot_contour(
    rosenbrock,
    vis_bounds,
    [traj_gd, traj_newton, traj_cg],
    ["Gradient Descent", "Newton's Method", "Conjugate Gradient"],
    "Optimization Trajectories"
)
plt.tight_layout()

# Plot convergence
plt.figure(figsize=(10, 6))
jaxopt.viz.Visualizer.plot_convergence(
    [results_gd, results_newton, results_cg],
    ["Gradient Descent", "Newton's Method", "Conjugate Gradient"],
    "Convergence Comparison"
)
plt.tight_layout()
plt.show()
```

## Available Optimization Algorithms

- **Gradient Descent**: First-order method using the negative gradient as search direction
- **Newton's Method**: Second-order method using Hessian information
- **Conjugate Gradient**: Efficient method that generates conjugate search directions

## Available Line Search Methods

- **Backtracking Line Search**: Simple method using the Armijo condition
- **Exact Line Search**: Golden section search for finding the optimal step size
- **Quadratic Line Search**: Analytical solution for quadratic functions

## Creating Custom Test Functions

You can create custom test functions by extending the `Function` base class:

```python
import jax.numpy as jnp
from jaxopt.functions.base import Function

class MyCustomFunction(Function):
    def __init__(self):
        super().__init__(
            dim=2,
            name="My Custom Function",
            global_minimum=(jnp.array([0.0, 0.0]), 0.0)
        )
    
    def _evaluate(self, x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
    
    def _gradient(self, x):
        return jnp.array([2*x[0] + x[1], 4*x[1] + x[0]])
```

## Creating Custom Optimizers

Extend the `Optimizer` base class to implement new optimization algorithms:

```python
from jaxopt.optimizers.base import Optimizer, OptimizerState
import jax.numpy as jnp

class MyCustomOptimizer(Optimizer):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def _init_state(self, f, grad, x0, **kwargs):
        # Initialize your optimizer's state
        f_x = f(x0)
        grad_x = grad(x0)
        
        return OptimizerState(
            x=x0,
            f_x=f_x,
            grad_x=grad_x,
            iteration=0,
            gradient_norm=jnp.linalg.norm(grad_x),
            step_size=0.0,
            improvement=0.0,
            success=True,
            # Add any extra state variables here
        )
    
    def _update(self, state, f, grad, **kwargs):
        # Implement your optimization step here
        direction = -state.grad_x  # Example direction
        
        # Determine step size (using line search or fixed)
        step_size = 0.1  # Example fixed step size
        if self.line_search is not None:
            line_search_results = self.line_search.search(
                f, state.x, direction, state.f_x, grad_x=state.grad_x
            )
            step_size = line_search_results.step_size
        
        # Update position
        x_new = state.x + step_size * direction
        
        # Compute new function value and gradient
        f_new = f(x_new)
        grad_new = grad(x_new)
        
        # Return updated state
        return OptimizerState(
            x=x_new,
            f_x=f_new,
            grad_x=grad_new,
            iteration=state.iteration + 1,
            gradient_norm=jnp.linalg.norm(grad_new),
            step_size=step_size,
            improvement=state.f_x - f_new,
            success=f_new <= state.f_x,
            # Update any extra state variables here
        )
```

## API Documentation

### Optimizer Classes

- `GradientDescent`: First-order method
- `NewtonMethod`: Second-order method
- `ConjugateGradient`: Nonlinear conjugate gradient

### Line Search Classes

- `BacktrackingLineSearch`: Using Armijo condition
- `GoldenSectionSearch`: Exact line search
- `QuadraticLineSearch`: For quadratic functions

### Function Classes

- `QuadraticFunction`: General quadratic function
- `IllConditionedQuadratic`: Quadratic with high condition number
- `RosenbrockFunction`: Classic non-convex test function

### Helper Functions

- `create_gradient_descent()`: Create a gradient descent optimizer
- `create_newton_method()`: Create a Newton's method optimizer
- `create_conjugate_gradient()`: Create a conjugate gradient optimizer

## Visualizing Results

JaxOpt provides several visualization utilities through the `Visualizer` class:

```python
# Plot contours and optimization path
jaxopt.viz.Visualizer.plot_contour(
    function,          # Function to plot
    visualization_bounds,  # Dictionary with x_range, y_range, levels
    [trajectory],      # List of trajectories to plot
    ["Optimizer"],     # Labels for trajectories
    "Contour Plot"     # Title
)

# Plot convergence behavior
jaxopt.viz.Visualizer.plot_convergence(
    [results],        # List of optimization results
    ["Optimizer"],    # Labels for results
    "Convergence Plot" # Title
)

# Plot step sizes
jaxopt.viz.Visualizer.plot_step_sizes(
    [results],        # List of optimization results
    ["Optimizer"],    # Labels for results
    "Step Sizes"      # Title
)
```

## Running Examples

The package includes several example scripts:

```bash
# Run the optimization comparison example
python examples/optimization_comparison.py

# Run ill-conditioned quadratic example
python examples/ill_conditioned_quadratic.py
```

## Contributing

Contributions are welcome! To add new optimization algorithms or test functions:

1. Follow the modular design pattern
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.