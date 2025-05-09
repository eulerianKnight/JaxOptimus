import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import JaxOptimus

def main():
    # Enable 64-bit precision for better numerical stability
    jax.config.update("jax_enable_x64", True)
    
    # Create a Rosenbrock function
    func = JaxOptimus.functions.quadratic.RosenbrockFunction()
    
    # Initial point
    x0 = jnp.array([-1.0, 1.0])  # Changed initial point to be more suitable for Rosenbrock
    
    print(f"Function: {func.name}")
    print(f"Dimension: {func.dim}")
    print(f"Initial point condition number: {func.condition_number(x0)}")
    print(f"Global minimum: {func.global_minimum}")
    
    # Create optimizers
    gd_backtracking = JaxOptimus.create_gradient_descent(
        line_search_type='backtracking',
        initial_step_size=0.001,  # Smaller step size for Rosenbrock
        max_iterations=1000  # More iterations for Rosenbrock
    )
    
    gd_exact = JaxOptimus.create_gradient_descent(
        line_search_type='exact',
        max_iterations=1000
    )
    
    newton = JaxOptimus.create_newton_method(
        line_search_type='backtracking',
        initial_step_size=1.0,
        max_iterations=100
    )
    
    cg = JaxOptimus.create_conjugate_gradient(
        line_search_type='exact',
        max_iterations=1000
    )
    
    # Run optimizers
    print("\nRunning gradient descent with backtracking line search...")
    results_gd_back = gd_backtracking.minimize(func, func.gradient, x0)
    
    print("\nRunning gradient descent with exact line search...")
    results_gd_exact = gd_exact.minimize(func, func.gradient, x0)
    
    print("\nRunning Newton's method...")
    results_newton = newton.minimize(func, func.gradient, x0, hessian_fn=func.hessian)
    
    print("\nRunning conjugate gradient...")
    results_cg = cg.minimize(func, func.gradient, x0)
    
    # Print results
    print("\nResults:")
    print(f"GD Backtracking: {results_gd_back.iteration} iterations, f(x) = {results_gd_back.f_x:.6e}")
    print(f"GD Exact: {results_gd_exact.iteration} iterations, f(x) = {results_gd_exact.f_x:.6e}")
    print(f"Newton: {results_newton.iteration} iterations, f(x) = {results_newton.f_x:.6e}")
    print(f"CG: {results_cg.iteration} iterations, f(x) = {results_cg.f_x:.6e}")
    
    # Extract trajectories
    traj_gd_back = results_gd_back.trajectory
    traj_gd_exact = results_gd_exact.trajectory
    traj_newton = results_newton.trajectory
    traj_cg = results_cg.trajectory
    
    # Visualization
    vis_bounds = func.get_visualization_bounds()
    
    # Use matplotlib's default style
    plt.style.use('default')
    
    # Plot contours and trajectories
    plt.figure(figsize=(12, 8))  # Increased width to accommodate legend
    ax1 = JaxOptimus.plot_contour(
        func,
        vis_bounds,
        [traj_gd_back, traj_gd_exact, traj_newton, traj_cg],
        ["GD Backtracking", "GD Exact", "Newton", "CG"],
        "Optimization Trajectories"
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for legend
    
    # Plot convergence
    plt.figure(figsize=(10, 8))
    ax2 = JaxOptimus.plot_convergence(
        [results_gd_back, results_gd_exact, results_newton, results_cg],
        ["GD Backtracking", "GD Exact", "Newton", "CG"],
        "Convergence Comparison",
        func
    )
    plt.tight_layout()
    
    # Plot step sizes
    plt.figure(figsize=(10, 8))
    ax3 = JaxOptimus.plot_step_sizes(
        [results_gd_back, results_gd_exact, results_newton, results_cg],
        ["GD Backtracking", "GD Exact", "Newton", "CG"],
        "Step Sizes Comparison"
    )
    plt.tight_layout()
    
    # Display all figures
    plt.show()

if __name__ == "__main__":
    main()