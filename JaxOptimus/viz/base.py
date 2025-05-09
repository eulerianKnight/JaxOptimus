"""Base visualization module."""
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..optimizers.base import OptimizationState

def plot_contour(
    func,
    vis_bounds: Dict[str, Any],
    trajectories: List[List[np.ndarray]],
    labels: List[str],
    title: str = "Optimization Trajectories",
) -> plt.Axes:
    """Plot contours and optimization trajectories."""
    # Get current axis
    ax = plt.gca()
    
    # Calculate the bounds to include all trajectory points
    all_points = np.vstack([np.array(traj) for traj in trajectories])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    # Extend bounds to include both visualization bounds and trajectory points
    x_min = min(x_min, vis_bounds["x_range"][0])
    x_max = max(x_max, vis_bounds["x_range"][1])
    y_min = min(y_min, vis_bounds["y_range"][0])
    y_max = max(y_max, vis_bounds["y_range"][1])
    
    # Add padding
    margin = 0.2  # 20% margin
    x_padding = (x_max - x_min) * margin
    y_padding = (y_max - y_min) * margin
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Plot contours with extended bounds
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = func(np.array([X[j, i], Y[j, i]]))
    
    # Plot filled contours for better visibility
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax, label='Function Value')
    
    # Plot contour lines
    ax.contour(X, Y, Z, levels=20, colors='k', alpha=0.3, linewidths=0.5)
    
    # Plot trajectories
    colors = ['r', 'b', 'g', 'm']
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], f'{colors[i % len(colors)]}.-', label=label, linewidth=1.5, markersize=4)
        ax.plot(traj[0, 0], traj[0, 1], f'{colors[i % len(colors)]}o', markersize=8, label=f'{label} Start')
        ax.plot(traj[-1, 0], traj[-1, 1], f'{colors[i % len(colors)]}s', markersize=8, label=f'{label} End')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # Move legend outside the plot to the right
    ax.legend(bbox_to_anchor=(1.3, 1.0), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return ax

def plot_convergence(
    results: List[OptimizationState],
    labels: List[str],
    title: str = "Convergence Comparison",
    objective_fn=None,
) -> plt.Axes:
    """Plot convergence behavior of different optimizers."""
    # Get current axis
    ax = plt.gca()
    
    colors = ['r', 'b', 'g', 'm']
    max_iter = 0
    min_val = float('inf')
    max_val = float('-inf')
    
    for i, (result, label) in enumerate(zip(results, labels)):
        # Calculate function values along the trajectory
        if objective_fn is not None:
            # Use provided objective function
            f_values = [float(objective_fn(x)) for x in result.trajectory]
        else:
            # Fallback to stored function values
            f_values = [float(result.f_x)] * len(result.trajectory)
        
        iterations = range(len(f_values))
        max_iter = max(max_iter, len(iterations))
        min_val = min(min_val, min(f_values))
        max_val = max(max_val, max(f_values))
        
        ax.semilogy(iterations, f_values, f'{colors[i % len(colors)]}.-', label=label, linewidth=1.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Function Value (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    # Set axis limits with padding
    ax.set_xlim(-max_iter * 0.05, max_iter * 1.05)
    ax.set_ylim(min_val * 0.5, max_val * 2.0)
    
    return ax

def plot_step_sizes(
    results: List[OptimizationState],
    labels: List[str],
    title: str = "Step Sizes Comparison",
) -> plt.Axes:
    """Plot step sizes used by different optimizers."""
    # Get current axis
    ax = plt.gca()
    
    colors = ['r', 'b', 'g', 'm']
    max_iter = 0
    min_step = float('inf')
    max_step = float('-inf')
    
    for i, (result, label) in enumerate(zip(results, labels)):
        # Calculate step sizes as distances between consecutive points
        trajectory = np.array(result.trajectory)
        step_sizes = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
        iterations = range(len(step_sizes))
        max_iter = max(max_iter, len(iterations))
        min_step = min(min_step, min(step_sizes))
        max_step = max(max_step, max(step_sizes))
        
        ax.semilogy(iterations, step_sizes, f'{colors[i % len(colors)]}.-', label=label, linewidth=1.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Step Size (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    # Set axis limits with padding
    ax.set_xlim(-max_iter * 0.05, max_iter * 1.05)
    ax.set_ylim(min_step * 0.5, max_step * 2.0)
    
    return ax

class Visualizer:
    """Base class for visualization."""

    def __init__(self):
        """Initialize visualizer."""
        pass

    def plot_trajectory(
        self,
        state: OptimizationState,
        f=None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        n_points: int = 100,
        show: bool = True,
    ):
        """Plot optimization trajectory.

        Args:
            state: Optimization state.
            f: Function to plot contours for (if 2D).
            xlim: x-axis limits.
            ylim: y-axis limits.
            n_points: Number of points for contour plot.
            show: Whether to show the plot.
        """
        if state.trajectory is None:
            raise ValueError("No trajectory stored in optimization state.")

        trajectory = np.array(state.trajectory)
        if trajectory.shape[1] != 2:
            raise ValueError("Can only plot 2D trajectories.")

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot contours if function provided
        if f is not None:
            if xlim is None or ylim is None:
                raise ValueError("Must provide xlim and ylim for contour plot.")

            x = np.linspace(xlim[0], xlim[1], n_points)
            y = np.linspace(ylim[0], ylim[1], n_points)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            for i in range(n_points):
                for j in range(n_points):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

            plt.contour(X, Y, Z, levels=20)

        # Plot trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Trajectory')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Optimization Trajectory')
        plt.legend()
        plt.grid(True)

        if show:
            plt.show()

    def animate_trajectory(
        self,
        state: OptimizationState,
        f=None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        n_points: int = 100,
        interval: int = 200,
        show: bool = True,
    ) -> Optional[FuncAnimation]:
        """Animate optimization trajectory.

        Args:
            state: Optimization state.
            f: Function to plot contours for (if 2D).
            xlim: x-axis limits.
            ylim: y-axis limits.
            n_points: Number of points for contour plot.
            interval: Animation interval in milliseconds.
            show: Whether to show the animation.

        Returns:
            Animation object if show is False, None otherwise.
        """
        if state.trajectory is None:
            raise ValueError("No trajectory stored in optimization state.")

        trajectory = np.array(state.trajectory)
        if trajectory.shape[1] != 2:
            raise ValueError("Can only animate 2D trajectories.")

        # Create figure
        fig = plt.figure(figsize=(10, 8))

        # Plot contours if function provided
        if f is not None:
            if xlim is None or ylim is None:
                raise ValueError("Must provide xlim and ylim for contour plot.")

            x = np.linspace(xlim[0], xlim[1], n_points)
            y = np.linspace(ylim[0], ylim[1], n_points)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            for i in range(n_points):
                for j in range(n_points):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

            plt.contour(X, Y, Z, levels=20)

        # Initialize plot
        line, = plt.plot([], [], 'r.-', label='Trajectory')
        point, = plt.plot([], [], 'bo', label='Current Point')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Optimization Trajectory')
        plt.legend()
        plt.grid(True)

        # Set axis limits
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # Animation update function
        def update(frame):
            line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
            point.set_data(trajectory[frame:frame+1, 0], trajectory[frame:frame+1, 1])
            return line, point

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            interval=interval,
            blit=True
        )

        if show:
            plt.show()
            return None
        return anim

    @staticmethod
    def plot_contour(*args, **kwargs):
        """Static method wrapper for plot_contour function."""
        return plot_contour(*args, **kwargs)
    
    @staticmethod
    def plot_convergence(*args, **kwargs):
        """Static method wrapper for plot_convergence function."""
        return plot_convergence(*args, **kwargs)
    
    @staticmethod
    def plot_step_sizes(*args, **kwargs):
        """Static method wrapper for plot_step_sizes function."""
        return plot_step_sizes(*args, **kwargs)