"""Base classes for optimization algorithms."""
from typing import Optional, Callable, List, Any
from dataclasses import dataclass
import jax.numpy as jnp
from ..line_search import LineSearch

@dataclass
class OptimizationState:
    """State of the optimization process."""
    x: jnp.ndarray
    f_x: float
    grad_x: jnp.ndarray
    iteration: int
    success: bool
    message: str
    trajectory: List[jnp.ndarray]

class Optimizer:
    """Base class for optimization algorithms."""

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        store_trajectory: bool = True,
    ):
        """Initialize optimizer.

        Args:
            max_iterations: Maximum number of iterations.
            tolerance: Tolerance for convergence.
            store_trajectory: Whether to store the optimization trajectory.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.store_trajectory = store_trajectory

    def _init_state(
        self,
        f: Callable[[jnp.ndarray], float],
        grad: Callable[[jnp.ndarray], jnp.ndarray],
        x0: jnp.ndarray,
        **kwargs
    ) -> OptimizationState:
        """Initialize the optimization state.

        Args:
            f: Objective function.
            grad: Gradient function.
            x0: Initial point.
            **kwargs: Additional parameters.

        Returns:
            Initial optimization state.
        """
        raise NotImplementedError

    def _update(
        self,
        state: OptimizationState,
        f: Callable[[jnp.ndarray], float],
        grad: Callable[[jnp.ndarray], jnp.ndarray],
        **kwargs
    ) -> OptimizationState:
        """Update the optimization state.

        Args:
            state: Current optimization state.
            f: Objective function.
            grad: Gradient function.
            **kwargs: Additional parameters.

        Returns:
            Updated optimization state.
        """
        raise NotImplementedError

    def _check_convergence(self, state: OptimizationState) -> bool:
        """Check if optimization has converged.

        Args:
            state: Current optimization state.

        Returns:
            True if converged, False otherwise.
        """
        # Check gradient norm
        grad_norm = jnp.linalg.norm(state.grad_x)
        if grad_norm < self.tolerance:
            return True

        # Check maximum iterations
        if state.iteration >= self.max_iterations:
            return False

        return False

    def minimize(
        self,
        f: Callable[[jnp.ndarray], float],
        grad: Callable[[jnp.ndarray], jnp.ndarray],
        x0: jnp.ndarray,
        **kwargs
    ) -> OptimizationState:
        """Minimize the objective function.

        Args:
            f: Objective function.
            grad: Gradient function.
            x0: Initial point.
            **kwargs: Additional parameters.

        Returns:
            Final optimization state.
        """
        # Initialize state
        state = self._init_state(f, grad, x0, **kwargs)

        # Main optimization loop
        while not self._check_convergence(state):
            state = self._update(state, f, grad, **kwargs)

        # Set success flag and message
        if state.iteration >= self.max_iterations:
            state.success = False
            state.message = "Maximum iterations reached without convergence."
        else:
            state.success = True
            state.message = "Optimization converged successfully."

        return state

    def _update_trajectory(
        self,
        trajectory: Optional[List[jnp.ndarray]],
        x: jnp.ndarray,
        f_x: float
    ) -> Optional[List[jnp.ndarray]]:
        """Update the optimization trajectory.

        Args:
            trajectory: Current trajectory.
            x: Current point.
            f_x: Current function value.

        Returns:
            Updated trajectory.
        """
        if trajectory is not None:
            trajectory.append(x)
        return trajectory