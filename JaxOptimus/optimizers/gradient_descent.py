"""Gradient descent optimizer implementation."""
from typing import Optional, Callable
import jax.numpy as jnp
from .base import Optimizer, OptimizationState
from ..line_search import LineSearch

class GradientDescent(Optimizer):
    """Gradient descent optimization algorithm."""

    def __init__(
        self,
        line_search: LineSearch,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        store_trajectory: bool = True,
    ):
        """Initialize gradient descent optimizer.

        Args:
            line_search: Line search algorithm.
            max_iterations: Maximum number of iterations.
            tolerance: Tolerance for convergence.
            store_trajectory: Whether to store the optimization trajectory.
        """
        super().__init__(
            max_iterations=max_iterations,
            tolerance=tolerance,
            store_trajectory=store_trajectory,
        )
        self.line_search = line_search

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
        # Compute initial function value and gradient
        f_x = f(x0)
        grad_x = grad(x0)

        # Initialize trajectory if needed
        trajectory = [x0] if self.store_trajectory else []

        return OptimizationState(
            x=x0,
            f_x=f_x,
            grad_x=grad_x,
            iteration=0,
            success=True,
            message="Initialization complete",
            trajectory=trajectory
        )

    def _update(
        self,
        state: OptimizationState,
        f: Callable[[jnp.ndarray], float],
        grad: Callable[[jnp.ndarray], jnp.ndarray],
        **kwargs
    ) -> OptimizationState:
        """Perform one gradient descent step.

        Args:
            state: Current optimization state.
            f: Objective function.
            grad: Gradient function.
            **kwargs: Additional parameters.

        Returns:
            Updated optimization state.
        """
        # Compute search direction (negative gradient)
        direction = -state.grad_x

        # Perform line search
        line_search_results = self.line_search.search(
            f, state.x, direction, state.f_x, grad_x=state.grad_x
        )

        # Update state
        x_new = state.x + line_search_results.step_size * direction
        f_new = f(x_new)
        grad_new = grad(x_new)

        # Update trajectory if needed
        trajectory = state.trajectory + [x_new] if self.store_trajectory else state.trajectory

        return OptimizationState(
            x=x_new,
            f_x=f_new,
            grad_x=grad_new,
            iteration=state.iteration + 1,
            success=line_search_results.success,
            message=line_search_results.message,
            trajectory=trajectory,
        )