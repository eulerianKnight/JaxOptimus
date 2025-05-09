"""Base line search module."""
from typing import Optional, Callable
from dataclasses import dataclass
import jax.numpy as jnp
import jax.lax as lax
import jax

@dataclass
class LineSearchResults:
    """Results of line search."""
    step_size: float
    success: bool
    message: str
    iterations: int = 0
    function_values: Optional[jnp.ndarray] = None
    step_sizes: Optional[jnp.ndarray] = None

class LineSearch:
    """Base class for line search algorithms."""

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """Initialize line search.

        Args:
            max_iterations: Maximum number of iterations.
            tolerance: Tolerance for convergence.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def search(
        self,
        f: Callable[[jnp.ndarray], float],
        x: jnp.ndarray,
        direction: jnp.ndarray,
        f_x: Optional[float] = None,
        grad_x: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> LineSearchResults:
        """Perform line search.

        Args:
            f: Objective function.
            x: Current point.
            direction: Search direction.
            f_x: Function value at current point (optional).
            grad_x: Gradient at current point (optional).
            **kwargs: Additional parameters to pass to _search.

        Returns:
            Line search results.
        """
        # Compute function value if not provided
        if f_x is None:
            f_x = f(x)

        # Perform line search
        results = self._search(f, x, direction, f_x, grad_x, **kwargs)

        # Return the complete results from _search
        return results

    def _search(
        self,
        f: Callable[[jnp.ndarray], float],
        x: jnp.ndarray,
        direction: jnp.ndarray,
        f_x: float,
        grad_x: Optional[jnp.ndarray] = None,
    ) -> LineSearchResults:
        """Perform line search.

        Args:
            f: Objective function.
            x: Current point.
            direction: Search direction.
            f_x: Function value at current point.
            grad_x: Gradient at current point (optional).

        Returns:
            Line search results.
        """
        raise NotImplementedError