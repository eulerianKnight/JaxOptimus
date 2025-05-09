"""Backtracking line search algorithm."""
import jax.numpy as jnp
from .base import LineSearch, LineSearchResults
from typing import Callable, Optional
import jax

class BacktrackingLineSearch(LineSearch):
    """Backtracking line search algorithm."""

    def __init__(
        self,
        initial_step_size: float = 1.0,
        contraction_factor: float = 0.5,
        c1: float = 1e-4,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        jit: bool = True,
    ):
        """Initialize backtracking line search.

        Args:
            initial_step_size: Initial step size to try.
            contraction_factor: Step size reduction factor (0 < rho < 1).
            c1: Armijo condition constant (0 < c1 < 1).
            max_iterations: Maximum number of iterations.
            tolerance: Tolerance for convergence.
            jit: Whether to JIT-compile the search method.

        Raises:
            ValueError: If contraction_factor or c1 are not in (0,1).
        """
        if not 0 < contraction_factor < 1:
            raise ValueError("contraction_factor must be in (0,1)")
        if not 0 < c1 < 1:
            raise ValueError("c1 must be in (0,1)")
            
        super().__init__(max_iterations=max_iterations, tolerance=tolerance)
        self.initial_step_size = initial_step_size
        self.contraction_factor = contraction_factor
        self.c1 = c1
        self._jit = jit

    def _search(
        self,
        f: Callable[[jnp.ndarray], float],
        x: jnp.ndarray,
        direction: jnp.ndarray,
        f_x: float,
        grad_x: Optional[jnp.ndarray] = None,
    ) -> LineSearchResults:
        """Perform backtracking line search.

        Args:
            f: Objective function.
            x: Current point.
            direction: Search direction.
            f_x: Function value at current point.
            grad_x: Gradient at current point.

        Returns:
            Line search results.
        """
        # Initialize step size
        step_size = self.initial_step_size
        
        # Compute gradient if not provided
        if grad_x is None:
            grad_x = jax.grad(f)(x)
        
        # Compute directional derivative
        d_f = jnp.dot(grad_x, direction)
        
        # Check if direction is a descent direction
        if d_f >= 0:
            return LineSearchResults(
                step_size=0.0,
                iterations=0,
                success=False,
                function_values=jnp.array([f_x]),
                step_sizes=jnp.array([0.0]),
                message="Not a descent direction"
            )
        
        # Initialize arrays to store function values and step sizes
        function_values = [f_x]
        step_sizes = [0.0]
        
        # Initialize iteration counter
        iterations = 0
        
        while iterations < self.max_iterations:
            # Compute function value at new point
            f_new = f(x + step_size * direction)
            function_values.append(f_new)
            step_sizes.append(step_size)
            
            # Check Armijo condition
            if f_new <= f_x + self.c1 * step_size * d_f:
                success = True
                break
            
            # Reduce step size
            step_size *= self.contraction_factor
            iterations += 1
        else:
            success = False
        
        # Create message
        if success:
            message = f"Converged after {iterations} iterations"
        else:
            message = f"Maximum iterations ({self.max_iterations}) reached"
        
        return LineSearchResults(
            step_size=step_size,
            iterations=iterations,
            success=success,
            function_values=jnp.array(function_values),
            step_sizes=jnp.array(step_sizes),
            message=message
        )