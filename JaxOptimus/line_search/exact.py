import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict, Any, Optional, List, Union
from .base import LineSearch, LineSearchResults
from .backtracking import BacktrackingLineSearch

class GoldenSectionSearch(LineSearch):
    """Exact line search using the golden section method."""
    
    def __init__(
        self,
        bracket_factor: float = 2.0,
        max_bracket_iterations: int = 10,
        tolerance: float = 1e-6,
        max_iterations: int = 50,
        jit: bool = True,
    ):
        """Initialize golden section search.
        
        Args:
            bracket_factor: Factor to expand bracket by when finding initial interval.
            max_bracket_iterations: Maximum number of iterations for bracket finding.
            tolerance: Tolerance for convergence.
            max_iterations: Maximum number of iterations.
            jit: Whether to JIT-compile the search method.
        """
        super().__init__(max_iterations, tolerance)
        self.bracket_factor = bracket_factor
        self.max_bracket_iterations = max_bracket_iterations
        self._jit = jit  # Use _jit to match test expectations
    
    def _bracket_minimum(
        self,
        f_line: Callable[[float], float],
        initial_step: float = 1.0,
    ) -> Tuple[float, float]:
        """Find a bracket [a, b] containing a minimum of f_line.
        
        Args:
            f_line: 1D function to minimize.
            initial_step: Initial step size.
            
        Returns:
            Tuple of (a, b) representing the bracket.
        """
        # Start with [0, initial_step]
        a = 0.0
        b = initial_step
        fa = f_line(a)
        fb = f_line(b)
        
        # If f(b) > f(a), swap points to ensure we have a proper descent direction
        if not self._jit and fb > fa:
            a, b = b, a
            fa, fb = fb, fa
        
        # Expand the bracket until we find a point c with f(c) > f(b)
        c = b + self.bracket_factor * (b - a)
        fc = f_line(c)
        
        # Initialize iteration counter
        i = 0
        
        while i < self.max_bracket_iterations and fc < fb:
            # Update bracket: a <- b, b <- c, c <- new point
            a, b = b, c
            fa, fb = fb, fc
            c = b + self.bracket_factor * (b - a)
            fc = f_line(c)
            i += 1
        
        # Ensure a < b
        a, b = jnp.minimum(a, c), jnp.maximum(a, c)
        
        return a, b
    
    def _search(
        self,
        f: Callable[[jnp.ndarray], float],
        x: jnp.ndarray,
        direction: jnp.ndarray,
        f_x: float,
        grad_x: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> LineSearchResults:
        """Golden section search implementation.
        
        Args:
            f: Objective function taking a vector input and returning a scalar.
            x: Current point.
            direction: Search direction.
            f_x: Function value at x.
            grad_x: Gradient at x (optional).
            **kwargs: Additional parameters including bracket if provided.
            
        Returns:
            LineSearchResults containing the step size and metadata.
        """
        # Define the 1D function along the search direction
        def f_line(alpha):
            return f(x + alpha * direction)
        
        # Find bracket if not provided
        bracket = kwargs.get('bracket')
        if bracket is None:
            a, b = self._bracket_minimum(f_line)
        else:
            a, b = bracket
        
        # Golden ratio
        gr = (jnp.sqrt(5) - 1) / 2  # ≈ 0.618
        
        # Initial points
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = f_line(c), f_line(d)
        
        # Initialize arrays to store function values and step sizes
        function_values = [f_x]
        step_sizes = [0.0]
        
        # Initialize iteration counter
        iterations = 0
        
        while iterations < self.max_iterations and (b - a) > self.tolerance:
            if fc < fd:
                b = d
                d = c
                c = b - gr * (b - a)
                fd = fc
                fc = f_line(c)
            else:
                a = c
                c = d
                d = a + gr * (b - a)
                fc = fd
                fd = f_line(d)
            
            function_values.append(min(fc, fd))
            step_sizes.append(c if fc < fd else d)
            iterations += 1
        
        # Select the minimum point as the final step size
        alpha = c if fc < fd else d
        min_value = min(fc, fd)
        
        # Success if we converged within tolerance
        success = (b - a) <= self.tolerance
        
        # Create message without using jnp.where for strings
        if success:
            message = f"Converged after {iterations} iterations, bracket width: {b-a:.6e}"
        else:
            message = f"Maximum iterations reached, bracket width: {b-a:.6e}"
        
        return LineSearchResults(
            step_size=alpha,
            iterations=iterations,
            success=success,
            function_values=jnp.array(function_values),
            step_sizes=jnp.array(step_sizes),
            message=message
        )

class QuadraticLineSearch(LineSearch):
    """Line search for quadratic functions using the analytical solution."""
    
    def __init__(
        self,
        jit: bool = True,
    ):
        """Initialize quadratic line search.
        
        Args:
            jit: Whether to JIT-compile the search method.
        """
        super().__init__(1, 0.0)  # Only needs 1 iteration
        self._jit = jit  # Use _jit to match test expectations
        self._fallback = BacktrackingLineSearch()  # Initialize without jit parameter
    
    def _search(
        self,
        f: Callable[[jnp.ndarray], float],
        x: jnp.ndarray,
        direction: jnp.ndarray,
        f_x: float,
        grad_x: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> LineSearchResults:
        """Exact line search for quadratic functions.
        
        Args:
            f: Objective function taking a vector input and returning a scalar.
            x: Current point.
            direction: Search direction.
            f_x: Function value at x.
            grad_x: Gradient at x (optional).
            **kwargs: Additional parameters including hessian if provided.
            
        Returns:
            LineSearchResults containing the step size and metadata.
        """
        # Extract hessian from kwargs if provided
        hessian = kwargs.get('hessian')
        
        if grad_x is None:
            grad_x = jax.grad(f)(x)
        
        if hessian is None:
            # Use fallback method if Hessian not provided
            return self._fallback._search(f, x, direction, f_x, grad_x)
        
        # Compute the optimal step size: alpha = -g^T d / (d^T H d)
        numerator = -jnp.dot(grad_x, direction)
        denominator = jnp.dot(direction, jnp.dot(hessian, direction))
        
        # Safeguard against division by zero or negative curvature
        eps = 1e-10
        is_valid = (denominator > eps) & (numerator >= 0)
        
        # Compute step size, with fallback to a small positive step if invalid
        alpha = jnp.where(is_valid, numerator / denominator, 1e-4)
        
        # Compute function value at new point for returning
        f_new = f(x + alpha * direction)
        
        # Create message without using jnp.where for strings
        if not self._jit:
            message = "Quadratic line search successful" if is_valid else "Warning: Negative curvature detected, using fallback step size"
        else:
            message = "Quadratic line search completed"
        
        return LineSearchResults(
            step_size=alpha,
            iterations=1,
            success=is_valid,
            function_values=jnp.array([f_x, f_new]),
            step_sizes=jnp.array([0.0, alpha]),
            message=message
        )