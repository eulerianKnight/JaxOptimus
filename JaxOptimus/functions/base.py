import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Callable, Union, List

class Function:
    """Base class for objective functions.
    
    This class provides a common interface for optimization test functions,
    including function evaluation, gradient, Hessian, and visualization properties.
    """
    
    def __init__(
        self,
        dim: int,
        name: str = "Function",
        global_minimum: Optional[Tuple[jnp.ndarray, float]] = None,
        domain: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
        """Initialize the function.
        
        Args:
            dim: Dimensionality of the function.
            name: Name of the function.
            global_minimum: Tuple of (x_min, f_min) for the global minimum, if known.
            domain: Tuple of (lower_bound, upper_bound) for the domain, if known.
        """
        self.dim = dim
        self.name = name
        self.global_minimum = global_minimum
        self.domain = domain
        
        # JIT-compile the function and gradient
        self._jitted_f = jax.jit(self._evaluate)
        self._jitted_grad = jax.jit(self._gradient)
        self._jitted_hessian = jax.jit(self._hessian) if hasattr(self, '_hessian') else None
    
    @abstractmethod
    def _evaluate(self, x: jnp.ndarray) -> float:
        """Raw function evaluation.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Function value at x.
        """
        pass
    
    def __call__(self, x: jnp.ndarray) -> float:
        """Evaluate the function.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Function value at x.
        """
        return self._jitted_f(x)
    
    @abstractmethod
    def _gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Raw gradient evaluation.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Gradient at x.
        """
        pass
    
    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the gradient.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Gradient at x.
        """
        return self._jitted_grad(x)
    
    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Hessian.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Hessian at x.
        """
        if self._jitted_hessian is None:
            raise NotImplementedError("Hessian not implemented for this function")
        return self._jitted_hessian(x)
    
    def condition_number(self, x: jnp.ndarray) -> float:
        """Compute the condition number of the Hessian at x.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Condition number at x.
        """
        H = self.hessian(x)
        eigvals = jnp.linalg.eigvalsh(H)
        return jnp.abs(eigvals.max() / eigvals.min())
    
    def get_visualization_bounds(self, dimension_indices: Tuple[int, int] = (0, 1)) -> Dict[str, Any]:
        """Get bounds for visualization.
        
        Args:
            dimension_indices: Indices of dimensions to visualize.
            
        Returns:
            Dictionary with visualization parameters.
        """
        if self.domain is not None:
            lower, upper = self.domain
            i, j = dimension_indices
            return {
                "x_range": (lower[i], upper[i]),
                "y_range": (lower[j], upper[j]),
                "levels": 50,
                "use_log_scale": False
            }
        
        # Default visualization bounds
        return {
            "x_range": (-5.0, 5.0),
            "y_range": (-5.0, 5.0),
            "levels": 50,
            "use_log_scale": False
        }
    
    def get_initial_points(self, n_points: int = 1) -> List[jnp.ndarray]:
        """Get suggested initial points for optimization.
        
        Args:
            n_points: Number of initial points to generate.
            
        Returns:
            List of initial points.
        """
        if n_points == 1 and self.global_minimum is not None:
            # Perturb the global minimum
            x_min = self.global_minimum[0]
            return [x_min + jnp.ones_like(x_min)]
        
        # Generate random points in the domain
        if self.domain is not None:
            lower, upper = self.domain
            key = jax.random.PRNGKey(0)
            return [lower + jax.random.uniform(key, shape=(self.dim,)) * (upper - lower)
                    for _ in range(n_points)]
        
        # Default: random points in [-5, 5]^dim
        key = jax.random.PRNGKey(0)
        return [jax.random.uniform(key, shape=(self.dim,), minval=-5.0, maxval=5.0)
                for _ in range(n_points)]