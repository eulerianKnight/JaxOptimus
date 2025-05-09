import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, Callable, Union, List
from .base import Function

class QuadraticFunction(Function):
    """Quadratic function of the form f(x) = 0.5 * x^T * A * x - b^T * x + c."""
    
    def __init__(
        self,
        A: jnp.ndarray,
        b: Optional[jnp.ndarray] = None,
        c: float = 0.0,
        name: str = "Quadratic",
    ):
        """Initialize the quadratic function.
        
        Args:
            A: Positive definite matrix.
            b: Linear term. If None, uses zeros.
            c: Constant term.
            name: Name of the function.
        """
        # Check if A is symmetric
        if not jnp.allclose(A, A.T):
            raise ValueError("Matrix A must be symmetric")
        
        # Check if A is positive definite
        eigvals = jnp.linalg.eigvalsh(A)
        if jnp.any(eigvals <= 0):
            raise ValueError("Matrix A must be positive definite")
        
        self.A = A
        self.b = b if b is not None else jnp.zeros(A.shape[0])
        self.c = c
        
        # Compute global minimum
        x_min = jnp.linalg.solve(A, self.b)
        f_min = self._evaluate(x_min)
        
        super().__init__(
            dim=A.shape[0],
            name=name,
            global_minimum=(x_min, f_min),
            domain=None  # No specific domain constraints
        )
    
    def _evaluate(self, x: jnp.ndarray) -> float:
        """Evaluate the quadratic function.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Function value at x.
        """
        return 0.5 * jnp.dot(x, jnp.dot(self.A, x)) - jnp.dot(self.b, x) + self.c
    
    def _gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient of the quadratic function.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Gradient at x.
        """
        return jnp.dot(self.A, x) - self.b
    
    def _hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the Hessian of the quadratic function.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Hessian at x.
        """
        return self.A
    
    def condition_number(self, x: Optional[jnp.ndarray] = None) -> float:
        """Compute the condition number of the quadratic function.
        
        For a quadratic function, the condition number is the ratio of the
        largest to smallest eigenvalue of A.
        
        Args:
            x: Unused, included for API compatibility.
            
        Returns:
            Condition number.
        """
        eigvals = jnp.linalg.eigvalsh(self.A)
        return jnp.abs(eigvals.max() / eigvals.min())

class IllConditionedQuadratic(QuadraticFunction):
    """Ill-conditioned quadratic function for testing."""
    
    def __init__(
        self,
        condition_number: float = 100.0,
        dim: int = 2,
        name: str = "Ill-Conditioned Quadratic",
    ):
        """Initialize an ill-conditioned quadratic function.
        
        Args:
            condition_number: Desired condition number.
            dim: Dimensionality of the function.
            name: Name of the function.
        """
        # Create a diagonal matrix with eigenvalues from 1 to condition_number
        if dim > 1:
            eigenvalues = jnp.linspace(1.0, condition_number, dim)
        else:
            eigenvalues = jnp.array([condition_number])
            
        # Create the matrix A = Q^T * D * Q where D is diagonal with our eigenvalues
        # and Q is a random orthogonal matrix
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (dim, dim))
        Q, _ = jnp.linalg.qr(M)  # Q is orthogonal
        A = jnp.dot(Q.T, jnp.dot(jnp.diag(eigenvalues), Q))
        
        # Create a random b vector
        b = jax.random.normal(key, (dim,))
        
        super().__init__(A, b, 0.0, name)

class RosenbrockFunction(Function):
    """Rosenbrock function."""
    
    def __init__(
        self,
        a: float = 1.0,
        b: float = 100.0,
        name: str = "Rosenbrock",
    ):
        """Initialize the Rosenbrock function.
        
        Args:
            a: Parameter a in the Rosenbrock function.
            b: Parameter b in the Rosenbrock function.
            name: Name of the function.
        """
        self.a = a
        self.b = b
        
        super().__init__(
            dim=2,
            name=name,
            global_minimum=(jnp.array([a, a**2]), 0.0),
            domain=(jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0]))
        )
    
    def _evaluate(self, x: jnp.ndarray) -> float:
        """Evaluate the Rosenbrock function.
        
        f(x,y) = (a - x)^2 + b * (y - x^2)^2
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Function value at x.
        """
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2
    
    def _gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient of the Rosenbrock function.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Gradient at x.
        """
        dx = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0]**2)
        dy = 2 * self.b * (x[1] - x[0]**2)
        return jnp.array([dx, dy])
    
    def _hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the Hessian of the Rosenbrock function.
        
        Args:
            x: Point to evaluate.
            
        Returns:
            Hessian at x.
        """
        H = jnp.zeros((2, 2))
        H = H.at[0, 0].set(2 + 4 * self.b * (2 * x[0]**2 - x[1]))
        H = H.at[0, 1].set(-4 * self.b * x[0])
        H = H.at[1, 0].set(-4 * self.b * x[0])
        H = H.at[1, 1].set(2 * self.b)
        return H
    
    def get_visualization_bounds(self, dimension_indices: Tuple[int, int] = (0, 1)) -> Dict[str, Any]:
        """Get bounds for visualization.
        
        Args:
            dimension_indices: Indices of dimensions to visualize.
            
        Returns:
            Dictionary with visualization parameters.
        """
        return {
            "x_range": (-2.0, 2.0),
            "y_range": (-1.0, 3.0),
            "levels": 50,
            "use_log_scale": True
        }