import jax
import jax.numpy as jnp
import pytest
from JaxOptimus.line_search.backtracking import BacktrackingLineSearch
from JaxOptimus.line_search.base import LineSearchResults

def test_backtracking_line_search_quadratic():
    """Test backtracking line search on a simple quadratic function."""
    def f(x):
        return x[0]**2 + x[1]**2
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    
    # Create line search instance with JIT disabled for testing
    line_search = BacktrackingLineSearch(
        initial_step_size=1.0,
        contraction_factor=0.5,
        c1=1e-4,
        max_iterations=20,
        jit=False
    )
    
    results = line_search.search(f, x, direction)
    
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.step_size > 0
    assert results.step_size <= 1.0  # Should not exceed initial step size
    assert len(results.function_values) == len(results.step_sizes)
    assert results.function_values[0] == f(x)  # First value should be f(x)

def test_backtracking_line_search_non_descent_direction():
    """Test backtracking line search with a non-descent direction."""
    def f(x):
        return x[0]**2 + x[1]**2
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([1.0, 1.0])  # Non-descent direction
    
    line_search = BacktrackingLineSearch(jit=False)
    results = line_search.search(f, x, direction)
    
    assert isinstance(results, LineSearchResults)
    assert not results.success
    assert results.step_size == 0.0
    assert "not a descent direction" in results.message.lower()

def test_backtracking_line_search_max_iterations():
    """Test backtracking line search when max iterations is reached."""
    def f(x):
        # Use a highly nonlinear function with multiple local minima
        return 100 * jnp.exp(x[0]) + jnp.sin(10 * x[0]) + 100 * jnp.exp(x[1]) + jnp.sin(10 * x[1]) + \
               10 * jnp.cos(x[0] * x[1]) + (x[0]**2 + x[1]**2) / 2
    
    x = jnp.array([3.0, 3.0])  # Start at a more challenging point
    direction = jnp.array([-0.5, -1.0])  # Asymmetric direction
    
    # Create line search with very small max iterations and very strict conditions
    line_search = BacktrackingLineSearch(
        max_iterations=2,
        c1=0.9,  # Extremely strict Armijo condition
        contraction_factor=0.95,  # Very slow reduction in step size
        initial_step_size=5.0,  # Much larger initial step
        jit=False
    )
    results = line_search.search(f, x, direction)
    
    assert isinstance(results, LineSearchResults)
    assert not results.success
    assert results.iterations == 2
    assert "maximum iterations" in results.message.lower()

def test_backtracking_line_search_rosenbrock():
    """Test backtracking line search on the Rosenbrock function."""
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    x = jnp.array([-1.0, 2.0])
    direction = jnp.array([-1.0, -1.0])
    
    line_search = BacktrackingLineSearch(jit=False)
    results = line_search.search(f, x, direction)
    
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.step_size > 0
    assert results.function_values[-1] < results.function_values[0]  # Function value should decrease

def test_backtracking_line_search_parameters():
    """Test backtracking line search with different parameters."""
    def f(x):
        # Use a simpler function that still requires backtracking
        return 10 * (x[0]**2 + x[1]**2) + jnp.exp(x[0] + x[1])
    
    x = jnp.array([1.0, 1.0])  # Start at a simpler point
    direction = jnp.array([-1.0, -1.0])  # Symmetric direction
    
    # Test with different contraction factors
    line_search1 = BacktrackingLineSearch(
        contraction_factor=0.1,  # Very aggressive reduction
        c1=0.5,  # Moderate condition
        initial_step_size=2.0,  # Moderate initial step
        max_iterations=20,  # Default iterations
        jit=False
    )
    line_search2 = BacktrackingLineSearch(
        contraction_factor=0.9,  # Very slow reduction
        c1=0.5,  # Moderate condition
        initial_step_size=2.0,  # Moderate initial step
        max_iterations=20,  # Default iterations
        jit=False
    )
    
    results1 = line_search1.search(f, x, direction)
    results2 = line_search2.search(f, x, direction)
    
    assert results1.success and results2.success
    assert results1.step_size != results2.step_size  # Different contraction factors should give different results
    
    # Test with different c1 values
    line_search3 = BacktrackingLineSearch(
        c1=0.001,  # Very lenient condition
        contraction_factor=0.5,
        initial_step_size=2.0,  # Moderate initial step
        max_iterations=20,  # Default iterations
        jit=False
    )
    line_search4 = BacktrackingLineSearch(
        c1=0.9,  # Very strict condition
        contraction_factor=0.5,
        initial_step_size=2.0,  # Moderate initial step
        max_iterations=20,  # Default iterations
        jit=False
    )
    
    results3 = line_search3.search(f, x, direction)
    results4 = line_search4.search(f, x, direction)
    
    assert results3.success and results4.success
    assert results3.step_size != results4.step_size  # Different c1 values should give different results

def test_backtracking_line_search_invalid_parameters():
    """Test backtracking line search with invalid parameters."""
    with pytest.raises(ValueError):
        BacktrackingLineSearch(contraction_factor=1.0)  # Should be in (0,1)
    
    with pytest.raises(ValueError):
        BacktrackingLineSearch(contraction_factor=0.0)  # Should be in (0,1)
    
    with pytest.raises(ValueError):
        BacktrackingLineSearch(c1=1.0)  # Should be in (0,1)
    
    with pytest.raises(ValueError):
        BacktrackingLineSearch(c1=0.0)  # Should be in (0,1)

def test_backtracking_line_search_jit():
    """Test backtracking line search with JIT enabled."""
    def f(x):
        return x[0]**2 + x[1]**2
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    
    # Create line search with JIT enabled
    line_search = BacktrackingLineSearch(jit=True)
    
    # First compile the function and gradient
    f_jit = jax.jit(f)
    grad_f = jax.jit(jax.grad(f))
    
    # Pre-compute gradient for JIT
    grad_x = grad_f(x)
    f_x = f_jit(x)
    
    # Now search with the compiled function and pre-computed values
    results = line_search.search(f_jit, x, direction, f_x=f_x, grad_x=grad_x)
    
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.step_size > 0
