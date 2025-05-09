import jax
import jax.numpy as jnp
import pytest
from JaxOptimus.line_search.exact import GoldenSectionSearch, QuadraticLineSearch
from JaxOptimus.line_search.base import LineSearchResults

def test_golden_section_search_quadratic():
    """Test golden section search on a quadratic function."""
    # Define a simple quadratic function
    def f(x):
        return x[0]**2 + x[1]**2
    
    # Initial point and direction
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    
    # Create line search instance with JIT disabled
    line_search = GoldenSectionSearch(tolerance=1e-6, jit=False)
    
    # Perform line search
    results = line_search.search(f, x, direction)
    
    # Check results
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.iterations > 0
    assert results.step_size > 0
    assert len(results.function_values) > 0
    assert len(results.step_sizes) > 0
    
    # Check that function values are decreasing
    f_values = results.function_values
    assert jnp.all(jnp.diff(f_values) <= 0)

def test_golden_section_search_rosenbrock():
    """Test golden section search on the Rosenbrock function."""
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    # Initial point and direction
    x = jnp.array([-1.0, 2.0])
    direction = jnp.array([1.0, -1.0])
    
    # Create line search instance with JIT disabled
    line_search = GoldenSectionSearch(tolerance=1e-6, jit=False)
    
    # Perform line search
    results = line_search.search(f, x, direction)
    
    # Check results
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.iterations > 0
    assert results.step_size > 0

def test_golden_section_search_bracket():
    """Test golden section search with a provided bracket."""
    def f(x):
        return x[0]**2 + x[1]**2
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    bracket = (0.0, 2.0)
    
    # Create line search instance with JIT disabled
    line_search = GoldenSectionSearch(tolerance=1e-6, jit=False)
    results = line_search.search(f, x, direction, bracket=bracket)
    
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.step_size >= bracket[0]
    assert results.step_size <= bracket[1]

def test_quadratic_line_search():
    """Test quadratic line search on a quadratic function."""
    def f(x):
        return x[0]**2 + x[1]**2
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    
    # Compute gradient and Hessian
    grad = jax.grad(f)
    hessian = jax.hessian(f)
    
    # Create line search instance with JIT disabled
    line_search = QuadraticLineSearch(jit=False)
    results = line_search.search(f, x, direction, grad_x=grad(x), hessian=hessian(x))
    
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.iterations == 1  # Quadratic search should take only one iteration
    assert results.step_size > 0

def test_quadratic_line_search_fallback():
    """Test quadratic line search with fallback to backtracking."""
    def f(x):
        return x[0]**2 + x[1]**2
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    
    # Create line search instance with JIT disabled
    line_search = QuadraticLineSearch(jit=False)
    results = line_search.search(f, x, direction)
    
    assert isinstance(results, LineSearchResults)
    assert results.success
    assert results.step_size > 0

def test_quadratic_line_search_negative_curvature():
    """Test quadratic line search with negative curvature."""
    def f(x):
        return -x[0]**2 - x[1]**2  # Negative definite quadratic
    
    x = jnp.array([1.0, 1.0])
    direction = jnp.array([-1.0, -1.0])
    
    grad = jax.grad(f)
    hessian = jax.hessian(f)
    
    # Create line search instance with JIT disabled
    line_search = QuadraticLineSearch(jit=False)
    results = line_search.search(f, x, direction, grad_x=grad(x), hessian=hessian(x))
    
    assert isinstance(results, LineSearchResults)
    assert not results.success  # Should fail due to negative curvature
    assert results.step_size > 0  # Should use fallback step size

def test_line_search_initialization():
    """Test line search initialization parameters."""
    # Test GoldenSectionSearch
    gs = GoldenSectionSearch(
        bracket_factor=3.0,
        max_bracket_iterations=15,
        tolerance=1e-8,
        max_iterations=100,
        jit=False
    )
    assert gs.bracket_factor == 3.0
    assert gs.max_bracket_iterations == 15
    assert gs.tolerance == 1e-8
    assert gs.max_iterations == 100
    assert not gs._jit
    
    # Test QuadraticLineSearch
    qs = QuadraticLineSearch(jit=False)
    assert qs.max_iterations == 1
    assert qs.tolerance == 0.0
    assert not qs._jit
