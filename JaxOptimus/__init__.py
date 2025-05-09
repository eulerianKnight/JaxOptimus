"""JaxOptimus: A JAX-based optimization library."""

from JaxOptimus import optimizers
from JaxOptimus import line_search
from JaxOptimus import functions
from JaxOptimus import viz

# Import commonly used classes for convenience
from JaxOptimus.line_search.base import LineSearch, LineSearchResults
from JaxOptimus.optimizers.base import Optimizer, OptimizationState
from JaxOptimus.functions.base import Function
from JaxOptimus.viz.base import Visualizer

# Import visualization functions
from JaxOptimus.viz.base import (
    plot_contour,
    plot_convergence,
    plot_step_sizes
)

# Convenient import functions
from JaxOptimus.line_search.backtracking import BacktrackingLineSearch
from JaxOptimus.line_search.exact import GoldenSectionSearch, QuadraticLineSearch
from JaxOptimus.optimizers.gradient_descent import GradientDescent
from JaxOptimus.optimizers.newton import NewtonMethod
from JaxOptimus.optimizers.conjugate_gradient import ConjugateGradient
from JaxOptimus.functions.quadratic import QuadraticFunction, IllConditionedQuadratic, RosenbrockFunction

__version__ = "0.1.0"

__all__ = [
    "optimizers",
    "line_search",
    "Optimizer",
    "OptimizationState",
    "GradientDescent",
    "NewtonMethod",
    "ConjugateGradient",
    "plot_contour",
    "plot_convergence",
    "plot_step_sizes"
]

# Convenience functions to create optimizers
def create_gradient_descent(line_search_type='backtracking', **kwargs):
    """Create a gradient descent optimizer with the specified line search type.
    
    Args:
        line_search_type: Type of line search to use ('backtracking', 'exact', 'quadratic', or None).
        **kwargs: Additional arguments for the optimizer and line search.
        
    Returns:
        GradientDescent optimizer.
    """
    ls_kwargs = {k: v for k, v in kwargs.items() if k in ['initial_step_size', 'contraction_factor', 'c1', 'tolerance']}
    opt_kwargs = {k: v for k, v in kwargs.items() if k in ['max_iterations', 'tolerance', 'store_trajectory', 'jit']}
    
    if line_search_type == 'backtracking':
        line_search = BacktrackingLineSearch(**ls_kwargs)
    elif line_search_type == 'exact':
        line_search = GoldenSectionSearch(**ls_kwargs)
    elif line_search_type == 'quadratic':
        line_search = QuadraticLineSearch(**ls_kwargs)
    else:
        line_search = None
    
    return GradientDescent(line_search=line_search, **opt_kwargs)

def create_newton_method(line_search_type='backtracking', **kwargs):
    """Create a Newton's method optimizer with the specified line search type.
    
    Args:
        line_search_type: Type of line search to use ('backtracking', 'exact', 'quadratic', or None).
        **kwargs: Additional arguments for the optimizer and line search.
        
    Returns:
        NewtonMethod optimizer.
    """
    ls_kwargs = {k: v for k, v in kwargs.items() if k in ['initial_step_size', 'contraction_factor', 'c1', 'tolerance']}
    opt_kwargs = {k: v for k, v in kwargs.items() if k in ['regularization', 'max_iterations', 'tolerance', 'store_trajectory', 'jit']}
    
    if line_search_type == 'backtracking':
        line_search = BacktrackingLineSearch(**ls_kwargs)
    elif line_search_type == 'exact':
        line_search = GoldenSectionSearch(**ls_kwargs)
    elif line_search_type == 'quadratic':
        line_search = QuadraticLineSearch(**ls_kwargs)
    else:
        line_search = None
    
    return NewtonMethod(line_search=line_search, **opt_kwargs)

def create_conjugate_gradient(line_search_type='backtracking', **kwargs):
    """Create a conjugate gradient optimizer with the specified line search type.
    
    Args:
        line_search_type: Type of line search to use ('backtracking', 'exact', 'quadratic', or None).
        **kwargs: Additional arguments for the optimizer and line search.
        
    Returns:
        ConjugateGradient optimizer.
    """
    ls_kwargs = {k: v for k, v in kwargs.items() if k in ['initial_step_size', 'contraction_factor', 'c1', 'tolerance']}
    opt_kwargs = {k: v for k, v in kwargs.items() if k in ['restart_interval', 'max_iterations', 'tolerance', 'store_trajectory', 'jit']}
    
    if line_search_type == 'backtracking':
        line_search = BacktrackingLineSearch(**ls_kwargs)
    elif line_search_type == 'exact':
        line_search = GoldenSectionSearch(**ls_kwargs)
    elif line_search_type == 'quadratic':
        line_search = QuadraticLineSearch(**ls_kwargs)
    else:
        line_search = None
    
    return ConjugateGradient(line_search=line_search, **opt_kwargs)