import numpy as np
import pytest
from evaluation.metrics import fit_constants, calculate_r2

def test_fit_constants_linear():
    """Test BFGS fitting for a simple linear function: y = 2.5 * x1 + 1.2 * x2"""
    var_names = ['x1', 'x2']
    X = np.random.uniform(1, 5, (100, 2)).astype(np.float32)
    y = 2.5 * X[:, 0] + 1.2 * X[:, 1]
    
    # RPN: x1 c1 * x2 c2 * +
    tokens = ['x1', 'c1', '*', 'x2', 'c2', '*', '+']
    
    constants, r2 = fit_constants(tokens, var_names, X, y, n_restarts=3)
    
    assert constants is not None
    assert 'c1' in constants
    assert 'c2' in constants
    assert np.isclose(constants['c1'], 2.5, atol=1e-2)
    assert np.isclose(constants['c2'], 1.2, atol=1e-2)
    assert r2 > 0.999

def test_fit_constants_nonlinear():
    """Test BFGS fitting for: y = 3.0 * sin(c1 * x1) + c2"""
    var_names = ['x1']
    X = np.random.uniform(0, 2, (200, 1)).astype(np.float32)
    y = 3.0 * np.sin(0.5 * X[:, 0]) + 1.5
    
    # RPN: x1 c1 * sin 3.0 * c2 +
    # Note: 3.0 is a literal, c1 and c2 are placeholders
    tokens = ['x1', 'c1', '*', 'sin', '3.0', '*', 'c2', '+']
    
    constants, r2 = fit_constants(tokens, var_names, X, y, n_restarts=5)
    
    assert constants is not None
    assert 'c1' in constants
    assert 'c2' in constants
    # BFGS might find 0.5 or -0.5 depending on symmetry if not constrained, 
    # but here sin is odd, so it should be close to 0.5 (or -0.5 with sign flip in 3.0 if it were a constant)
    assert np.isclose(abs(constants['c1']), 0.5, atol=1e-2)
    assert np.isclose(constants['c2'], 1.5, atol=1e-2)
    assert r2 > 0.99

def test_no_constants():
    """Test fit_constants with no constants in the tokens."""
    var_names = ['x1']
    X = np.random.uniform(1, 5, (50, 1)).astype(np.float32)
    y = X[:, 0] ** 2
    
    # RPN: x1 sq
    tokens = ['x1', 'sq']
    
    constants, r2 = fit_constants(tokens, var_names, X, y)
    
    assert constants == {}
    assert r2 > 0.999

def test_invalid_rpn():
    """Test fit_constants with an invalid RPN sequence."""
    var_names = ['x1']
    X = np.random.uniform(1, 5, (10, 1)).astype(np.float32)
    y = X[:, 0]
    
    # RPN: x1 + (missing operand)
    tokens = ['x1', '+']
    
    constants, r2 = fit_constants(tokens, var_names, X, y)
    
    # If no constants found, it enters the 'if not const_names' block
    # and returns {} on error.
    assert constants == {}
    assert r2 == -np.inf

if __name__ == "__main__":
    pytest.main([__file__])
