"""
Tests for evaluation/metrics.py

Tests NED, Symbolic Accuracy, Constant Recovery, and other metrics.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics import (
    calculate_node_count,
    calculate_acc_tau,
    calculate_r2,
    calculate_ned,
    verify_symbolic_accuracy,
    calculate_constant_recovery,
    fit_constants,
)
import sympy


class TestNodeCount:
    """Test formula complexity measurement."""
    
    def test_single_variable(self):
        """Test single variable node count."""
        x = sympy.Symbol('x')
        assert calculate_node_count(x) == 1
        
    def test_binary_operation(self):
        """Test binary operation node count."""
        x, y = sympy.symbols('x y')
        assert calculate_node_count(x + y) == 3  # Add, x, y
        
    def test_unary_operation(self):
        """Test unary operation node count."""
        x = sympy.Symbol('x')
        assert calculate_node_count(sympy.sin(x)) == 2  # sin, x
        
    def test_complex_expression(self):
        """Test complex expression node count."""
        x, y = sympy.symbols('x y')
        # x * y + z has 5 nodes: Add, Mul, x, y, z
        expr = x * y + sympy.Symbol('z')
        assert calculate_node_count(expr) == 5


class TestAccuracyTau:
    """Test accuracy to tolerance metric."""
    
    def test_perfect_accuracy(self):
        """Test perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_acc_tau(y_true, y_pred, tau=0.01) == 1.0
        
    def test_within_tolerance(self):
        """Test predictions within tolerance."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.001, 2.002, 3.003])
        assert calculate_acc_tau(y_true, y_pred, tau=0.01) == 1.0
        assert calculate_acc_tau(y_true, y_pred, tau=0.0001) == 0.0
        
    def test_empty_input(self):
        """Test empty input handling."""
        assert calculate_acc_tau(np.array([]), np.array([]), tau=0.01) == 0.0


class TestR2Score:
    """Test R² score calculation."""
    
    def test_perfect_fit(self):
        """Test perfect fit R² = 1."""
        y_true = np.array([1.0, 2.0, 3.0])
        assert calculate_r2(y_true, y_true) == 1.0
        
    def test_good_fit(self):
        """Test good fit R² > 0.99."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.01, 1.99, 3.01])
        assert calculate_r2(y_true, y_pred) > 0.99
        
    def test_constant_prediction(self):
        """Test constant prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # mean
        r2 = calculate_r2(y_true, y_pred)
        assert r2 >= 0.0  # Mean prediction should give R² = 0


class TestNED:
    """Test Normalized Edit Distance."""
    
    def test_identical_expressions(self):
        """Test identical expressions have NED = 0."""
        ned = calculate_ned('x1 + x2', 'x1 + x2')
        assert ned == 0.0
        
    def test_different_operators(self):
        """Test different operators have NED > 0."""
        ned = calculate_ned('x1 + x2', 'x1 * x2')
        assert ned > 0.0
        
    def test_commutative_expressions(self):
        """Test commutative expressions have low NED."""
        ned = calculate_ned('x1 + x2', 'x2 + x1')
        # Should be low but not zero due to tree structure difference
        assert ned < 0.5


class TestSymbolicAccuracy:
    """Test functional equivalence checking."""
    
    def test_commutative_addition(self):
        """Test commutative addition is equivalent."""
        sa = verify_symbolic_accuracy('x1 + x2', 'x2 + x1', ['x1', 'x2'])
        assert sa == True
        
    def test_different_expressions(self):
        """Test different expressions are not equivalent."""
        sa = verify_symbolic_accuracy('x1 + x2', 'x1 * x2', ['x1', 'x2'])
        assert sa == False
        
    def test_algebraic_equivalence(self):
        """Test algebraically equivalent expressions."""
        # (x+1)^2 = x^2 + 2x + 1
        sa = verify_symbolic_accuracy(
            '(x1 + 1)**2', 
            'x1**2 + 2*x1 + 1', 
            ['x1']
        )
        assert sa == True


class TestConstantRecovery:
    """Test constant recovery metric."""
    
    def test_exact_recovery(self):
        """Test exact constant recovery."""
        true_consts = [2.5, 3.0]
        pred_consts = [2.5, 3.0]
        cr = calculate_constant_recovery(pred_consts, true_consts)
        assert cr == 1.0
        
    def test_approximate_recovery(self):
        """Test approximate constant recovery."""
        true_consts = [2.5, 3.0]
        pred_consts = [2.49, 3.01]  # Within 1%
        cr = calculate_constant_recovery(pred_consts, true_consts)
        assert cr > 0.9
        
    def test_poor_recovery(self):
        """Test poor constant recovery."""
        true_consts = [2.5, 3.0]
        pred_consts = [10.0, 20.0]  # Way off
        cr = calculate_constant_recovery(pred_consts, true_consts)
        assert cr < 0.5
        
    def test_length_mismatch(self):
        """Test length mismatch handling."""
        true_consts = [2.5, 3.0]
        pred_consts = [2.5]  # Different length
        cr = calculate_constant_recovery(pred_consts, true_consts)
        assert cr == 0.0


class TestFitConstants:
    """Test BFGS constant fitting."""
    
    def test_linear_fit(self):
        """Test fitting linear function constants."""
        # y = c1*x + c2, true: c1=2.5, c2=3.0
        tokens = ['c1', 'x1', '*', 'c2', '+']
        X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float64)
        y = 2.5 * X.flatten() + 3.0
        
        constants, r2 = fit_constants(tokens, ['x1'], X, y, n_restarts=3)
        
        assert r2 > 0.99, f"Expected R² > 0.99, got {r2}"
        # Check constants are approximately correct
        assert abs(constants.get('c1', 0) - 2.5) < 0.1
        assert abs(constants.get('c2', 0) - 3.0) < 0.1
        
    def test_no_constants(self):
        """Test expression without constants."""
        tokens = ['x1', 'x2', '+']
        X = np.random.randn(100, 2).astype(np.float64)
        y = X[:, 0] + X[:, 1]
        
        constants, r2 = fit_constants(tokens, ['x1', 'x2'], X, y)
        
        assert r2 > 0.99
        assert constants == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
