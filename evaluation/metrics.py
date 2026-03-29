"""
Evaluation Metrics for LLM-JEPA Symbolic Regression.

Pure metric functions — no model dependency, no side effects.
Used by evaluate.py for the comprehensive evaluation suite.

Metrics implemented:
    - Node count (formula complexity)
    - Accuracy to tolerance (Acc_τ)
    - R² score
    - BFGS constant fitting
    - Exact symbolic equivalence check
"""

from __future__ import annotations
import numpy as np
import sympy
from scipy.optimize import minimize
from typing import List, Optional, Tuple
import warnings


# ── Formula Complexity ────────────────────────────────────────────────────────

def calculate_node_count(expr: sympy.Expr) -> int:
    """
    Count the number of nodes in a SymPy expression tree.

    Leaves (symbols, numbers) count as 1.
    Operators count as 1 + sum of children.

    Examples:
        x + y         → 3 nodes (Add, x, y)
        sin(x)        → 2 nodes (sin, x)
        x * y + z     → 5 nodes (Add, Mul, x, y, z)
    """
    if expr.is_Atom:
        return 1
    return 1 + sum(calculate_node_count(arg) for arg in expr.args)


# ── Accuracy to Tolerance ─────────────────────────────────────────────────────

def calculate_acc_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
) -> float:
    """
    Fraction of predictions where max relative error < τ.

    Args:
        y_true: ground truth values [N]
        y_pred: predicted values [N]
        tau:    tolerance threshold (e.g. 0.01 = 1%)

    Returns:
        Fraction in [0.0, 1.0]. Returns 0.0 if inputs are empty.
    """
    if len(y_true) == 0:
        return 0.0

    # Avoid division by zero: use max(|y_true|, eps) as denominator
    eps = 1e-12
    denom = np.maximum(np.abs(y_true), eps)
    rel_error = np.abs(y_true - y_pred) / denom
    max_rel_error = np.max(rel_error)
    return 1.0 if max_rel_error < tau else 0.0


# ── R² Score ──────────────────────────────────────────────────────────────────

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Returns -inf if prediction is degenerate.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf
    return 1.0 - ss_res / ss_tot


# ── BFGS Constant Fitting ────────────────────────────────────────────────────

def fit_constants(
    tokens: List[str],
    var_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 3,
) -> Tuple[Optional[dict], float]:
    """
    Fit constant placeholders (c1..c5) in an RPN token sequence
    via BFGS to minimise MSE against data.

    Args:
        tokens:     RPN token list (may contain c1..c5)
        var_names:  original variable names (maps x1→var_names[0], etc.)
        X:          input data [N, n_vars]
        y:          target values [N]
        n_restarts: number of random restarts for BFGS

    Returns:
        (constants_dict, best_r2)
        constants_dict maps e.g. {'c1': 3.14, 'c2': 2.0}
        best_r2 is the R² achieved with the best constants
    """
    from data.tokenizer import rpn_to_sympy as _rpn_to_sympy

    # Find which constant placeholders are used
    const_names = sorted(set(t for t in tokens if t.startswith('c') and t[1:].isdigit()))

    if not const_names:
        # No constants to fit — evaluate directly
        try:
            expr = _rpn_to_sympy(tokens)
            y_pred = _evaluate_expr(expr, var_names, X)
            r2 = calculate_r2(y, y_pred)
            return {}, r2
        except Exception:
            return {}, -np.inf

    # Build SymPy expression with constant symbols
    try:
        expr = _rpn_to_sympy(tokens)
    except Exception:
        return None, -np.inf

    const_symbols = [sympy.Symbol(c) for c in const_names]

    # Build fast numerical evaluator
    all_symbols = [sympy.Symbol(f'x{i+1}') for i in range(len(var_names))] + const_symbols

    try:
        f_numpy = sympy.lambdify(all_symbols, expr, modules='numpy')
    except Exception:
        return None, -np.inf

    def objective(const_vals):
        """MSE objective for BFGS."""
        try:
            args = list(X.T) + list(const_vals)
            y_pred = np.asarray(f_numpy(*args), dtype=np.float64)
            if not np.all(np.isfinite(y_pred)):
                return 1e12
            return float(np.mean((y - y_pred) ** 2))
        except Exception:
            return 1e12

    # Multi-restart BFGS
    best_r2 = -np.inf
    best_constants = None
    n_consts = len(const_names)

    for restart in range(n_restarts):
        x0 = np.random.randn(n_consts) * (1.0 if restart == 0 else 5.0)
        if restart == 0:
            x0 = np.ones(n_consts)  # Start with 1s first

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-12})
            except Exception:
                continue

        try:
            args = list(X.T) + list(result.x)
            y_pred = np.asarray(f_numpy(*args), dtype=np.float64)
            if not np.all(np.isfinite(y_pred)):
                continue
            r2 = calculate_r2(y, y_pred)
        except Exception:
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_constants = dict(zip(const_names, result.x))

    return best_constants, best_r2


def fit_constants_with_scipy(
    expr: sympy.Expr, 
    X_data: np.ndarray, 
    y_data: np.ndarray, 
    var_names: List[str]
) -> Tuple[float, sympy.Expr]:
    """
    Takes a sympy skeleton, finds the best constants using BFGS, 
    and returns the Mean Squared Error and the finalized equation.
    """
    # 1. Extract constants (c1, c2, etc.) and variables (x1, x2, etc.)
    free_symbols = list(expr.free_symbols)
    constants = [sym for sym in free_symbols if str(sym).startswith('c')]
    
    # If the model didn't predict any constants, just evaluate the raw error
    if not constants:
        try:
            func = sympy.lambdify([sympy.Symbol(v) for v in var_names], expr, modules=['numpy'])
            var_inputs = [X_data[:, i] for i in range(len(var_names))]
            y_pred = func(*var_inputs)
            mse = np.nanmean((y_data - y_pred)**2)
            return mse, expr
        except Exception:
            return float('inf'), expr

    # 2. Convert sympy expression to a fast numpy function
    args = constants + [sympy.Symbol(v) for v in var_names]
    try:
        func = sympy.lambdify(args, expr, modules=['numpy'])
    except Exception:
        return float('inf'), expr # Failsafe for mathematically invalid structures

    # 3. Define the objective function for SciPy to minimize
    def objective(c_values):
        try:
            # Unpack the columns of X_data into separate variable arrays
            var_inputs = [X_data[:, i] for i in range(len(var_names))]
            
            # Predict y using the current guesses for constants
            y_pred = func(*c_values, *var_inputs)
            
            # Catch physics domain errors (e.g., square roots of negative numbers)
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                return 1e10 # Massive penalty so SciPy turns around
                
            mse = np.nanmean((y_data - y_pred)**2)
            return mse
        except Exception:
            return 1e10

    # 4. Run the BFGS Optimizer
    # Start by guessing 1.0 for every constant
    initial_guess = np.ones(len(constants))
    
    try:
        res = minimize(objective, initial_guess, method='BFGS')
        best_c = res.x
        best_mse = res.fun
    except Exception:
        # If the optimizer completely crashes, return infinity error
        return float('inf'), expr

    # 5. Substitute the optimized decimals back into the SymPy equation
    optimized_expr = expr.subs({c: val for c, val in zip(constants, best_c)})
    
    return best_mse, optimized_expr


# ── Exact Symbolic Equivalence ────────────────────────────────────────────────

def verify_exact(
    predicted_str: str,
    ground_truth_str: str,
    var_names: List[str],
) -> bool:
    """
    Check if predicted formula is algebraically equivalent to ground truth.

    Uses SymPy simplify(pred - truth) == 0.
    Falls back to numerical comparison if simplification times out.
    """
    try:
        local_dict = {name: sympy.Symbol(name) for name in var_names}
        local_dict['pi'] = sympy.pi
        local_dict['e'] = sympy.E

        pred_expr = sympy.sympify(predicted_str, locals=local_dict)
        true_expr = sympy.sympify(ground_truth_str, locals=local_dict)

        diff = sympy.simplify(pred_expr - true_expr)
        if diff == 0:
            return True

        # Numerical fallback: evaluate at random points
        symbols = [sympy.Symbol(name) for name in var_names]
        for _ in range(10):
            point = {s: float(np.random.uniform(1, 5)) for s in symbols}
            try:
                v_pred = complex(pred_expr.subs(point))
                v_true = complex(true_expr.subs(point))
                if abs(v_pred - v_true) > 1e-6 * max(abs(v_true), 1e-12):
                    return False
            except Exception:
                return False
        return True

    except Exception:
        return False


# ── Evaluate Expression ───────────────────────────────────────────────────────

def _evaluate_expr(
    expr: sympy.Expr,
    var_names: List[str],
    X: np.ndarray,
) -> np.ndarray:
    """
    Numerically evaluate a SymPy expression on data.

    Args:
        expr:      SymPy expression using x1..xN symbols
        var_names: original variable names (for mapping)
        X:         input data [N, n_vars]

    Returns:
        y_pred [N] array
    """
    symbols = [sympy.Symbol(f'x{i+1}') for i in range(len(var_names))]
    try:
        f_numpy = sympy.lambdify(symbols, expr, modules='numpy')
        y_pred = np.asarray(f_numpy(*X.T), dtype=np.float64)
        return y_pred
    except Exception:
        return np.full(X.shape[0], np.nan)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test node count
    x, y_sym = sympy.symbols('x y')
    assert calculate_node_count(x) == 1
    assert calculate_node_count(x + y_sym) == 3  # Add, x, y
    assert calculate_node_count(sympy.sin(x)) == 2  # sin, x
    print('Node count: OK')

    # Test accuracy to tolerance
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.001, 2.002, 3.003])
    assert calculate_acc_tau(y_true, y_pred, tau=0.01) == 1.0
    assert calculate_acc_tau(y_true, y_pred, tau=0.0001) == 0.0
    print('Accuracy to tolerance: OK')

    # Test R²
    assert calculate_r2(y_true, y_true) == 1.0
    assert calculate_r2(y_true, y_pred) > 0.99
    print('R² score: OK')

    # Test verify_exact
    assert verify_exact('x + y', 'y + x', ['x', 'y']) == True
    assert verify_exact('x + y', 'x * y', ['x', 'y']) == False
    print('Exact verification: OK')

    # Test fit_constants_with_scipy
    c1, c2, x1 = sympy.symbols('c1 c2 x1')
    expr_test = c1 * sympy.sin(c2 * x1)
    X_test = np.linspace(0, 5, 100).reshape(-1, 1).astype(np.float32)
    y_test = 2.5 * np.sin(1.2 * X_test.flatten()).astype(np.float32)
    
    mse, opt_expr = fit_constants_with_scipy(expr_test, X_test, y_test, ['x1'])
    assert mse < 1e-5
    # Check if constants are roughly correct (2.5 and 1.2)
    const_vals = [float(val) for val in opt_expr.atoms(sympy.Float)]
    assert any(abs(v - 2.5) < 0.1 for v in const_vals)
    assert any(abs(v - 1.2) < 0.1 for v in const_vals)
    print('fit_constants_with_scipy: OK')

    print('\nAll metrics tests passed.')
