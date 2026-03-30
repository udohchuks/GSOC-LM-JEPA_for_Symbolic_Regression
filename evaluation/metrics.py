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
from typing import List, Optional, Tuple, Any
from sklearn.metrics import r2_score
import difflib
import warnings

# Silence all mathematical runtime warnings to keep Colab output clean
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


# ── Normalized Edit Distance (NED) ───────────────────────────────────────────

def calculate_ned(pred_expr_str: str, true_expr_str: str) -> float:
    """
    Convert expressions to prefix notation trees and compute Normalized Edit Distance.
    
    Lower is better (0 = identical, 1 = completely different).
    Uses difflib.SequenceMatcher on the serialized prefix trees.
    """
    def expr_to_prefix(expr: sympy.Expr) -> List[str]:
        """Recursive prefix serialization of a SymPy expression tree."""
        if expr.is_Atom:
            return [str(expr)]
        # Use type name for operators to avoid string representation noise
        return [type(expr).__name__] + [
            node for arg in expr.args 
            for node in expr_to_prefix(arg)
        ]

    try:
        # Standardize symbols first
        pred = sympy.sympify(pred_expr_str)
        true = sympy.sympify(true_expr_str)
        
        pred_tree = expr_to_prefix(pred)
        true_tree = expr_to_prefix(true)
        
        # normalized ratio of shared elements
        ratio = difflib.SequenceMatcher(None, pred_tree, true_tree).ratio()
        return 1.0 - ratio
    except Exception:
        return 1.0


# ── Functional Symbolic Accuracy (Functional Equivalence) ───────────────────

def verify_symbolic_accuracy(
    pred_expr_str: str,
    true_expr_str: str,
    var_names: List[str],
    n_points: int = 1000,
    tol: float = 1e-5
) -> bool:
    """
    Check functional equivalence via random point evaluation.
    More robust than exact algebraic simplification for complex forms.
    
    Args:
        pred_expr_str: recovered expression string
        true_expr_str: ground truth expression string
        var_names:     list of variable names (x1, x2, ...)
        n_points:      number of random evaluation points
        tol:           absolute and relative tolerance
    """
    try:
        local_dict = {name: sympy.Symbol(name) for name in var_names}
        pred = sympy.sympify(pred_expr_str, locals=local_dict)
        true = sympy.sympify(true_expr_str, locals=local_dict)
        
        # 1. Algebraic shortcut
        try:
            if sympy.simplify(pred - true) == 0:
                return True
        except: pass
        
        # 2. Numerical equivalence check
        # Evaluate on a standardized domain (0.1 to 5.0) to avoid common zeros/poles
        X_rand = np.random.uniform(0.1, 5.0, (n_points, len(var_names)))
        
        f_pred = sympy.lambdify([sympy.Symbol(v) for v in var_names], pred, 'numpy')
        f_true = sympy.lambdify([sympy.Symbol(v) for v in var_names], true, 'numpy')
        
        y_pred = f_pred(*X_rand.T)
        y_true = f_true(*X_rand.T)
        
        if not np.all(np.isfinite(y_pred)): return False
        
        return np.allclose(y_pred, y_true, atol=tol, rtol=tol)
    except Exception:
        return False


def calculate_constant_recovery(
    fitted_params: List[float],
    true_params:   List[float],
    tol:           float = 0.01
) -> float:
    """
    Check if fitted constants match true constants within relative tolerance.
    
    Args:
        fitted_params: constants from BFGS fitting
        true_params:   constants from ground truth source
        tol:           relative tolerance (default 0.01 = 1%)
        
    Returns:
        1.0 if all match, else 0.0
    """
    if len(fitted_params) != len(true_params):
        return 0.0
    
    # Sort for robust comparison since we are in SA-true case
    f_sorted = sorted(fitted_params)
    t_sorted = sorted(true_params)
    
    for f, t in zip(f_sorted, t_sorted):
        denom = abs(t) + 1e-10
        if abs(f - t) / denom >= tol:
            return 0.0
    return 1.0


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

