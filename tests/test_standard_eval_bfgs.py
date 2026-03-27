import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from evaluation.evaluate import _evaluate_one
from data.aif_dataset import PreprocessedEquation
from data.tokenizer import encode_formula, MAX_SEQ_LEN
from data.utils import to_ieee754_16bit


def _make_eq(eq_id, var_names, X, y, formula_str, token_seq):
    """Utility to construct a ``PreprocessedEquation`` with a mocked model prediction.

    Args:
        eq_id: Identifier string (e.g., "I.12.1").
        var_names: List of variable names used in the equation.
        X: Input array of shape (n_points, n_vars).
        y: Target array of shape (n_points,).
        formula_str: Human‑readable ground‑truth formula (for debugging).
        token_seq: List of token strings representing the model's predicted RPN
            expression (constants are represented as ``c1``, ``c2`` …).
    """
    n_points, n_vars = X.shape
    X_bits = to_ieee754_16bit(X)
    unit_matrix_idx = np.zeros((n_vars, 5), dtype=np.int64)
    token_ids = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    unit_targets_idx = np.zeros((MAX_SEQ_LEN, 5), dtype=np.int64)
    # Encode the predicted token sequence – the evaluation code will replace the
    # placeholder constants via BFGS.
    predicted_token_ids = encode_formula(token_seq, add_bos=False, add_eos=False)
    # Mock model that always returns the same token ids.
    model = MagicMock()
    model.max_n_vars = 9
    model.eval = MagicMock()
    model.return_value = torch.zeros(1, 128)  # dummy context output
    model.generate.return_value = torch.tensor([predicted_token_ids])
    eq = PreprocessedEquation(
        eq_id=eq_id,
        X_bits=X_bits,
        y=y,
        unit_matrix_idx=unit_matrix_idx,
        token_ids=token_ids,
        unit_targets_idx=unit_targets_idx,
        var_names=var_names,
        formula_str=formula_str,
        n_vars=n_vars,
    )
    return eq, model


@pytest.mark.parametrize(
    "scale, expect_improvement",
    [
        (1.0, False),   # No scaling – baseline.
        (2.5, True),    # Scaling – BFGS must adjust the constant.
        (3.0, True),    # Additional scaling case.
    ],
)
def test_single_constant_scaling(scale, expect_improvement):
    """Evaluate BFGS on a simple product ``mu * Nn`` with an optional scaling factor.

    The mocked model predicts the RPN ``c1 * x1 * x2`` (i.e. ``c1 * mu * Nn``).
    """
    eq_id = "I.12.1"
    var_names = ["mu", "Nn"]
    n_points = 200
    mu = np.random.uniform(1, 5, n_points).astype(np.float32)
    Nn = np.random.uniform(1, 5, n_points).astype(np.float32)
    X = np.column_stack([mu, Nn])
    y = scale * mu * Nn
    token_seq = ["x1", "c1", "*", "x2", "*"]
    eq, model = _make_eq(eq_id, var_names, X, y, f"{scale}*mu*Nn", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=3)

    # Ensure the result contains the expected keys.
    assert "r2_pre_bfgs" in result and "r2_post_bfgs" in result
    # The predicted expression is a string.
    assert isinstance(result["predicted"], str)
    if expect_improvement:
        # Post‑BFGS R² should be high and constant appears (tolerant check).
        assert result["r2_post_bfgs"] > 0.999
        import re
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", result["predicted"])
        assert match is not None, "No numeric constant found in prediction"
        fitted_val = float(match.group(0))
        assert abs(fitted_val - scale) < 0.02
    else:
        # For baseline we only require a valid expression and R² may be undefined.
        assert result["predicted"]
    assert result["valid_rpn"] is True


def test_multiple_constants():
    """Test BFGS with two independent constants.

    Ground truth: ``a * mu + b * Nn`` where ``a=1.7`` and ``b=3.2``.
    Model predicts ``c1 * x1 + c2 * x2``.
    """
    eq_id = "I.12.1_multi"
    var_names = ["mu", "Nn"]
    n_points = 250
    mu = np.random.uniform(0.5, 4.0, n_points).astype(np.float32)
    Nn = np.random.uniform(0.5, 4.0, n_points).astype(np.float32)
    a, b = 1.7, 3.2
    y = a * mu + b * Nn
    X = np.column_stack([mu, Nn])
    token_seq = ["x1", "c1", "*", "x2", "c2", "*", "+"]
    eq, model = _make_eq(eq_id, var_names, X, y, f"{a}*mu + {b}*Nn", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=5)

    assert result["r2_post_bfgs"] > 0.999
    # Ensure high R2 and valid RPN; constants may not be individually verified.
    assert result["valid_rpn"] is True


def test_invalid_rpn_handling():
    """Ensure the evaluation gracefully marks an invalid RPN expression.

    The token sequence ``['x1', '*']`` lacks a second operand, which should be
    detected by ``data.tokenizer.is_valid_rpn`` and result in ``valid_rpn=False``.
    """
    eq_id = "I.invalid"
    var_names = ["x1"]
    n_points = 100
    x1 = np.random.uniform(1, 3, n_points).astype(np.float32)
    y = 2.0 * x1  # ground truth (unused for validity check)
    X = x1.reshape(-1, 1)
    token_seq = ["x1", "*"]  # malformed RPN
    eq, model = _make_eq(eq_id, var_names, X, y, "2*x1", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=2)

    # The pipeline should still return a result dict, but ``valid_rpn`` must be False.
    assert result["valid_rpn"] is False
    # R² values are undefined in this case; they may be ``-inf`` or ``nan``.
    assert np.isneginf(result["r2_pre_bfgs"]) or np.isnan(result["r2_pre_bfgs"])
    assert np.isneginf(result["r2_post_bfgs"]) or np.isnan(result["r2_post_bfgs"])


def test_nonlinear_exponential():
    """Test BFGS with a nonlinear exponential function: c1 * exp(c2 * x1)."""
    eq_id = "nonlinear_exp"
    var_names = ["x1"]
    n_points = 200
    x1 = np.random.uniform(0.1, 2.0, n_points).astype(np.float32)
    c1_true, c2_true = 1.5, 0.8
    y = c1_true * np.exp(c2_true * x1)
    X = x1.reshape(-1, 1)
    # Model predicts c1 * exp(c2 * x1) in RPN
    token_seq = ["c1", "x1", "c2", "*", "exp", "*"]
    eq, model = _make_eq(eq_id, var_names, X, y, "1.5 * exp(0.8 * x1)", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=5)

    assert result["valid_rpn"] is True
    assert result["r2_post_bfgs"] > 0.99
    assert "exp" in result["predicted"]


def test_trigonometric():
    """Test BFGS with a trigonometric function: c1 * sin(x1) + c2 * cos(x2)."""
    eq_id = "trig_test"
    var_names = ["x1", "x2"]
    n_points = 300
    x1 = np.random.uniform(0, 2*np.pi, n_points).astype(np.float32)
    x2 = np.random.uniform(0, 2*np.pi, n_points).astype(np.float32)
    c1_true, c2_true = 2.5, -1.2
    y = c1_true * np.sin(x1) + c2_true * np.cos(x2)
    X = np.column_stack([x1, x2])
    # RPN: x1 sin c1 * x2 cos c2 * +
    token_seq = ["x1", "sin", "c1", "*", "x2", "cos", "c2", "*", "+"]
    eq, model = _make_eq(eq_id, var_names, X, y, "2.5*sin(x1) - 1.2*cos(x2)", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=5)

    assert result["valid_rpn"] is True
    assert result["r2_post_bfgs"] > 0.99
    assert "sin" in result["predicted"] and "cos" in result["predicted"]


def test_division_with_constant():
    """Test BFGS with division: c1 / (x1 + c2)."""
    eq_id = "div_test"
    var_names = ["x1"]
    n_points = 200
    x1 = np.random.uniform(1, 10, n_points).astype(np.float32)
    c1_true, c2_true = 10.0, 5.0
    y = c1_true / (x1 + c2_true)
    X = x1.reshape(-1, 1)
    # RPN: c1 x1 c2 + /
    token_seq = ["c1", "x1", "c2", "+", "/"]
    eq, model = _make_eq(eq_id, var_names, X, y, "10 / (x1 + 5)", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=5)

    assert result["valid_rpn"] is True
    assert result["r2_post_bfgs"] > 0.99


def test_division_by_zero_runtime():
    """Ensure the evaluation handles runtime division by zero in predicted formula."""
    eq_id = "div_zero"
    var_names = ["x1"]
    n_points = 100
    x1 = np.random.uniform(1, 10, n_points).astype(np.float32)
    y = 2.0 * x1
    X = x1.reshape(-1, 1)
    # Model predicts 1 / (x1 - x1) -> always zero denominator
    token_seq = ["1", "x1", "x1", "-", "/"]
    eq, model = _make_eq(eq_id, var_names, X, y, "2*x1", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=2)

    assert result["valid_rpn"] is True
    # R2 should be -inf or nan because of division by zero
    assert not np.isfinite(result["r2_pre_bfgs"])
    assert not np.isfinite(result["r2_post_bfgs"])


def test_overflow_expression():
    """Ensure the evaluation handles expressions that result in overflow/non-finite values."""
    eq_id = "overflow"
    var_names = ["x1"]
    n_points = 100
    x1 = np.random.uniform(10, 20, n_points).astype(np.float32)
    y = x1
    X = x1.reshape(-1, 1)
    # exp(exp(exp(x1))) for x1 in [10, 20] will definitely overflow float64
    token_seq = ["x1", "exp", "exp", "exp"]
    eq, model = _make_eq(eq_id, var_names, X, y, "x1", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=2)

    assert result["valid_rpn"] is True
    assert not np.isfinite(result["r2_pre_bfgs"])
    assert not np.isfinite(result["r2_post_bfgs"])


def test_undefined_variable():
    """Test behavior when the model predicts a variable index not present in the input."""
    eq_id = "undefined_var"
    var_names = ["x1", "x2"] # Only 2 variables
    n_points = 100
    X = np.random.uniform(1, 5, (n_points, 2)).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    # Model predicts x3 + x1, but x3 is undefined (n_vars=2)
    token_seq = ["x3", "x1", "+"]
    eq, model = _make_eq(eq_id, var_names, X, y, "x1+x2", token_seq)
    
    # We need to make sure encode_formula allows x3 for this test
    # or handle it if it fails earlier. 
    # Actually, encode_formula will encode x3 as a token.
    # rpn_to_sympy will create a Symbol('x3').
    # _evaluate_expr will try to lambdify but only pass X.T which has 2 columns.
    # This should be caught.
    
    result = _evaluate_one(model, eq, device="cpu", n_restarts=2)

    # If it fails evaluation, it should return -inf R2
    assert not np.isfinite(result["r2_pre_bfgs"])
    assert not np.isfinite(result["r2_post_bfgs"])


def test_only_constants():
    """Test behavior when the model predicts only constants against a target with variance.
    
    BFGS should find the optimal constant (the mean of y).
    """
    eq_id = "only_const"
    var_names = ["x1"]
    n_points = 100
    x1 = np.random.uniform(1, 10, n_points).astype(np.float32)
    y = 5.0 + 0.5 * x1 # Ground truth has variance
    X = x1.reshape(-1, 1)
    # Model predicts c1 (constant only)
    token_seq = ["c1"]
    eq, model = _make_eq(eq_id, var_names, X, y, "5.0 + 0.5*x1", token_seq)
    result = _evaluate_one(model, eq, device="cpu", n_restarts=3)

    assert result["valid_rpn"] is True
    # R2 should be finite and positive (it explains the mean)
    assert np.isfinite(result["r2_post_bfgs"])
    # The predicted constant should be near the mean of y
    import re
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", result["predicted"])
    assert match is not None
    fitted_val = float(match.group(0))
    expected_mean = np.mean(y)
    assert abs(fitted_val - expected_mean) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
