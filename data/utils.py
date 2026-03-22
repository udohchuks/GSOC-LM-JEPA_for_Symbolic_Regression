"""
Data utilities for LLM-JEPA Symbolic Regression.

Three independent utilities:
    1. IEEE-754 16-bit scalar encoding
    2. Gaussian noise addition
    3. Unit stack simulation for unit target computation

All functions are stateless and can be called from both
the AIF dataset loader and the synthetic data generator.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
import sympy

def to_ieee754_16bit(x: np.ndarray) -> np.ndarray:
    """
    Convert float array to 16-bit IEEE-754 multi-hot binary representation.

    Every scalar maps to exactly 16 binary values {0.0, 1.0} regardless
    of magnitude. Scale-invariant by construction.

    Validated encoding used by NeSymReS (Biggio et al. 2021) and
    SNIP (Meidani et al. 2024).

    Why numpy only (no torch):
        np.unpackbits operates on uint8 arrays — no torch equivalent.
        Pre-compute this on CPU during data loading and cache the result.
        Do NOT run this inside the model forward pass.

    Args:
        x: numpy float array of any shape [...]

    Returns:
        numpy float32 array of shape [..., 16]
        values are exactly 0.0 or 1.0
    """
    # Step 1: cast to float16 — half precision
    # This quantises the values but gives us exactly 16 bits to work with
    x_f16 = x.astype(np.float16)

    # Step 2: view the raw memory as uint16
    # This reinterprets the same bytes as unsigned integers
    # No computation — just a different lens on the same memory
    x_unit16 = x_f16.view(np.uint16)

    # Step 2: view the raw memory as uint16
    # This reinterprets the same bytes as unsigned integers
    # No computation — just a different lens on the same memory
    x_unit8 = x_unit16.view(np.uint8)

    # Step 4: unpack each byte into 8 individual bits
    # bitorder='big': most significant bit first
    # Result shape: [..., 16] 
    bits = np.unpackbits(x_unit8, axis=-1, bitorder='big')

    target_shape = x.shape + (16,)
    # Step 5: reshape to [..., 16] and cast to float32 for PyTorch
    return bits.reshape(target_shape).astype(np.float32)

# ── 2. Gaussian noise ─────────────────────────────────────────────────────────

# Noise level distribution for pretraining
# ~50% clean examples ensures the model learns exact recovery
# ~50% noisy examples ensures robustness to measurement error
NOISE_LEVELS = [0.0, 0.0, 0.0, 1e-4, 1e-3, 1e-2]

def add_gaussian_noise(y: np.ndarray,
                       epsilon: Optional[float] = None) -> np.ndarray:
    """
    Add Gaussian noise to output variable y.

    Noise is added to y ONLY, not to X. This matches real laboratory
    conditions where input variables are measured precisely but outputs
    have measurement error (sensor noise, quantisation, interference).

    Noise is scaled by y_rms so epsilon represents a relative level:
        epsilon = 0.01 means noise standard deviation is 1% of signal RMS
    Same convention as the AI Feynman paper (Section III.D).

    Args:
        y:       1D array of output values, shape [N]
        epsilon: relative noise level. None = sample from NOISE_LEVELS.
                 0.0 = clean, 0.01 = 1% noise (realistic lab condition)

    Returns:
        Noisy y array, same shape and dtype as input.
    """
    if epsilon is None:
        epsilon = float(np.random.choice(NOISE_LEVELS))

    if epsilon == 0.0:
        return y.copy() # explicit copy so caller can modify freely

    y_rms = np.sqrt(np.mean(y ** 2))
    if y_rms < 1e-10:
        return y.copy() # near-zero signal: noise would dominate, skip
    
    noise = np.random.normal(
        loc=0.0,
        scale=epsilon * y_rms,
        size=y.shape
    ).astype(y.dtype)

    return y + noise


# ── 3. Unit stack simulation ──────────────────────────────────────────────────

# Import here to avoid circular imports at module level
    from data.unit_table import (
        N_UNIT_DIMS, UNIT_OFFSET, N_UNIT_CLASSES,
        get_unit_vector, DIMENSIONLESS
    )
    from data.tokenizer import ARITY, UNARY_TOKENS, BINARY_TOKENS

def _propagate_units(operator: str,
                     stack: List[List[int]]) -> Optional[List[int]]:
    """
    Compute output units after applying an operator to the unit stack.

    This implements the physical unit algebra:
        multiply:  add exponents
        divide:    subtract exponents
        add/sub:   require identical units, preserve them
        sqrt:      halve exponents (only valid if all even)
        sin/exp:   require dimensionless argument, output dimensionless

    Returns None if the operation is dimensionally invalid.
    This should not happen for well-formed physics equations.
    """
    if operator in ('+', '-'):
        if len(stack) < 2:
            return None
        A, B = stack[-2], stack[-1]
        if A != B:
            return None        # cannot add quantities with different units
        return A[:]            # copy
    elif operator == '*':
        if len(stack) < 2:
            return None
        A, B = stack[-2], stack[-1]
        return [A[i] + B[i] for i in range(N_UNIT_DIMS)]
    elif operator == '/':
        if len(stack) < 2:
            return None
        A, B = stack[-2], stack[-1]
        return [A[i] - B[i] for i in range(N_UNIT_DIMS)]
    elif operator == 'sqrt':
        if len(stack) < 1:
            return None
        A = stack[-1]
        if any(e % 2 != 0 for e in A):
            return None        # sqrt([1,0,...]) = m^0.5, not a valid SI unit
        return [e // 2 for e in A]
    elif operator == 'sq':
        if len(stack) < 1:
            return None
        return [e * 2 for e in stack[-1]]
    elif operator in ('neg', 'abs'):
        if len(stack) < 1:
            return None
        return stack[-1][:]  # preserve units
    elif operator in ('exp', 'log', 'sin', 'cos', 'tan',
                      'arcsin', 'arccos', 'arctan'):
        if len(stack) < 1:
            return None
        if stack[-1] != [0] * N_UNIT_DIMS:
            return None        # argument must be dimensionless
        return [0] * N_UNIT_DIMS 
    return None  # unkown operator

def compute_unit_targets(
    rpn_tokens: List[str],
    var_names: List[str],
) -> List[List[int]]:
    """
    Compute the unit vector each token pushes onto the stack,
    for every position in the RPN sequence.

    This is computed once offline from the ground truth formula.
    The results are stored in the dataset and used as supervision
    targets for the unit prediction head during training.

    Why this specific information as the target?
        At each decoder step t, the unit head predicts:
        "what units must the token I am about to generate have?"
        That is exactly what this function computes — for each
        position t, what units did the ground truth token push?

    Args:
        rpn_tokens: list of RPN token strings (without BOS/EOS)
        var_names:  variable names in order (x1 → var_names[0], etc.)

    Returns:
        List of unit vectors, one per token, same length as rpn_tokens.
        Each unit vector is a list of 5 integers.
    """
    # Build token → unit vector mapping for this equation's variables
    # x1 maps to the first variable's units, x2 to second, etc.
    token_units: dict[str, List[int]] = {}
    for i, name in enumerate(var_names):
        tok = f'x{i+1}'
        token_units[tok] = get_unit_vector(name, warn_unknown=True)
    
    # Constants are always dimensionless
    for c in ('0', '1', '2', '3', 'pi', 'e'):
        token_units[c] = [0] * N_UNIT_DIMS
    unit_targets: List[List[int]] = []
    unit_stack:   List[List[int]] = []

    for tok in rpn_tokens:
        arity = ARITY.get(tok, 0)

        if arity == 0:
            # Leaf: push its unit vector
            units = token_units.get(tok, [0] * N_UNIT_DIMS)
            unit_targets.append(units[:])
            unit_stack.append(units[:])
        elif arity == 1:
            # Unary operator: pop one, compute result, push result
            result = _propagate_units(tok, unit_stack)
            if result is None:
                result = [0] * N_UNIT_DIMS   # fallback for invalid
            unit_targets.append(result[:])
            if unit_stack:
                unit_stack.pop()
            unit_stack.append(result[:])
        elif arity == 2:
            # Binary operator: pop two, compute result, push result
            result = _propagate_units(tok, unit_stack)
            if result is None:
                result = [0] * N_UNIT_DIMS
            unit_targets.append(result[:])
            if len(unit_stack) >= 2:
                unit_stack.pop()
                unit_stack.pop()
            unit_stack.append(result[:])
    return unit_targets

def unit_targets_to_class_indices(
    unit_targets: List[List[int]]
) -> np.ndarray:
    """
    Convert list of unit vectors to classifier class index array.

    Applies UNIT_OFFSET=4 to shift raw exponents [-4,+4] to
    valid PyTorch class indices [0,8].

    Args:
        unit_targets: list of T unit vectors, each of length 5

    Returns:
        np.ndarray of shape (T, 5), dtype int64, values in [0, 8]
    """
    arr = np.array(unit_targets, dtype=np.int64)   # shape [T, 5]
    arr = arr + UNIT_OFFSET
    arr = np.clip(arr, 0, N_UNIT_CLASSES - 1)      # safety clip
    return arr



if __name__ == '__main__':
    # ── Test 1: IEEE-754 ──────────────────────────────────────────────────
    x = np.array([[1.0, 2.0, -1.5]], dtype=np.float32)
    bits = to_ieee754_16bit(x)
    assert bits.shape == (1, 3, 16), f"Got shape {bits.shape}"
    assert set(bits.flatten().tolist()).issubset({0.0, 1.0})

    # Different values must produce different bit patterns
    a = to_ieee754_16bit(np.array([1.0], dtype=np.float32))
    b = to_ieee754_16bit(np.array([2.0], dtype=np.float32))
    assert not np.array_equal(a, b)
    print('IEEE-754: OK')

    # ── Test 2: Gaussian noise ────────────────────────────────────────────
    y = np.sin(np.linspace(0, 3, 1000)).astype(np.float32)

    y_clean = add_gaussian_noise(y, epsilon=0.0)
    assert np.array_equal(y, y_clean)   # zero noise = identical

    y_noisy = add_gaussian_noise(y, epsilon=0.01)
    assert y_noisy.shape == y.shape
    assert not np.allclose(y, y_noisy)  # noise was added

    # Noise level should be approximately 1% of RMS
    y_rms = np.sqrt(np.mean(y ** 2))
    noise_std = np.std(y_noisy - y)
    assert abs(noise_std / y_rms - 0.01) < 0.005  # within 0.5%
    print('Gaussian noise: OK')

    # ── Test 3: unit stack — simple cases ─────────────────────────────────

    # x1 * x2 where x1=m1 (kg), x2=m2 (kg) → result should be kg²
    targets = compute_unit_targets(['x1', 'x2', '*'], ['m1', 'm2'])
    assert len(targets) == 3
    assert targets[0] == [0, 0, 1, 0, 0]   # m1: mass [kg]
    assert targets[1] == [0, 0, 1, 0, 0]   # m2: mass [kg]
    assert targets[2] == [0, 0, 2, 0, 0]   # *: kg² (add exponents)
    print('Unit stack (m1*m2=kg²): OK')

    # x1 / x2 where both are lengths → result is dimensionless
    targets = compute_unit_targets(['x1', 'x2', '/'], ['r1', 'r2'])
    assert targets[2] == [0, 0, 0, 0, 0]   # dimensionless
    print('Unit stack (r1/r2=dimensionless): OK')

    # sin(theta) where theta is dimensionless → result is dimensionless
    targets = compute_unit_targets(['x1', 'sin'], ['theta'])
    assert targets[0] == [0, 0, 0, 0, 0]   # theta: dimensionless
    assert targets[1] == [0, 0, 0, 0, 0]   # sin: dimensionless out
    print('Unit stack (sin(theta)): OK')

    # ── Test 4: class index conversion ────────────────────────────────────
    targets = compute_unit_targets(['x1', 'x2', '*'], ['m1', 'F'])
    indices = unit_targets_to_class_indices(targets)
    assert indices.shape == (3, 5)
    assert indices.min() >= 0 and indices.max() <= 8

    # mass [0,0,1,0,0] → class indices [4,4,5,4,4]
    assert indices[0].tolist() == [4, 4, 5, 4, 4]
    print('Class index conversion: OK')

    print('\nAll utils tests passed.')