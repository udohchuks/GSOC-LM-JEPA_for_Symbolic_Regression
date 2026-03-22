"""
Unit Table for LLM-JEPA Symbolic Regression.

Physical unit vectors for all variables in the Feynman database.
Source: Table III of Udrescu & Tegmark (2020), AI Feynman.

Unit vector format: [meters, seconds, kilograms, kelvin, volts]
Each entry is integer exponents, range [-4, +4].

Classifier conversion:
    Raw exponent + UNIT_OFFSET → class index in [0, 8]
    UNIT_OFFSET = 4 (shifts [-4,+4] to [0,8] for PyTorch cross_entropy)
"""

from __future__ import annotations
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
N_UNIT_DIMS    = 5          # [m, s, kg, K, V]
UNIT_MIN       = -4         # minimum exponent in Feynman dataset
UNIT_MAX       =  4         # maximum exponent in Feynman dataset
UNIT_OFFSET    =  4         # subtract UNIT_MIN: shifts [-4,4] → [0,8]
N_UNIT_CLASSES = 9          # UNIT_MAX - UNIT_MIN + 1

DIMENSIONLESS  = [0, 0, 0, 0, 0]

# ── Unit lookup table ─────────────────────────────────────────────────────────
# Key: variable name exactly as it appears in the AIF CSV
# Value: [m, s, kg, K, V] exponents
UNIT_TABLE: dict[str, list[int]] = {

    # Dimensionless
    'theta':  [0,  0,  0,  0,  0],
    'theta1': [0,  0,  0,  0,  0],
    'theta2': [0,  0,  0,  0,  0],
    'sigma':  [0,  0,  0,  0,  0],
    'alpha':  [0,  0,  0,  0,  0],
    'n':      [0,  0,  0,  0,  0],
    'n0':     [0,  0,  0,  0,  0],
    'delta':  [0,  0,  0,  0,  0],
    'beta':   [0,  0,  0,  0,  0],
    'gamma':  [0,  0,  0,  0,  0],
    'chi':    [0,  0,  0,  0,  0],
    'kf':     [0,  0,  0,  0,  0],
    'sigma_stefan': [0, -3, 1, -4, 0],
    'sigma_charge': [0, -2, 1, 0, -1],
    'n_density': [-3, 0, 0, 0, 0],

    # Length [m]
    'x':      [1,  0,  0,  0,  0],
    'x1':     [1,  0,  0,  0,  0],
    'x2':     [1,  0,  0,  0,  0],
    'x3':     [1,  0,  0,  0,  0],
    'y1':     [1,  0,  0,  0,  0],
    'y2':     [1,  0,  0,  0,  0],
    'z1':     [1,  0,  0,  0,  0],
    'z2':     [1,  0,  0,  0,  0],
    'r':      [1,  0,  0,  0,  0],
    'r1':     [1,  0,  0,  0,  0],
    'r2':     [1,  0,  0,  0,  0],
    'd':      [1,  0,  0,  0,  0],
    'd1':     [1,  0,  0,  0,  0],
    'd2':     [1,  0,  0,  0,  0],
    'lambda': [1,  0,  0,  0,  0],
    'af':     [1,  0,  0,  0,  0],
    'ff':     [1,  0,  0,  0,  0],

    # Velocity [m/s]
    'v':      [1, -1,  0,  0,  0],
    'v1':     [1, -1,  0,  0,  0],
    'u':      [1, -1,  0,  0,  0],
    'c':      [1, -1,  0,  0,  0],
    'w':      [1, -1,  0,  0,  0],

    # Acceleration [m/s²]
    'a':      [1, -2,  0,  0,  0],

    # Mass [kg]
    'm':      [0,  0,  1,  0,  0],
    'm0':     [0,  0,  1,  0,  0],
    'm1':     [0,  0,  1,  0,  0],
    'm2':     [0,  0,  1,  0,  0],

    # Time [s]
    't':      [0,  1,  0,  0,  0],
    't1':     [0,  1,  0,  0,  0],

    # Force [kg·m/s²]
    'F':      [1, -2,  1,  0,  0],
    'Nn':     [1, -2,  1,  0,  0],

    # Energy [kg·m²/s²]
    'E':      [2, -2,  1,  0,  0],
    'K':      [2, -2,  1,  0,  0],
    'U':      [2, -2,  1,  0,  0],

    # Power [kg·m²/s³]
    'P':      [2, -3,  1,  0,  0],

    # Momentum [kg·m/s]
    'p':      [1, -1,  1,  0,  0],

    # Angular momentum [kg·m²/s]
    'h':      [2, -1,  1,  0,  0],
    'hbar':   [2, -1,  1,  0,  0],
    'L':      [2, -1,  1,  0,  0],
    'Jz':     [2, -1,  1,  0,  0],

    # Torque [kg·m²/s²]
    'tau':    [2, -2,  1,  0,  0],

    # Temperature [K]
    'T':      [0,  0,  0,  1,  0],
    'T1':     [0,  0,  0,  1,  0],
    'T2':     [0,  0,  0,  1,  0],

    # Boltzmann constant [kg·m²/(s²·K)]
    'kb':     [2, -2,  1, -1,  0],

    # Gravitational constant [m³/(kg·s²)]
    'G':      [3, -2, -1,  0,  0],
    'epsilon0': [1, -2, 1, 0, -2],
    'mu0':      [-3, 4, -1, 0, 2],

    # Electric charge [kg·m²/(s²·V)]
    'q':      [2, -2,  1,  0, -1],
    'q1':     [2, -2,  1,  0, -1],
    'q2':     [2, -2,  1,  0, -1],

    # Voltage [V]
    'Ve':     [0,  0,  0,  0,  1],
    'V1':     [0,  0,  0,  0,  1],
    'V2':     [0,  0,  0,  0,  1],
    'R':        [-2, 3, -1, 0, 2],
    'C_cap':    [2, -2, 1, 0, -2],
    'I_c':      [2, -3, 1, 0, -1],

    # Electric field [V/m]
    'Ef':     [-1,  0,  0,  0,  1],

    # Magnetic field [V·s/m²]
    'B':      [-2,  1,  0,  0,  1],
    'Bx':     [-2,  1,  0,  0,  1],
    'By':     [-2,  1,  0,  0,  1],
    'Bz':     [-2,  1,  0,  0,  1],

    # Frequency [1/s]
    'omega':  [0, -1,  0,  0,  0],
    'omega0': [0, -1,  0,  0,  0],
    'nu':       [0, -1,  0,  0,  0],

    # Wave number [1/m]
    'k':      [-1,  0,  0,  0,  0],

    # Spring constant [kg/s²]
    'kspring': [0, -2,  1,  0,  0],

    # Density [kg/m³]
    'rho':    [-3,  0,  1,  0,  0],
    'rho0':   [-3,  0,  1,  0,  0],

    # Pressure [kg/(m·s²)]
    'pF':     [-1, -2,  1,  0,  0],

    # Area [m²]
    'A':      [2,  0,  0,  0,  0],

    # Volume [m³]
    'V':      [3,  0,  0,  0,  0],

    # Electric dipole moment [kg·m³/(s²·V)]
    'pd':     [3, -2,  1,  0, -1],

    # Inductance [kg·m²/(s²·V²)·s²] → [-2, 4, -1, 0, 2] from paper
    'Lind':   [-2,  4, -1,  0,  2],

    # Light intensity [kg/s³]
    'I':      [0, -3,  1,  0,  0],
    'I0':     [0, -3,  1,  0,  0],
    'I1':     [0, -3,  1,  0,  0],
    'I2':     [0, -3,  1,  0,  0],

    # Hubble constant [1/s]
    'H':      [0, -1,  0,  0,  0],

    # Diffusion coefficient [m²/s]
    'D':      [2, -1,  0,  0,  0],
}

# ── Lookup functions ──────────────────────────────────────────────────────────

def get_unit_vector(var_name: str,
                    warn_unknown: bool = True) -> list[int]:
    """
    Look up the unit vector for a variable name.

    Returns DIMENSIONLESS [0,0,0,0,0] for unknown variables.
    This is a safe fallback — the model will simply not have
    unit information for that variable.

    Args:
        var_name:     variable name as in the AIF CSV
        warn_unknown: print warning for unknown variables
    """
    if var_name in UNIT_TABLE:
        return UNIT_TABLE[var_name]

    if warn_unknown:
        print(f"Warning: unknown variable '{var_name}', "
              f"using dimensionless [0,0,0,0,0]")
    return DIMENSIONLESS.copy()

def get_unit_matrix(var_names: list[str],
                    warn_unknown: bool = True) -> np.ndarray:
    """
    Build unit matrix for a list of variable names.

    Args:
        var_names: variable names in order

    Returns:
        np.ndarray of shape (n_vars, 5), dtype int32
    """
    vectors = [get_unit_vector(name, warn_unknown) for name in var_names]
    return np.array(vectors, dtype=np.int32)

def unit_to_class_indices(unit_matrix: np.ndarray) -> np.ndarray:
    """
    Convert raw unit exponents to classifier class indices.

    Shifts [-4, +4] → [0, 8] by adding UNIT_OFFSET = 4.
    Required because PyTorch cross_entropy needs non-negative indices.

    Args:
        unit_matrix: (..., 5) integer array of unit exponents

    Returns:
        (..., 5) integer array of class indices in [0, 8]
    """
    indices = unit_matrix + UNIT_OFFSET

    # Safety check — if this fires you have a variable outside [-4,+4]
    assert (indices >= 0).all() and (indices <= 8).all(), (
        f"Unit exponents outside [-4,+4] range. "
        f"min={unit_matrix.min()}, max={unit_matrix.max()}"
    )
    return indices

def class_indices_to_units(indices: np.ndarray) -> np.ndarray:
    """
    Convert class indices back to unit exponents.
    Inverse of unit_to_class_indices.
    """
    return indices - UNIT_OFFSET


if __name__ == '__main__':
    # Test 1: known variables
    assert get_unit_vector('theta') == [0, 0, 0, 0, 0]
    assert get_unit_vector('F')     == [1, -2, 1, 0, 0]
    assert get_unit_vector('G')     == [3, -2, -1, 0, 0]
    assert get_unit_vector('kb')    == [2, -2, 1, -1, 0]
    print('Known variables: OK')

    # Test 2: unknown variable fallback
    result = get_unit_vector('unknown_var', warn_unknown=False)
    assert result == [0, 0, 0, 0, 0]
    print('Unknown fallback: OK')

    # Test 3: unit matrix
    mat = get_unit_matrix(['theta', 'F', 'G'])
    assert mat.shape == (3, 5)
    assert mat[0].tolist() == [0, 0, 0, 0, 0]   # theta
    assert mat[1].tolist() == [1, -2, 1, 0, 0]  # F
    print('Unit matrix: OK')

    # Test 4: class index conversion
    idx = unit_to_class_indices(mat)
    assert idx.min() >= 0 and idx.max() <= 8
    # theta [0,0,0,0,0] → all indices = 4 (offset)
    assert idx[0].tolist() == [4, 4, 4, 4, 4]
    # F [1,-2,1,0,0] → [5, 2, 5, 4, 4]
    assert idx[1].tolist() == [5, 2, 5, 4, 4]
    print('Class index conversion: OK')

    # Test 5: round trip
    recovered = class_indices_to_units(idx)
    assert (recovered == mat).all()
    print('Round trip: OK')

    print('\nAll unit table tests passed.')

    assert get_unit_vector('epsilon0') == [1, -2, 1, 0, -2]
    assert get_unit_vector('mu0')      == [-3, 4, -1, 0, 2]
    assert get_unit_vector('R')        == [-2, 3, -1, 0, 2]
    assert get_unit_vector('I_c')      == [2, -3, 1, 0, -1]
    assert get_unit_vector('sigma_stefan') == [0, -3, 1, -4, 0]
    assert get_unit_vector('n_density')    == [-3, 0, 0, 0, 0]
    print('Extended unit table: OK')