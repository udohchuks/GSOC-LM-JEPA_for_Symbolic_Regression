"""
Unit Table for LLM-JEPA Symbolic Regression.

Physical unit vectors for all variables in the Feynman database.
Source: Table III of Udrescu & Tegmark (2020), AI Feynman + units.csv.

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
# 
# NOTE: Ambiguous variables (H, I, V) use the MOST COMMON meaning.
# For equations where they differ, the CSV row's unit info should override.
UNIT_TABLE: dict[str, list[int]] = {
    # ── Dimensionless ─────────────────────────────────────────────────────
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
    'mu':     [0,  0,  0,  0,  0],
    'prob':   [0,  0,  0,  0,  0],
    'f':      [0,  0,  0,  0,  0],
    'g_':     [0,  0,  0,  0,  0],
    'Z_1':    [0,  0,  0,  0,  0],
    'Z_2':    [0,  0,  0,  0,  0],
    
    # ── Length [m] ────────────────────────────────────────────────────────
    'x':      [1,  0,  0,  0,  0],
    'x1':     [1,  0,  0,  0,  0],
    'x2':     [1,  0,  0,  0,  0],
    'x3':     [1,  0,  0,  0,  0],
    'y':      [1,  0,  0,  0,  0],
    'y1':     [1,  0,  0,  0,  0],
    'y2':     [1,  0,  0,  0,  0],
    'y3':     [1,  0,  0,  0,  0],
    'z':      [1,  0,  0,  0,  0],
    'z1':     [1,  0,  0,  0,  0],
    'z2':     [1,  0,  0,  0,  0],
    'r':      [1,  0,  0,  0,  0],
    'r1':     [1,  0,  0,  0,  0],
    'r2':     [1,  0,  0,  0,  0],
    'd':      [1,  0,  0,  0,  0],
    'd1':     [1,  0,  0,  0,  0],
    'd2':     [1,  0,  0,  0,  0],
    'lambda': [1,  0,  0,  0,  0],
    'lambd':  [1,  0,  0,  0,  0],
    'af':     [1,  0,  0,  0,  0],
    'ff':     [1,  0,  0,  0,  0],
    'foc':    [1,  0,  0,  0,  0],
    
    # ── Velocity [m/s] ────────────────────────────────────────────────────
    'v':      [1, -1,  0,  0,  0],
    'v1':     [1, -1,  0,  0,  0],
    'u':      [1, -1,  0,  0,  0],
    'c':      [1, -1,  0,  0,  0],
    'w':      [1, -1,  0,  0,  0],
    
    # ── Acceleration [m/s²] ───────────────────────────────────────────────
    'a':      [1, -2,  0,  0,  0],
    'g':      [1, -2,  0,  0,  0],
    
    # ── Mass [kg] ─────────────────────────────────────────────────────────
    'm':      [0,  0,  1,  0,  0],
    'm0':     [0,  0,  1,  0,  0],
    'm_0':    [0,  0,  1,  0,  0],
    'm1':     [0,  0,  1,  0,  0],
    'm2':     [0,  0,  1,  0,  0],
    
    # ── Time [s] ──────────────────────────────────────────────────────────
    't':      [0,  1,  0,  0,  0],
    't1':     [0,  1,  0,  0,  0],
    
    # ── Force [kg·m/s²] ───────────────────────────────────────────────────
    'F':      [1, -2,  1,  0,  0],
    'Nn':     [1, -2,  1,  0,  0],
    
    # ── Energy [kg·m²/s²] ─────────────────────────────────────────────────
    'E':      [2, -2,  1,  0,  0],
    'E_n':    [2, -2,  1,  0,  0],
    'K':      [2, -2,  1,  0,  0],
    'U':      [2, -2,  1,  0,  0],
    
    # ── Power [kg·m²/s³] ──────────────────────────────────────────────────
    'P':      [2, -3,  1,  0,  0],
    'Pwr':    [2, -3,  1,  0,  0],
    
    # ── Momentum [kg·m/s] ─────────────────────────────────────────────────
    'p':      [1, -1,  1,  0,  0],
    
    # ── Angular momentum [kg·m²/s] ────────────────────────────────────────
    'h':      [2, -1,  1,  0,  0],
    'hbar':   [2, -1,  1,  0,  0],
    'L':      [2, -1,  1,  0,  0],
    'Jz':     [2, -1,  1,  0,  0],
    
    # ── Torque [kg·m²/s²] ─────────────────────────────────────────────────
    'tau':    [2, -2,  1,  0,  0],
    
    # ── Temperature [K] ───────────────────────────────────────────────────
    'T':      [0,  0,  0,  1,  0],
    'T1':     [0,  0,  0,  1,  0],
    'T2':     [0,  0,  0,  1,  0],
    
    # ── Boltzmann constant [kg·m²/(s²·K)] ─────────────────────────────────
    'kb':     [2, -2,  1, -1,  0],
    
    # ── Gravitational constant [m³/(kg·s²)] ───────────────────────────────
    'G':      [3, -2, -1,  0,  0],
    
    # ── Electric permitivity [m⁻³·kg⁻¹·s⁴·A²] ─────────────────────────────
    'epsilon':[1, -2,  1,  0, -2],
    'epsilon0':[1, -2,  1,  0, -2],
    
    # ── Electric permeability [kg·m/(s²·A²)] ──────────────────────────────
    'mu0':    [-3,  4, -1,  0,  2],
    
    # ── Electric charge [kg·m²/(s²·V)] ────────────────────────────────────
    'q':      [2, -2,  1,  0, -1],
    'q1':     [2, -2,  1,  0, -1],
    'q2':     [2, -2,  1,  0, -1],
    
    # ── Voltage [V] ───────────────────────────────────────────────────────
    'Ve':     [0,  0,  0,  0,  1],
    'Volt':   [0,  0,  0,  0,  1],
    
    # ── Resistance [kg·m²/(s³·A²)] ────────────────────────────────────────
    'R':      [-2,  3, -1,  0,  2],
    
    # ── Capacitance [s⁴·A²/(kg·m²)] ───────────────────────────────────────
    'C':      [2, -2,  1,  0, -2],
    'C_cap':  [2, -2,  1,  0, -2],
    
    # ── Current [A] ───────────────────────────────────────────────────────
    'I':      [2, -3,  1,  0, -1],
    'I_0':    [2, -3,  1,  0, -1],
    'I_c':    [2, -3,  1,  0, -1],
    
    # ── Light intensity [kg/s³] ───────────────────────────────────────────
    'Int':    [0, -3,  1,  0,  0],
    'Int_0':  [0, -3,  1,  0,  0],
    'I1':     [0, -3,  1,  0,  0],
    'I2':     [0, -3,  1,  0,  0],
    'I_rad':  [0, -3,  1,  0,  0],
    'L_rad':  [0, -2,  1,  0,  0],
    
    # ── Electric field [V/m] ──────────────────────────────────────────────
    'Ef':     [-1,  0,  0,  0,  1],
    
    # ── Magnetic field [kg/(s²·A)] ────────────────────────────────────────
    'B':      [-2,  1,  0,  0,  1],
    'Bx':     [-2,  1,  0,  0,  1],
    'By':     [-2,  1,  0,  0,  1],
    'Bz':     [-2,  1,  0,  0,  1],
    'H':      [-2,  1,  0,  0,  1],  # Magnetic field (most common in Feynman)
    
    # ── Hubble constant [1/s] ─────────────────────────────────────────────
    'H_G':    [0, -1,  0,  0,  0],
    'H_hub':  [0, -1,  0,  0,  0],
    
    # ── Frequency [1/s] ───────────────────────────────────────────────────
    'omega':  [0, -1,  0,  0,  0],
    'omega0': [0, -1,  0,  0,  0],
    'nu':     [0, -1,  0,  0,  0],
    
    # ── Wave number [1/m] ─────────────────────────────────────────────────
    'k':      [-1,  0,  0,  0,  0],
    
    # ── Spring constant [kg/s²] ───────────────────────────────────────────
    'kspring':[0, -2,  1,  0,  0],
    'k_spring':[0, -2,  1,  0,  0],
    
    # ── Density [kg/m³] ───────────────────────────────────────────────────
    'rho':    [-3,  0,  1,  0,  0],
    'rho0':   [-3,  0,  1,  0,  0],
    'rho_0':  [-3,  0,  1,  0,  0],
    'n_rho':  [-3,  0,  0,  0,  0],
    'n_density':[-3, 0,  0,  0,  0],
    
    # ── Volume charge density [A·s/m³] ────────────────────────────────────
    'rho_c':  [-1, -2,  1,  0, -1],
    'rho_c_0':[-1, -2,  1,  0, -1],
    
    # ── Pressure [kg/(m·s²)] ──────────────────────────────────────────────
    'pF':     [-1, -2,  1,  0,  0],
    'pr':     [-1, -2,  1,  0,  0],
    
    # ── Area [m²] ─────────────────────────────────────────────────────────
    'A':      [2,  0,  0,  0,  0],
    
    # ── Volume [m³] ───────────────────────────────────────────────────────
    'V':      [3,  0,  0,  0,  0],
    'V_vol':  [3,  0,  0,  0,  0],
    
    # ── Electric dipole moment [A·s·m] ────────────────────────────────────
    'pd':     [3, -2,  1,  0, -1],
    'p_d':    [3, -2,  1,  0, -1],
    
    # ── Inductance [kg·m²/(s²·A²)] ────────────────────────────────────────
    'Lind':   [-2,  4, -1,  0,  2],
    'L_ind':  [-2,  4, -1,  0,  2],
    
    # ── Diffusion coefficient [m²/s] ──────────────────────────────────────
    'D':      [2, -1,  0,  0,  0],
    
    # ── Young modulus [kg/(m·s²)] ─────────────────────────────────────────
    'Y':      [-1, -2,  1,  0,  0],
    
    # ── Shear modulus [kg/(m·s²)] ─────────────────────────────────────────
    'mu_S':   [-1, -2,  1,  0,  0],
    
    # ── Stefan-Boltzmann constant [kg/(s³·K⁴)] ────────────────────────────
    'sigma_stefan': [0, -3,  1, -4,  0],
    
    # ── Surface charge density [A·s/m²] ───────────────────────────────────
    'sigma_charge': [0, -2,  1,  0, -1],
    'sigma_den':    [0, -2,  1,  0, -1],
    
    # ── Magnetisation [A/m] ───────────────────────────────────────────────
    'M':      [1, -3,  1,  0, -1],
    
    # ── Polarization [A·s/m²] ─────────────────────────────────────────────
    'Pol':    [0, -2,  1,  0, -1],
    
    # ── Mobility [m²/(V·s)] ───────────────────────────────────────────────
    'mob':    [0,  1, -1,  0,  0],
    
    # ── Magnetic moment [A·m²] ────────────────────────────────────────────
    'mom':    [4, -3,  1,  0, -1],
    
    # ── Drift velocity constant [kg/s] ────────────────────────────────────
    'mu_drift':[0, -1,  1,  0,  0],
    
    # ── Gravitational coupling [kg·m³/s²] ─────────────────────────────────
    'k_G':    [3, -2,  1,  0,  0],
    'kG':     [3, -2,  1,  0,  0],
    
    # ── Thermal conductivity [kg·m/(s³·K)] ────────────────────────────────
    'kappa':  [1, -3,  1, -1,  0],

    # ── Missing AIF variables ─────────────────────────────────────────────
    'omega_0':[0, -1,  0,  0,  0],
    'n_0':    [0,  0,  0,  0,  0],
    'V1':     [3,  0,  0,  0,  0],
    'V2':     [3,  0,  0,  0,  0],
    'A_vec':  [-1, 1,  0,  0,  1],
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