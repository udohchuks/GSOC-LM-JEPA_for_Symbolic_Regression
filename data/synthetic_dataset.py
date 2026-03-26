"""
Physics-Informed Synthetic Data Generator for LLM-JEPA.

Generates dimensionally valid synthetic equations for pretraining.
Key differentiator from NeSymReS/SNIP: expression trees respect
dimensional homogeneity throughout construction.

Pipeline per equation:
    1. Sample variable types from physics domain pools
    2. Build expression tree with unit propagation constraints
    3. Apply affine transformation for diversity (Kamienny et al.)
    4. Evaluate on sampled data points
    5. Add Gaussian noise to output
    6. Encode with IEEE-754 and compute all targets

Expected yield: ~10% of sampled trees pass unit consistency.
Run offline, parallelised across CPU cores.
"""

from __future__ import annotations
import numpy as np
import sympy
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
import random
import warnings
warnings.filterwarnings('ignore')

from data.tokenizer import (
    BINARY_TOKENS, UNARY_TOKENS, CONSTANT_TOKENS,
    ARITY, MAX_SEQ_LEN, encode_formula,
)
from data.unit_table import (
    get_unit_vector, get_unit_matrix,
    unit_to_class_indices, N_UNIT_DIMS, DIMENSIONLESS,
)
from data.utils import (
    to_ieee754_16bit,
    add_gaussian_noise,
    compute_unit_targets,
    unit_targets_to_class_indices,
)

# ── Physics domain variable pools ─────────────────────────────────────────────
# Each entry: (unit_table_name, sympy_symbol_name, (sampling_low, sampling_high))
# unit_table_name: must match a key in UNIT_TABLE
# sympy_symbol_name: used to build the SymPy expression
# sampling range: physically motivated range for this variable type

DOMAIN_POOLS: Dict[str, List[Tuple[str, str, Tuple[float, float]]]] = {

    'mechanics': [
        ('m1',  'mass1',  (0.5, 5.0)),
        ('m2',  'mass2',  (0.5, 5.0)),
        ('x1',  'pos_x1', (1.0, 5.0)),
        ('x2',  'pos_x2', (1.0, 5.0)),
        ('y1',  'pos_y1', (1.0, 5.0)),
        ('y2',  'pos_y2', (1.0, 5.0)),
        ('v',   'vel',    (0.1, 3.0)),
        ('v1',  'vel1',   (0.1, 3.0)),
        ('G',   'G_grav', (1.0, 3.0)),
        ('t',   'time',   (0.1, 5.0)),
        ('F',   'force',  (1.0, 5.0)),
        ('E',   'energy', (1.0, 10.0)),
        ('kspring', 'kspring', (0.5, 3.0)),
    ],

    'electromagnetism': [
        ('q1',  'charge1', (1.0, 5.0)),
        ('q2',  'charge2', (1.0, 5.0)),
        ('r',   'dist',    (0.5, 5.0)),
        ('Ef',  'efield',  (1.0, 5.0)),
        ('B',   'bfield',  (1.0, 5.0)),
        ('Ve',  'voltage', (1.0, 5.0)),
    ],

    'thermodynamics': [
        ('T',   'temp',    (200.0, 500.0)),
        ('T1',  'temp1',   (200.0, 500.0)),
        ('T2',  'temp2',   (200.0, 500.0)),
        ('kb',  'kb',      (1.0,   3.0)),
        ('pF',  'press',   (1.0,   5.0)),
    ],

    'dimensionless': [
        ('theta',  'theta',  (0.1, 3.0)),
        ('theta1', 'theta1', (0.1, 3.0)),
        ('theta2', 'theta2', (0.1, 3.0)),
        ('alpha',  'alpha',  (0.1, 3.0)),
        ('n',      'n_idx',  (1.0, 3.0)),
    ],
}

# Flat pool for mixed-domain sampling
ALL_VARIABLES = [v for pool in DOMAIN_POOLS.values() for v in pool]


# ── Expression tree node ───────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """
    One node in a symbolic expression tree.

    token:    the operator or variable name at this node
    children: child nodes (empty for leaf nodes)
    units:    the unit vector this subtree produces

   
    During top-down tree construction we need to know what units
    each subtree produces in order to decide which operators are
    valid at the parent node. Storing units avoids recomputing
    them by traversing the whole subtree every time.
    """
    token:    str
    children: List['TreeNode'] = field(default_factory=list)
    units:    List[int] = field(default_factory=lambda: [0] * N_UNIT_DIMS)


# ── Unit propagation ───────────────────────────────────────────────────────────

def propagate_units(
    operator: str,
    child_units: List[List[int]]
) -> Optional[List[int]]:
    """
    Given an operator and its children's unit vectors,
    compute the output unit vector.

    Returns None if the operation is dimensionally invalid.
    This is what constrains the tree builder.

    Args:
        operator:    token string of the operator
        child_units: list of unit vectors from child nodes

    Returns:
        output unit vector, or None if invalid
    """
    if operator in ('+', '-'):
        if len(child_units) < 2:
            return None
        # Addition requires identical units on both sides
        if child_units[0] != child_units[1]:
            return None
        return child_units[0][:]

    elif operator == '*':
        if len(child_units) < 2:
            return None
        # Multiplication adds unit exponents
        return [child_units[0][i] + child_units[1][i]
                for i in range(N_UNIT_DIMS)]

    elif operator == '/':
        if len(child_units) < 2:
            return None
        # Division subtracts unit exponents
        return [child_units[0][i] - child_units[1][i]
                for i in range(N_UNIT_DIMS)]

    elif operator == 'sqrt':
        if len(child_units) < 1:
            return None
        # Only valid when all exponents are even
        # sqrt([2,0,0,0,0]) = [1,0,0,0,0]  (sqrt of m² = m) ✓
        # sqrt([1,0,0,0,0]) = [0.5,...] which is not integer  ✗
        if any(e % 2 != 0 for e in child_units[0]):
            return None
        return [e // 2 for e in child_units[0]]

    elif operator == 'sq':
        if len(child_units) < 1:
            return None
        return [e * 2 for e in child_units[0]]

    elif operator == 'inv':
        if len(child_units) < 1:
            return None
        return [-e for e in child_units[0]]

    elif operator in ('neg', 'abs'):
        if len(child_units) < 1:
            return None
        return child_units[0][:]

    elif operator in ('exp', 'log', 'sin', 'cos', 'tan',
                      'arcsin', 'arccos', 'arctan'):
        if len(child_units) < 1:
            return None
        # Transcendental functions require dimensionless argument
        if child_units[0] != [0] * N_UNIT_DIMS:
            return None
        return [0] * N_UNIT_DIMS

    return None # leaf or invalid operation

# ── Tree builder ───────────────────────────────────────────────────────────────

# Depth distribution: how often to generate trees of each depth
# Stratified to cover the range of Feynman equation complexities
DEPTH_WEIGHTS = [0.10, 0.20, 0.25, 0.25, 0.15, 0.05]  # depths 1-6

# Variable count distribution: matches AIF dataset distribution
N_VARS_WEIGHTS = {
    1: 0.10, 2: 0.20, 3: 0.20, 4: 0.15,
    5: 0.15, 6: 0.10, 7: 0.05, 8: 0.03, 9: 0.02
}

class PhysicsTreeBuilder:
    """
    Builds random expression trees respecting dimensional homogeneity.

    Top-down construction: at each node, only operators that produce
    valid units given available child units are considered.

    Why top-down not bottom-up?
    Bottom-up: build leaves first, combine upward.
        Problem: hard to control tree depth and shape.

    Top-down: decide operator first, then recurse into children.
        Problem: you do not know child units before recursing.
        Solution: recurse first, then check if the resulting
                  child units are compatible with each operator.
                  If not, try a different operator or fall back to leaf.
    """
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def sample(
        self,
    ) -> Optional[Tuple[TreeNode, List[Tuple[str, str, Tuple[float, float]]]]]:
        """
        Sample one random physics-informed expression tree.

        Returns:
            (root_node, var_pool) where var_pool describes the
            variables used, or None if sampling failed.
        """
        # Sample number of variables
        n_vars = random.choices(
            list(N_VARS_WEIGHTS.keys()),
            weights=list(N_VARS_WEIGHTS.values())
        )[0]

        # Sample target depth
        target_depth = random.choices(
            range(1, len(DEPTH_WEIGHTS) + 1),
            weights=DEPTH_WEIGHTS
        )[0]

        # Sample variable types from a domain (or mixed)
        domain = random.choice(list(DOMAIN_POOLS.keys()) + ['mixed'])
        if domain == 'mixed':
            var_pool = random.choices(ALL_VARIABLES, k=n_vars)
        else:
            var_pool = random.choices(DOMAIN_POOLS[domain], k=n_vars)
        
        # Build the tree
        root = self._build(depth=0, target_depth=target_depth,
                           var_pool=var_pool)
        if root is None:
            return None
        
        return root, var_pool
    
    def _build(
        self,
        depth:        int,
        target_depth: int,
        var_pool:     List,
    ) -> Optional[TreeNode]:
        """
        Recursively build one tree node.

        At or beyond target_depth: always return a leaf.
        Otherwise: probabilistically choose operator vs leaf,
        then find a valid operator for the resulting children.
        """
         # Force leaf at max depth
        if depth >= target_depth:
            return self._leaf(var_pool)
        
        # Probability of choosing a leaf increases with depth
        # At depth 0: 10% leaf, 90% operator
        # At depth target_depth-1: 80% leaf, 20% operator
        leaf_prob = depth / (target_depth + 1)
        if random.random() < leaf_prob:
            return self._leaf(var_pool)
        
        # Try a binary operator (60% of operator nodes)
        if random.random() < 0.6:
            return self._binary_node(depth, target_depth, var_pool)
        else:
            return self._unary_node(depth, target_depth, var_pool)
    
    def _binary_node(self, depth, target_depth, var_pool):
        """Try to build a binary operator node."""
        left  = self._build(depth + 1, target_depth, var_pool)
        right = self._build(depth + 1, target_depth, var_pool)

        if left is None or right is None:
            return self._leaf(var_pool)
        
        # Try binary operators in random order
        ops = random.sample(BINARY_TOKENS, len(BINARY_TOKENS))
        for op in ops:
            out_units = propagate_units(op, [left.units, right.units])
            if out_units is not None:
                node = TreeNode(token=op, units=out_units)
                node.children = [left, right]
                return node
        # No valid binary operator found — return left child
        return left
    
    def _unary_node(self, depth, target_depth, var_pool):
        """Try to build a unary operator node."""

        child = self._build(depth + 1, target_depth, var_pool)
        if child is None:
            return self._leaf(var_pool)

        # For transcendental functions, child must contain a variable
        # Applying sin/cos/exp to a pure constant just gives another constant
        TRANSCENDENTAL = {'sin', 'cos', 'tan', 'exp', 'log',
                      'arcsin', 'arccos', 'arctan'}
        
        def has_variable(node: TreeNode) -> bool:
            if node.token.startswith('x'):
                return True
            return any(has_variable(c) for c in node.children)
        
        child_has_var = has_variable(child)

        # Try unary operators in random order
        ops = random.sample(UNARY_TOKENS, len(UNARY_TOKENS))
        for op in ops:
            if op == 'abs':
                continue

            if op in TRANSCENDENTAL and not child_has_var:
                continue

            out_units = propagate_units(op, [child.units])
            if out_units is not None:
                node = TreeNode(token=op, units=out_units)
                node.children = [child]
                return node
        
        return child
    
    def _leaf(self, var_pool: List) -> TreeNode:
        """Sample a leaf node: 80% variable, 20% dimensionless constant."""
        if random.random() < 0.8:
            # Sample a variable from the pool
            entry    = random.choice(var_pool)
            unit_name = entry[0]
            units    = get_unit_vector(unit_name, warn_unknown=False)
            # Map to x1...xN token based on position in pool
            idx = var_pool.index(entry)
            tok = f'x{idx + 1}'
            return TreeNode(token=tok, units=units[:])
        else:
            # Dimensionless constant
            tok = random.choice(['1', '2', 'pi'])
            return TreeNode(token=tok, units=[0] * N_UNIT_DIMS)


# ── Tree conversion ────────────────────────────────────────────────────────────

# SymPy function mapping for unary operators
UNARY_TO_SYMPY = {
    'sqrt':   sympy.sqrt,
    'sq':     lambda x: x ** 2,
    'exp':    sympy.exp,
    'log':    sympy.log,
    'sin':    sympy.sin,
    'cos':    sympy.cos,
    'tan':    sympy.tan,
    'arcsin': sympy.asin,
    'arccos': sympy.acos,
    'arctan': sympy.atan,
    'inv':    lambda x: sympy.Integer(1) / x,
    'neg':    lambda x: -x,
    'abs':    sympy.Abs,
}

BINARY_TO_SYMPY = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
}

CONSTANT_TO_SYMPY = {
    '0': sympy.Integer(0),
    '1': sympy.Integer(1),
    '2': sympy.Integer(2),
    '3': sympy.Integer(3),
    'pi': sympy.pi,
    'e':  sympy.E,
}

def tree_to_sympy(
    node:     TreeNode,
    sym_vars: Dict[str, sympy.Symbol],
) -> Optional[sympy.Expr]:
    """
    Convert expression tree to SymPy expression.

    Args:
        node:     root TreeNode
        sym_vars: mapping from token (x1, x2...) to SymPy Symbol

    Returns:
        SymPy expression, or None if conversion fails.
    """
    tok = node.token
    # Leaf: variable
    if tok in sym_vars:
        return sym_vars[tok]
    
    # Leaf: constant
    if tok in CONSTANT_TO_SYMPY:
        return CONSTANT_TO_SYMPY[tok]
    
    if ARITY.get(tok) == 1 and tok in UNARY_TO_SYMPY:
        if not node.children:
            return None
        child_expr = tree_to_sympy(node.children[0], sym_vars)
        if child_expr is None:
            return None
        try:
            result = UNARY_TO_SYMPY[tok](child_expr)
            if result.has(sympy.zoo, sympy.nan, sympy.oo, sympy.I):
                return None
            return result
        except Exception:
            return None
    
    if ARITY.get(tok) == 2 and tok in BINARY_TO_SYMPY:
        if len(node.children) < 2:
            return None
        left  = tree_to_sympy(node.children[0], sym_vars)
        right = tree_to_sympy(node.children[1], sym_vars)
        if left is None or right is None:
            return None
        try:
            result = BINARY_TO_SYMPY[tok](left, right)
            if result.has(sympy.zoo, sympy.nan, sympy.oo, sympy.I):
                return None
            return result
        except Exception:
            return None
    
    return None

def tree_to_rpn(node: TreeNode) -> List[str]:
    """
    Convert tree to RPN token list via postorder traversal.
    Children first, then current node.
    """
    tokens = []
    for child in node.children:
        tokens.extend(tree_to_rpn(child))
    tokens.append(node.token)
    return tokens

# ── Affine transformation ──────────────────────────────────────────────────────

def apply_affine_transform(
    var_pool: List[Tuple[str, str, Tuple[float, float]]],
    sym_vars: Dict[str, sympy.Symbol],
    expr:     sympy.Expr,
) -> Tuple[sympy.Expr, List[Tuple[str, str, Tuple[float, float]]]]:
    """
    Apply affine transformation to each variable: replace xd with a*xd + b.

    Following Kamienny et al. (2022) — diversifies the training distribution
    so the model never sees the exact same function twice.

    Rules:
        Dimensionless variables: both scaling (a) and shifting (b) allowed
        Dimensioned variables:   scaling (a) only
                                 Shifting adds a constant with units,
                                 e.g. x + 3 where x is in meters means
                                 "3 meters" which changes the physics

    Args:
        var_pool: variable metadata list
        sym_vars: token → SymPy Symbol mapping
        expr:     current SymPy expression

    Returns:
        (transformed_expr, updated_var_pool_with_new_ranges)
    """
    new_expr     = expr
    new_var_pool = list(var_pool)

    for i, entry in enumerate(var_pool):
        unit_name, sym_name, (v_low, v_high) = entry
        tok     = f'x{i + 1}'
        sym     = sym_vars[tok]
        units   = get_unit_vector(unit_name, warn_unknown=False)
        is_dim  = (units == DIMENSIONLESS)

        # Scale factor: always applied
        a = np.random.uniform(0.5, 2.0)

        # Shift: only for dimensionless variables
        b = np.random.uniform(-1.0, 1.0) if is_dim else 0.0

        # Replace sym with a*sym + b in the expression
        new_expr = new_expr.subs(sym, a * sym + b)

        # Update the sampling range to reflect the transformation
        # If original x ∈ [v_low, v_high],
        # then a*x + b ∈ [a*v_low+b, a*v_high+b]
        new_low  = a * v_low  + b
        new_high = a * v_high + b
        if new_low > new_high:
            new_low, new_high = new_high, new_low

        new_var_pool[i] = (unit_name, sym_name, (new_low, new_high))

    return new_expr, new_var_pool

# ── Synthetic equation ─────────────────────────────────────────────────────────

@dataclass
class SyntheticEquation:
    """
    One generated synthetic equation with all precomputed fields.
    Mirrors PreprocessedEquation from aif_dataset.py so both can be
    used interchangeably in the training loop.
    """
    var_names:         List[str]           # unit_table names
    expr_str:          str                 # SymPy canonical string
    rpn_tokens:        List[str]           # RPN token list
    token_ids:         np.ndarray          # [MAX_SEQ_LEN]
    X_bits:            np.ndarray          # [N, n_vars, 16] IEEE-754
    y_noisy:           np.ndarray          # [N] with noise
    unit_matrix_idx:   np.ndarray          # [n_vars, 5] class indices
    unit_targets_idx:  np.ndarray          # [MAX_SEQ_LEN, 5]
    n_vars:            int
    epsilon:           float               # noise level used

def generate_one_equation(
    builder:        PhysicsTreeBuilder,
    n_data_points:  int = 1000,
    max_attempts:   int = 50,
) -> Optional[SyntheticEquation]:
    """
    Generate one valid synthetic physics equation.

    Attempts up to max_attempts times.
    Expected success rate ~10% due to unit consistency filtering.

    Args:
        builder:       PhysicsTreeBuilder instance
        n_data_points: data rows to generate
        max_attempts:  retry budget per equation

    Returns:
        SyntheticEquation or None if all attempts fail.
    """
    for _ in range(max_attempts):

        # ── Sample tree ───────────────────────────────────────────────────
        result = builder.sample()
        if result is None:
            continue
        
        root, var_pool = result
        n_vars = len(var_pool)

        # ── Build SymPy symbol dict ───────────────────────────────────────
        sym_vars: Dict[str, sympy.Symbol] = {
            f'x{i+1}': sympy.Symbol(var_pool[i][1])
            for i in range(n_vars)
        }

        # ── Convert tree to SymPy ─────────────────────────────────────────
        expr = tree_to_sympy(root, sym_vars)
        if expr is None:
            continue
        
        try:
            expr = sympy.simplify(expr)
        except Exception:
            continue
        
        # Skip trivial: constant or zero
        if expr.is_number:
            continue
        
        # ── Apply affine transformation ───────────────────────────────────
        try:
            expr, var_pool = apply_affine_transform(
                var_pool, sym_vars, expr
            )
        except Exception:
            continue
        
        # ── Convert to RPN ────────────────────────────────────────────────
        # Map SymPy symbols back to x1...xN tokens
        sym_to_token = {
            sym_vars[f'x{i+1}']: f'x{i+1}'
            for i in range(n_vars)
        }
        try:
            from data.tokenizer import _sympy_to_rpn
            rpn_tokens = _sympy_to_rpn(expr, {
                str(v): t for v, t in sym_to_token.items()
            })
        except Exception:
            continue
        
        # Skip if too long for our sequence length
        if len(rpn_tokens) == 0 or len(rpn_tokens) > MAX_SEQ_LEN - 2:
            continue
        
        # After getting rpn_tokens, check constant count
        n_consts = sum(1 for t in rpn_tokens if t.startswith('c'))
        if n_consts > 5:
            continue   # too many constants, discard this equation
        
        # ── Generate data points ──────────────────────────────────────────
        symbols = [sym_vars[f'x{i+1}'] for i in range(n_vars)]
        ranges  = [var_pool[i][2] for i in range(n_vars)]

        try:
            f_lambda = sympy.lambdify(symbols, expr, 'numpy')

            X = np.column_stack([
                np.random.uniform(lo, hi, n_data_points)
                for lo, hi in ranges
            ]).astype(np.float32)

            y = f_lambda(*[X[:, i] for i in range(n_vars)])

            # Force to array (lambdify may return scalar for constants)
            y = np.broadcast_to(
                np.asarray(y, dtype=np.float32), (n_data_points,)
            ).copy()

            # Reject NaN, Inf, or constant output
            if not np.isfinite(y).all():
                continue
            if not np.isfinite(X).all():
                continue
            if np.std(y) < 1e-10:
                continue   # constant output — nothing to learn

        except Exception:
            continue
        
        # ── Add Gaussian noise ────────────────────────────────────────────
        epsilon = float(np.random.choice([0.0, 0.0, 0.0, 1e-4, 1e-3, 1e-2]))
        y_noisy = add_gaussian_noise(y, epsilon=epsilon)

        # ── IEEE-754 encode ───────────────────────────────────────────────
        X_bits = to_ieee754_16bit(X)   # [N, n_vars, 16]

        # ── Unit matrix ───────────────────────────────────────────────────
        var_names       = [var_pool[i][0] for i in range(n_vars)]
        unit_matrix     = get_unit_matrix(var_names)
        unit_matrix_idx = unit_to_class_indices(unit_matrix)

        # ── Token IDs ─────────────────────────────────────────────────────
        token_ids = np.array(
            encode_formula(rpn_tokens, add_bos=True,
                           add_eos=True, pad_to=MAX_SEQ_LEN),
            dtype=np.int64
        )

        # ── Unit targets ──────────────────────────────────────────────────
        raw_targets = compute_unit_targets(rpn_tokens, var_names)
        padded = [[0]*5] + raw_targets + [[0]*5]
        while len(padded) < MAX_SEQ_LEN:
            padded.append([0]*5)
        padded = padded[:MAX_SEQ_LEN]
        unit_targets_idx = unit_targets_to_class_indices(padded)
        
        from data.tokenizer import rpn_to_sympy
        return SyntheticEquation(
            var_names=var_names,
            expr_str=str(rpn_to_sympy(rpn_tokens)),
            rpn_tokens=rpn_tokens,
            token_ids=token_ids,
            X_bits=X_bits,
            y_noisy=y_noisy,
            unit_matrix_idx=unit_matrix_idx,
            unit_targets_idx=unit_targets_idx,
            n_vars=n_vars,
            epsilon=epsilon,
        )

    return None   # all attempts failed

# ── PyTorch Dataset ───────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset, DataLoader
from data.aif_dataset import collate_fn   # reuse same collate function


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-generated synthetic equations.

    Designed to be identical in interface to AIFDataset so both
    can be used with the same training loop and collate_fn.
    """

    def __init__(
        self,
        equations:  List[SyntheticEquation],
        max_n_vars: int = 9,
    ):
        self.equations  = equations
        self.max_n_vars = max_n_vars

    def __len__(self) -> int:
        return len(self.equations)
    
    def __getitem__(self, idx: int) -> Dict:
        eq     = self.equations[idx]
        n_vars = eq.n_vars

        X_bits = eq.X_bits   # [N, n_vars, 16]

        # Pad variable dimension to max_n_vars
        pad_vars = self.max_n_vars - n_vars
        if pad_vars > 0:
            pad_x    = np.zeros(
                (X_bits.shape[0], pad_vars, 16), dtype=np.float32
            )
            X_bits   = np.concatenate([X_bits, pad_x], axis=1)
            pad_u    = np.full((pad_vars, 5), 4, dtype=np.int64)
            unit_idx = np.concatenate(
                [eq.unit_matrix_idx, pad_u], axis=0
            )
        else:
            unit_idx = eq.unit_matrix_idx

        var_mask = np.zeros(self.max_n_vars, dtype=np.float32)
        var_mask[:n_vars] = 1.0

        return {
            'X_bits':           torch.from_numpy(X_bits).float(),
            'unit_idx':         torch.from_numpy(unit_idx).long(),
            'var_mask':         torch.from_numpy(var_mask).float(),
            'n_vars':           torch.tensor(n_vars, dtype=torch.long),
            'token_ids':        torch.from_numpy(eq.token_ids).long(),
            'unit_targets_idx': torch.from_numpy(eq.unit_targets_idx).long(),
            'eq_id':            f'syn_{idx}',
            'formula_str':      eq.expr_str,
            'var_names':        eq.var_names,
        }


def build_synthetic_dataloader(
    n_equations:   int,
    batch_size:    int = 32,
    n_data_points: int = 1000,
    max_n_vars:    int = 9,
    num_workers:   int = 2,
    cache_path:    Optional[str] = None,
) -> DataLoader:
    """
    Build DataLoader for synthetic pretraining data.

    Generates equations on first call, caches to cache_path.
    Loads from cache on subsequent calls.

    Args:
        n_equations:    target number of unique equations to generate
        batch_size:     equations per batch
        n_data_points:  data rows per equation
        max_n_vars:     variable padding size
        num_workers:    DataLoader workers
        cache_path:     .pt file to cache generated equations

    Returns:
        DataLoader with same batch structure as AIF DataLoader.
    """
    from pathlib import Path

    # Load from cache if available
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached synthetic data from {cache_path}")
        equations = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(equations)} equations")
    else:
        print(f"Generating {n_equations} synthetic equations...")
        equations = _generate_corpus(n_equations, n_data_points)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(equations, cache_path)
            print(f"Cached to {cache_path}")
    dataset = SyntheticDataset(equations, max_n_vars=max_n_vars)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

def _generate_corpus(
    n_equations:   int,
    n_data_points: int = 1000,
    verbose:       bool = True,
) -> List[SyntheticEquation]:
    """
    Generate a corpus of physics-informed synthetic equations.
    Simple loop — parallelise with multiprocessing for large corpora.
    """
    builder    = PhysicsTreeBuilder(max_depth=6)
    equations  = []
    n_attempted = 0

    while len(equations) < n_equations:
        n_attempted += 1
        eq = generate_one_equation(builder, n_data_points=n_data_points)
        if eq is not None:
            equations.append(eq)

        if verbose and len(equations) % 100 == 0 and len(equations) > 0:
            rate = len(equations) / n_attempted * 100
            print(f"  {len(equations)}/{n_equations} | "
                  f"attempts: {n_attempted} | "
                  f"yield: {rate:.1f}%")

    if verbose:
        print(f"Done: {len(equations)} equations "
              f"from {n_attempted} attempts "
              f"({len(equations)/n_attempted*100:.1f}% yield)")
    return equations



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    # ── Test 1: Unit propagation ──────────────────────────────────────────
    # mass * mass = kg²
    result = propagate_units('*', [[0,0,1,0,0], [0,0,1,0,0]])
    assert result == [0, 0, 2, 0, 0], f"Got {result}"

    # length / length = dimensionless
    result = propagate_units('/', [[1,0,0,0,0], [1,0,0,0,0]])
    assert result == [0, 0, 0, 0, 0]

    # sin requires dimensionless
    result = propagate_units('sin', [[0,0,0,0,0]])
    assert result == [0, 0, 0, 0, 0]

    # sin of length is invalid
    result = propagate_units('sin', [[1,0,0,0,0]])
    assert result is None

    # sqrt of m² = m
    result = propagate_units('sqrt', [[2,0,0,0,0]])
    assert result == [1, 0, 0, 0, 0]

    # sqrt of m is invalid (odd exponent)
    result = propagate_units('sqrt', [[1,0,0,0,0]])
    assert result is None
    print('Unit propagation: OK')

    # ── Test 2: Tree builder ──────────────────────────────────────────────
    builder  = PhysicsTreeBuilder(max_depth=4)
    n_built  = sum(1 for _ in range(50) if builder.sample() is not None)
    print(f'Tree builder: {n_built}/50 trees built')
    assert n_built > 10

    # ── Test 3: Full equation generation ─────────────────────────────────
    n_success = 0
    for _ in range(20):
        eq = generate_one_equation(builder, n_data_points=200)
        if eq is not None:
            n_success += 1
            print(eq.expr_str)

    print(f'Equation generation: {n_success}/10 succeeded')
    assert n_success > 0, "No equations generated — check unit propagation"

    # Inspect last successful equation
    if eq is not None:
        print(f'  Formula: {eq.expr_str[:60]}')
        print(f'  n_vars:  {eq.n_vars}')
        print(f'  RPN:     {eq.rpn_tokens}')
        print(f'  X_bits:  {eq.X_bits.shape}')
        print(f'  epsilon: {eq.epsilon}')
        assert eq.X_bits.shape         == (200, eq.n_vars, 16)
        assert eq.token_ids.shape      == (MAX_SEQ_LEN,)
        assert eq.unit_targets_idx.shape == (MAX_SEQ_LEN, 5)

    # ── Test 4: Dataset __getitem__ ───────────────────────────────────────
    equations = [eq for _ in range(20)
                 for eq in [generate_one_equation(builder, 100)]
                 if eq is not None][:5]

    if equations:
        dataset = SyntheticDataset(equations, max_n_vars=9)
        item    = dataset[0]
        assert item['X_bits'].shape    == (100, 9, 16)
        assert item['var_mask'].shape  == (9,)
        assert item['token_ids'].shape == (MAX_SEQ_LEN,)
        print('Dataset __getitem__: OK')

        loader = DataLoader(dataset, batch_size=2,
                            collate_fn=collate_fn, num_workers=0)
        batch  = next(iter(loader))
        assert batch['X_bits'].shape   == (2, 100, 9, 16)
        assert batch['token_ids'].shape == (2, MAX_SEQ_LEN)
        print('DataLoader batch: OK')

    print('\nAll synthetic dataset tests passed.')    