"""
ODEFormer-style Inference Pipeline.

1. Diversity Pool Sampling: generate N candidates using temperature-controlled sampling.
2. Skeleton deduplication: SymPy-based structural collapse.
3. BFGS Budgeting: optimization with strict iteration limit.
4. Parallel Execution: parallelized fitting across CPU cores.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sympy
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import warnings
import yaml
from pathlib import Path

from data.tokenizer import (
    IDX2TOKEN, TOKEN2IDX, BOS_IDX, EOS_IDX, PAD_IDX,
    MAX_SEQ_LEN, decode_formula, rpn_to_sympy
)

# ── Load Config Defaults ──────────────────────────────────────────────────────
def _load_default_config():
    try:
        config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            return full_cfg.get("inference", {})
    except Exception:
        return {}

_INF_CFG = _load_default_config()

@torch.no_grad()
def diversity_pool_sample(
    model, 
    z_context: torch.Tensor, 
    unit_idx: torch.Tensor,
    pool_size: int = _INF_CFG.get("pool_size", 50), 
    temperature: float = _INF_CFG.get("temperature", 0.1), 
    max_len: int = _INF_CFG.get("max_len", 45),
    device: str = 'cuda'
) -> List[List[int]]:
    """
    Generate a pool of N diverse candidates using temperature sampling.
    Uses the model's constrained generate method.
    """
    # Expand context to batch size
    z_exp = z_context.expand(pool_size, -1)
    u_exp = unit_idx.expand(pool_size, -1, -1)
    
    # Generate batch [pool_size, seq_len]
    generated = model.generate(
        z_context=z_exp,
        unit_idx=u_exp,
        max_len=max_len,
        greedy=False,
        temperature=temperature
    )
    
    return generated.cpu().tolist()

def skeletonize(token_ids: List[int]) -> Optional[str]:
    """
    Convert RPN tokens -> SymPy skeleton.
    Replaces all 'c1'..'c5' and numeric literals with generic CONST symbols.
    Each unique constant position gets a unique placeholder (CONST1, CONST2, ...)
    to preserve structural distinction during deduplication.
    """
    try:
        tokens = decode_formula(token_ids)
        expr = rpn_to_sympy(tokens)

        # Collect all constant placeholders (c1, c2, ...) and numeric literals
        free_symbols = {str(s) for s in expr.free_symbols}
        const_placeholders = sorted([s for s in free_symbols if s.startswith('c')])

        # Replace c-tokens with unique CONST placeholders
        skel_expr = expr
        for i, cp in enumerate(const_placeholders):
            skel_expr = skel_expr.subs(cp, sympy.Symbol(f'CONST{i}'))

        # Replace numeric literals with unique CONST placeholders
        # Sort by string representation for deterministic ordering
        numbers = sorted(list(skel_expr.atoms(sympy.Number)), key=str)
        for i, atom in enumerate(numbers):
            skel_expr = skel_expr.subs(atom, sympy.Symbol(f'CONST_NUM{i}'))

        return str(sympy.simplify(skel_expr))
    except Exception:
        return None

def fit_and_score(
    skeleton_str: str,
    tokens: List[str],
    X: np.ndarray,
    y_true: np.ndarray,
    var_names: List[str],
    max_iter: int = _INF_CFG.get("max_iter", 100),
    n_restarts: int = _INF_CFG.get("n_restarts", 5)
) -> Dict:
    """
    Fit constants in a skeleton using BFGS with strict budgeting on FULL data.
    Handles CONST{i} and CONST_NUM{i} placeholders from skeletonize.
    
    Uses multi-restart BFGS with diverse initializations for better convergence.
    """
    try:
        # Define local symbols for parsing
        local_dict = {name: sympy.Symbol(name) for name in var_names}
        # Pre-define common CONST placeholders for sympify
        for i in range(20):
            local_dict[f'CONST{i}'] = sympy.Symbol(f'CONST{i}')
            local_dict[f'CONST_NUM{i}'] = sympy.Symbol(f'CONST_NUM{i}')

        expr = sympy.sympify(skeleton_str, locals=local_dict)

        # Replace all CONST{i} and CONST_NUM{i} with unique c1, c2, ...
        const_count = 0
        def next_c():
            nonlocal const_count
            const_count += 1
            return sympy.Symbol(f'c{const_count}')

        # Replace CONST{i} placeholders (from c1, c2, ... in tokens)
        final_expr = expr
        for i in range(20):
            const_sym = sympy.Symbol(f'CONST{i}')
            if const_sym in final_expr.free_symbols:
                final_expr = final_expr.subs(const_sym, next_c())

        # Replace CONST_NUM{i} placeholders (from numeric literals)
        for i in range(20):
            const_num_sym = sympy.Symbol(f'CONST_NUM{i}')
            if const_num_sym in final_expr.free_symbols:
                final_expr = final_expr.subs(const_num_sym, next_c())

        const_symbols = sorted(list(final_expr.free_symbols - set(local_dict.values())), key=str)

        if not const_symbols:
            # Evaluate directly
            f = sympy.lambdify([sympy.Symbol(v) for v in var_names], final_expr, 'numpy')
            y_pred = f(*X.T)
            r2 = r2_score(y_true, y_pred)
            return {'expression': str(final_expr), 'r2': r2, 'tokens': tokens}

        # BFGS Setup - Create lambdified function ONCE for numerical stability
        var_syms = [sympy.Symbol(v) for v in var_names]
        all_args = var_syms + const_symbols
        try:
            f_lambdified = sympy.lambdify(all_args, final_expr, 'numpy')
        except Exception:
            return {'expression': str(final_expr), 'r2': -np.inf, 'tokens': tokens}

        def objective(params):
            try:
                args = list(X.T) + list(params)
                y_p = f_lambdified(*args)
                if not np.all(np.isfinite(y_p)): return 1e12
                return float(np.mean((y_true - y_p) ** 2))
            except Exception:
                return 1e12

        # Multi-restart BFGS with diverse initializations
        best_r2 = -np.inf
        best_params = None
        iters_per_restart = max(10, max_iter // n_restarts)

        for restart in range(n_restarts):
            # Diverse initialization strategies
            if restart == 0:
                x0 = np.ones(len(const_symbols))  # Start with 1s
            elif restart == 1:
                x0 = np.zeros(len(const_symbols)) + 0.1  # Small values
            elif restart == 2:
                x0 = np.random.uniform(0.1, 2.0, len(const_symbols))  # Random [0.1, 2]
            elif restart == 3:
                x0 = np.random.uniform(0.5, 5.0, len(const_symbols))  # Random [0.5, 5]
            else:
                x0 = np.random.uniform(-2.0, 10.0, len(const_symbols))  # Wide range

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': iters_per_restart, 'ftol': 1e-10, 'gtol': 1e-8})

            if np.isfinite(res.fun):
                # Calculate R2 for this restart
                try:
                    args = list(X.T) + list(res.x)
                    y_p = f_lambdified(*args)
                    if np.all(np.isfinite(y_p)):
                        r2 = r2_score(y_true, y_p)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = res.x
                except Exception:
                    pass

        if best_params is None:
            best_params = np.ones(len(const_symbols))
            best_r2 = -np.inf

        # Final substitution
        best_subs = dict(zip(const_symbols, best_params))
        optimized_expr = final_expr.subs(best_subs)
        f_final = sympy.lambdify([sympy.Symbol(v) for v in var_names], optimized_expr, 'numpy')
        y_final = f_final(*X.T)
        r2 = r2_score(y_true, y_final)

        return {
            'expression': str(optimized_expr),
            'r2': r2,
            'tokens': tokens,
            'fitted_params': best_params.tolist()
        }

    except Exception as e:
        return {
            'expression': skeleton_str,
            'r2': -np.inf,
            'tokens': tokens,
            'fitted_params': [],
            'error': str(e)
        }

def odeformer_inference(
    model,
    X_bits: torch.Tensor,
    unit_idx: torch.Tensor,
    var_mask: torch.Tensor,
    X_data: np.ndarray,
    y_data: np.ndarray,
    var_names: List[str],
    pool_size: int = _INF_CFG.get("pool_size", 50),
    temperature: float = _INF_CFG.get("temperature", 0.1),
    top_k: int = _INF_CFG.get("top_k", 5),
    max_iter: int = _INF_CFG.get("max_iter", 100),
    n_workers: int = _INF_CFG.get("n_workers", 8),
    n_restarts: int = _INF_CFG.get("n_restarts", 5)
) -> List[Dict]:
    """
    Full ODEFormer Inference Pipeline using Diversity Pool Sampling.
    """
    device = X_bits.device
    # 1. Encode
    with torch.no_grad():
        z_context = model.encode(X_bits, unit_idx, var_mask)

    # 2. Diversity Pool Generation
    token_sequences = diversity_pool_sample(
        model, z_context, unit_idx,
        pool_size=pool_size,
        temperature=temperature,
        device=device
    )

    # 3. Skeletonize and Deduplicate
    unique_skeletons = {} # skel -> tokens
    for token_ids in token_sequences:
        skel = skeletonize(token_ids)
        if skel and skel not in unique_skeletons:
            tokens = decode_formula(token_ids)
            unique_skeletons[skel] = tokens

    # 4. Parallel Fitting (on FULL data)
    results = []
    skeletons = list(unique_skeletons.keys())

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(fit_and_score, skel, unique_skeletons[skel], X_data, y_data, var_names, max_iter, n_restarts)
            for skel in skeletons
        ]

        for future in futures:
            res = future.result()
            results.append(res)

    # 5. Rank by R2 on full data
    results.sort(key=lambda x: -x['r2'])
    
    return results[:top_k]
