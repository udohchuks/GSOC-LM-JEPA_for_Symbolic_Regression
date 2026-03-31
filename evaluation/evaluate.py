"""
Simplified Goldilocks Evaluation Suite for LLM-JEPA Symbolic Regression.

Metrics:
1. Numeric Precision (R²): Evaluated on full data.
2. Symbolic Accuracy (SA): Functional equivalence via random points.
3. Normalized Edit Distance (NED): Structural similarity via prefix trees.
"""

from __future__ import annotations
import time
import numpy as np
import sympy
import torch
from typing import List, Dict, Optional, Tuple

from data.tokenizer import decode_formula, is_valid_rpn, rpn_to_sympy
from evaluation.metrics import (
    calculate_node_count,
    calculate_r2,
    calculate_ned,
    verify_symbolic_accuracy,
    calculate_constant_recovery,
    _evaluate_expr,
)
from inference.beam_search import odeformer_inference
import yaml
from pathlib import Path

def _load_inference_config():
    """Helper to load default inference params from config."""
    try:
        config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            return full_cfg.get("inference", {})
    except:
        return {}

_INF_CFG = _load_inference_config()


# ── Main evaluation entry point ──────────────────────────────────────────────

def evaluate_dataset(
    model,
    dataset,
    device:     str   = 'cpu',
    n_restarts: int   = 3,
    n_candidates: int = _INF_CFG.get("pool_size", 50),
    temperature: float = _INF_CFG.get("temperature", 0.1),
    verbose:    bool  = True,
    inf_config: Optional[dict] = None,
) -> Dict:
    """
    Evaluate model on a full dataset (AIF evaluation set).
    Returns dict of aggregated Goldilocks metrics.
    """
    results = []

    for i, eq in enumerate(dataset.equations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluating {i+1}/{len(dataset.equations)}: {eq.eq_id}")

        try:
            result = _evaluate_one(
                model=model,
                eq=eq,
                device=device,
                n_restarts=n_restarts,
                n_candidates=n_candidates,
                temperature=temperature,
                inf_config=inf_config,
            )
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Failed {eq.eq_id}: {e}")
            results.append(_failed_result(eq))

    return aggregate_results(results)


def _failed_result(eq) -> dict:
    """Return a default failed result dict for an equation."""
    return {
        'eq_id': eq.eq_id,
        'n_vars': eq.n_vars,
        'r2': -np.inf,
        'symbolic_accuracy': False,
        'constant_recovery': 0.0,
        'ned': 1.0,
        'node_count': 0,
        'latency_s': 0.0,
        'predicted': None,
        'truth': eq.formula_str
    }


# ── Per-equation evaluation ──────────────────────────────────────────────────

def _evaluate_one(model, eq, device, n_restarts, n_candidates=_INF_CFG.get("pool_size", 50), temperature=_INF_CFG.get("temperature", 0.1), inf_config=None) -> dict:
    """Goldilocks evaluation for one equation."""
    model.eval()
    
    # 1. Prepare Data
    X_raw = _reconstruct_X(eq) 
    y = eq.y
    X_t, unit_idx, var_mask = _prepare_model_inputs(eq, model, device)

    # 2. ODEFormer Inference (Pool generation + Full-data fitting + R2 Ranking)
    t0 = time.perf_counter()
    inf_cfg = inf_config or {}
    ode_results = odeformer_inference(
        model=model,
        X_bits=X_t,
        unit_idx=unit_idx,
        var_mask=var_mask,
        X_data=X_raw,
        y_data=y,
        var_names=eq.var_names,
        pool_size=inf_cfg.get('pool_size', n_candidates),
        temperature=inf_cfg.get('temperature', temperature),
        top_k=5,
        max_iter=inf_cfg.get('max_iter', _INF_CFG.get('max_iter', 100)),
        n_workers=inf_cfg.get('n_workers', 4),
        n_restarts=inf_cfg.get('n_restarts', 5)
    )
    latency = time.perf_counter() - t0

    if not ode_results:
        return _failed_result(eq)
        
    # 3. Extract Best Result
    best = ode_results[0]
    best_expr_str = best['expression']
    r2 = best['r2']
    fitted_params = best.get('fitted_params', [])
    
    # 4. Symbolic Accuracy (Functional Equivalence)
    sa = verify_symbolic_accuracy(best_expr_str, eq.formula_str, eq.var_names)
    
    # 5. Constant Recovery (CR) - Only meaningful if SA is true
    cr = 0.0
    if sa:
        true_params = _extract_true_constants(eq.formula_str, eq.var_names)
        cr = calculate_constant_recovery(fitted_params, true_params)
    
    # 6. NED (Structural Similarity)
    ned = calculate_ned(best_expr_str, eq.formula_str)
    
    # 7. Node Count (Complexity)
    try:
        node_count = calculate_node_count(sympy.sympify(best_expr_str))
    except:
        node_count = 0

    return {
        'eq_id': eq.eq_id,
        'n_vars': eq.n_vars,
        'r2': r2,
        'symbolic_accuracy': sa,
        'constant_recovery': cr,
        'ned': ned,
        'node_count': node_count,
        'latency_s': latency,
        'predicted': best_expr_str,
        'truth': eq.formula_str
    }


def _extract_true_constants(formula_str: str, var_names: List[str]) -> List[float]:
    """
    Extract numerical constants from ground truth formula for comparison.
    Follows the same logic as skeletonize() but on the truth.
    """
    try:
        local_dict = {name: sympy.Symbol(name) for name in var_names}
        expr = sympy.sympify(formula_str, locals=local_dict)
        
        # We define "constants" as structural numbers that would be skeletonized
        # atoms(sympy.Number) catches floats and integers
        constants = []
        for atom in expr.atoms(sympy.Number):
            # Only include "real" constants, skip small integers that are often structural 
            # (e.g. x**2, common in Feynman). 
            # Actually, to be fair to SA-true case, we take ALL numbers.
            constants.append(float(atom))
            
        return constants
    except:
        return []


# ── Internal Helpers (Preserved & Fixed) ──────────────────────────────────────

def _prepare_model_inputs(eq, model, device):
    """Prepare padded tensor inputs."""
    X_bits = eq.X_bits  # [N, n_vars, 16] float16
    
    # If legacy scalar format, convert to bits
    if X_bits.ndim == 2:
        from data.utils import to_ieee754_16bit
        X_bits = to_ieee754_16bit(X_bits)

    model_dtype = next(model.parameters()).dtype
    X_t = torch.from_numpy(X_bits).unsqueeze(0).to(device=device, dtype=model_dtype)
    
    # Append y unit (dimensionless=4)
    y_unit = np.full((1, 5), 4, dtype=np.int64)
    unit_idx_np = np.concatenate([eq.unit_matrix_idx, y_unit], axis=0)
    unit_idx = torch.from_numpy(unit_idx_np).long().unsqueeze(0).to(device)

    max_n = model.max_n_vars
    pad = max_n - (eq.n_vars + 1)

    if pad > 0:
        # Match input dtype
        pad_x = torch.zeros(1, X_t.shape[1], pad, 16, device=device, dtype=model_dtype)
        X_t = torch.cat([X_t, pad_x], dim=2)
        pad_u = torch.full((1, pad, 5), 4, dtype=torch.long, device=device)
        unit_idx = torch.cat([unit_idx, pad_u], dim=1)

    var_mask = torch.zeros(1, max_n, device=device, dtype=model_dtype)
    var_mask[:, :eq.n_vars + 1] = 1.0

    return X_t, unit_idx, var_mask


def _reconstruct_X(eq) -> np.ndarray:
    """Reconstruct float X from bits."""
    # eq.X_bits is [N, n_vars + 1, 16] float16 (0.0/1.0)
    bits = eq.X_bits.astype(np.uint8)
    u8 = np.packbits(bits, axis=-1) # [N, n_vars + 1, 2]
    reconstructed = u8.view(np.float16).reshape(eq.X_bits.shape[0], eq.n_vars + 1).astype(np.float32)
    return reconstructed[:, :eq.n_vars]  # Drop y column and return only X


# ── Aggregation & Printing ────────────────────────────────────────────────────

def aggregate_results(results: List[dict]) -> Dict:
    """Aggregate per-equation results into Goldilocks table metrics."""
    n = len(results)
    if n == 0: return {}

    r2_vals = [r['r2'] for r in results if np.isfinite(r['r2'])]
    sa_vals = [r['symbolic_accuracy'] for r in results]
    cr_vals = [r['constant_recovery'] for r in results]
    ned_vals = [r['ned'] for r in results]
    nodes = [r['node_count'] for r in results]

    return {
        'mean_r2': np.mean(r2_vals) if r2_vals else -np.inf,
        'median_r2': np.median(r2_vals) if r2_vals else -np.inf,
        'symbolic_accuracy': np.mean(sa_vals),
        'constant_recovery': np.mean(cr_vals),
        'mean_ned': np.mean(ned_vals),
        'mean_nodes': np.mean(nodes),
        'mean_latency': np.mean([r['latency_s'] for r in results]),
        'n_equations': n,
        'per_eq_results': results
    }


def print_results(metrics: dict) -> None:
    """Pretty-print the Goldilocks Results Table."""
    print("\n" + "=" * 100)
    print(f"{'Equation':<40} | {'R²':>6} | {'SymAcc':>6} | {'ConstRec':>8} | {'NED':>6} | {'Nodes':>5}")
    print("-" * 100)
    
    for r in metrics.get('per_eq_results', []):
        eq_id = r['eq_id']
        r2 = f"{r['r2']:>6.2f}" if np.isfinite(r['r2']) else "  -inf"
        sa = "1.0" if r['symbolic_accuracy'] else "0.0"
        cr = "1.0" if r['constant_recovery'] else "0.0"
        ned = f"{r['ned']:>6.2f}"
        nodes = f"{r['node_count']:>5}"
        print(f"{eq_id:<40} | {r2} | {sa:>6} | {cr:>8} | {ned} | {nodes}")
    
    print("-" * 100)
    mean_r2 = f"{metrics.get('mean_r2', 0):.2f}"
    sym_acc = f"{metrics.get('symbolic_accuracy', 0):.2f}"
    const_rec = f"{metrics.get('constant_recovery', 0):.2f}"
    mean_ned = f"{metrics.get('mean_ned', 0):.2f}"
    print(f"{'MEAN':<40} | {mean_r2:>6} | {sym_acc:>6} | {const_rec:>8} | {mean_ned:>6} |")
    print("=" * 100)
    print("Legend: SymAcc = Symbolic Accuracy (Skeleton Match), ConstRec = Constant Recovery (Parameter Match)")
