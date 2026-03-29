"""
Evaluation suite for LLM-JEPA Symbolic Regression.

Comprehensive metrics logged to TensorBoard:
    1. Robustness & Extrapolation (noise tolerance, data efficiency, OOD)
    2. Interpretability (formula complexity / node count)
    3. Granular Precision (Acc_τ at multiple tolerances, pre/post BFGS R²)
    4. Operational (inference latency)
    5. Standard (exact recovery, mean R², valid RPN rate, dimensional validity)
"""

from __future__ import annotations
import time
import numpy as np
import sympy
from typing import List, Dict, Optional, Tuple

from data.tokenizer import decode_formula, is_valid_rpn, rpn_to_sympy
from data.utils import compute_unit_targets
from evaluation.metrics import (
    calculate_node_count,
    calculate_acc_tau,
    calculate_r2,
    fit_constants,
    fit_constants_with_scipy,
    verify_exact,
    _evaluate_expr,
)


# ── Constants ─────────────────────────────────────────────────────────────────

NOISE_LEVELS    = [0.001, 0.01, 0.1]
DATA_SIZES      = [10, 50, 100, 200]
TAU_THRESHOLDS  = [0.1, 0.01, 0.001]
EXTRAP_FACTOR   = 10.0   # multiply original range by this for OOD test


# ── Main evaluation entry point ──────────────────────────────────────────────

def evaluate_dataset(
    model,
    dataset,
    device:     str   = 'cpu',
    n_restarts: int   = 3,
    n_candidates: int = 1,
    temperature: float = 0.8,
    verbose:    bool  = True,
) -> Dict:
    """
    Evaluate model on a full dataset (AIF evaluation set).

    Returns dict of aggregated metric values.
    """
    import torch

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
        'exact': False,
        'r2_pre_bfgs': -np.inf,
        'r2_post_bfgs': -np.inf,
        'valid_rpn': False,
        'dim_valid': False,
        'node_count': 0,
        'acc_tau': {tau: 0.0 for tau in TAU_THRESHOLDS},
        'latency_generate_s': 0.0,
        'latency_bfgs_s': 0.0,
        'noise_r2': {eps: -np.inf for eps in NOISE_LEVELS},
        'data_size_r2': {n: -np.inf for n in DATA_SIZES},
        'extrap_r2': -np.inf,
        'tokens': [],
        'predicted': None,
    }


# ── Per-equation evaluation ──────────────────────────────────────────────────

def _evaluate_one(model, eq, device, n_restarts, n_candidates=1, temperature=0.8) -> dict:
    """Full evaluation for one AIF equation with all metric categories."""
    import torch

    model.eval()
    n_vars = eq.n_vars

    # Reconstruct raw float X from IEEE-754 bits
    X_raw = _reconstruct_X(eq)  # [N, n_vars]
    y = eq.y                     # [N]

    # ── 1. Prepare model inputs ───────────────────────────────────────────
    X_t, unit_idx, var_mask = _prepare_model_inputs(eq, model, device)

    # ── 2. Generate formula candidates ────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        z_context = model(X_t, unit_idx, var_mask)
        if n_candidates > 1:
            # Expand context for batch generation
            z_context_exp = z_context.expand(n_candidates, -1, -1)
            unit_idx_exp = unit_idx.expand(n_candidates, -1, -1)
            generated = model.generate(z_context_exp, unit_idx_exp, greedy=False, temperature=temperature)
        else:
            generated = model.generate(z_context, unit_idx, greedy=True)
    latency_generate = time.perf_counter() - t0

    # Deduplicate unique skeletons (Pro-Tip from the user!)
    unique_skeletons = {}
    valid_tokens_list = []
    
    for i in range(n_candidates):
        token_ids = generated[i].cpu().tolist()
        tokens = decode_formula(token_ids, strip_special=True)
        if tokens and is_valid_rpn(tokens):
            try:
                skeleton = rpn_to_sympy(tokens)
                skel_str = str(skeleton)
                if skel_str not in unique_skeletons:
                    unique_skeletons[skel_str] = (skeleton, tokens)
                valid_tokens_list.append(tokens)
            except Exception:
                continue

    if not unique_skeletons:
        result = _failed_result(eq)
        result['latency_generate_s'] = latency_generate
        return result

    # Standard "best-guess" tokens for backward compatibility in the result dict
    # We'll use the tokens from the first candidate (highest likelihood if greedy/low-temp)
    best_tokens = valid_tokens_list[0] if valid_tokens_list else []
    valid_rpn = len(valid_tokens_list) > 0

    # ── 3. Post-BFGS Optimization (pick best candidate) ───────────────────
    print(f"    Fitting constants for {len(unique_skeletons)} unique candidates...")
    t1 = time.perf_counter()
    best_mse = float('inf')
    best_expr = None
    best_tokens_final = None

    for skel_str, (skeleton, tokens) in unique_skeletons.items():
        mse, optimized_expr = fit_constants_with_scipy(skeleton, X_raw, y, eq.var_names)
        if mse < best_mse:
            best_mse = mse
            best_expr = optimized_expr
            best_tokens_final = tokens

    latency_bfgs = time.perf_counter() - t1
    r2_post = calculate_r2(y, _evaluate_expr(best_expr, eq.var_names, X_raw)) if best_expr else -np.inf

    # ── 4. Pre-BFGS R² (for reporting) ───────────────────────────────────
    # (Using the best final skeleton but with constants=1.0)
    try:
        y_pred_pre = _evaluate_expr(best_expr.subs({s: 1.0 for s in best_expr.free_symbols if str(s).startswith('c')}), eq.var_names, X_raw)
        r2_pre = calculate_r2(y, y_pred_pre)
    except Exception:
        r2_pre = -np.inf

    # ── 5. Node count (complexity) ────────────────────────────────────────
    try:
        node_count = calculate_node_count(best_expr)
    except Exception:
        node_count = len(best_tokens_final) if best_tokens_final else 0

    # ── 6. Accuracy to tolerance ──────────────────────────────────────────
    acc_tau_results = {}
    try:
        if best_expr and r2_post > -np.inf:
            y_pred_fitted = _evaluate_expr(best_expr, eq.var_names, X_raw)
        elif r2_pre > -np.inf:
            y_pred_fitted = y_pred_pre
        else:
            y_pred_fitted = None

        for tau in TAU_THRESHOLDS:
            if y_pred_fitted is not None and np.all(np.isfinite(y_pred_fitted)):
                acc_tau_results[tau] = calculate_acc_tau(y, y_pred_fitted, tau)
            else:
                acc_tau_results[tau] = 0.0
    except Exception:
        acc_tau_results = {tau: 0.0 for tau in TAU_THRESHOLDS}

    exact = False
    predicted_str = None
    try:
        if best_expr:
            predicted_str = str(sympy.simplify(best_expr))
            exact = verify_exact(predicted_str, eq.formula_str, eq.var_names)
    except Exception:
        pass

    # ── 8. Dimensional validity ───────────────────────────────────────────
    dim_valid = _check_dimensional_validity(best_tokens_final, eq.var_names)

    # ── 9. Stress tests ──────────────────────────────────────────────────
    # Note: Stress tests still use existing logic but with best_tokens_final.
    # To avoid complex SymPy extraction of BFGS-fitted constants, we re-run 
    # a quick fit_constants on the best skeleton to get the dict format.
    stress_constants, _ = fit_constants(best_tokens_final, eq.var_names, X_raw, y, n_restarts=1)

    noise_r2 = _test_noise_tolerance(best_tokens_final, eq, X_raw, y, stress_constants)
    data_size_r2 = _test_data_efficiency(model, eq, device, n_restarts)
    extrap_r2 = _test_extrapolation(best_tokens_final, eq, stress_constants)

    return {
        'eq_id':              eq.eq_id,
        'n_vars':             n_vars,
        'exact':              exact,
        'r2_pre_bfgs':        r2_pre,
        'r2_post_bfgs':       r2_post,
        'mse':                best_mse,
        'valid_rpn':          valid_rpn,
        'dim_valid':          dim_valid,
        'node_count':         node_count,
        'acc_tau':            acc_tau_results,
        'latency_generate_s': latency_generate,
        'latency_bfgs_s':     latency_bfgs,
        'noise_r2':           noise_r2,
        'data_size_r2':       data_size_r2,
        'extrap_r2':          extrap_r2,
        'tokens':             best_tokens_final,
        'predicted':          predicted_str,
    }


# ── Stress Tests ──────────────────────────────────────────────────────────────

def _test_noise_tolerance(
    tokens: List[str],
    eq,
    X_raw: np.ndarray,
    y: np.ndarray,
    constants: Optional[dict],
) -> Dict[float, float]:
    """
    Evaluate R² as a function of additive noise on y.
    Noise is Gaussian with std = epsilon * std(y).
    """
    results = {}
    for eps in NOISE_LEVELS:
        try:
            with np.errstate(all='ignore'):
                noise = np.random.randn(len(y)) * eps * np.std(y)
                y_noisy = y + noise.astype(np.float32)

                # Re-fit constants on noisy data
                _, r2 = fit_constants(tokens, eq.var_names, X_raw, y_noisy, n_restarts=2)
                results[eps] = r2
        except Exception:
            results[eps] = -np.inf
    return results


def _test_data_efficiency(
    model,
    eq,
    device: str,
    n_restarts: int,
) -> Dict[int, float]:
    """
    Evaluate R² when the model sees only N data points.
    Tests data efficiency by subsampling.
    """
    import torch

    results = {}
    X_raw = _reconstruct_X(eq)
    y = eq.y
    N_total = X_raw.shape[0]

    for n_points in DATA_SIZES:
        try:
            if n_points >= N_total:
                results[n_points] = -np.inf
                continue

            # Subsample data
            idx = np.random.choice(N_total, n_points, replace=False)
            X_sub = X_raw[idx]
            y_sub = y[idx]

            # Re-encode and generate for the subsampled data
            from data.utils import to_ieee754_16bit
            X_bits_compact = to_ieee754_16bit(X_sub)
            
            # Unpack compact bits to [N, n_vars, 16]
            nr, nv = X_bits_compact.shape
            X_bits_sub = X_bits_compact.view(np.uint8).reshape(nr, nv, 2)
            X_bits_sub = np.unpackbits(X_bits_sub, axis=-1, bitorder='big').reshape(nr, nv, 16)
            
            X_t = torch.from_numpy(X_bits_sub).float().unsqueeze(0).to(device)

            unit_idx = torch.from_numpy(eq.unit_matrix_idx).long().unsqueeze(0).to(device)
            max_n = model.max_n_vars
            pad = max_n - eq.n_vars

            if pad > 0:
                pad_x = torch.zeros(1, X_t.shape[1], pad, 16, device=device)
                X_t = torch.cat([X_t, pad_x], dim=2)
                pad_u = torch.full((1, pad, 5), 4, dtype=torch.long, device=device)
                unit_idx = torch.cat([unit_idx, pad_u], dim=1)

            var_mask = torch.zeros(1, max_n, device=device)
            var_mask[:, :eq.n_vars] = 1.0

            with torch.no_grad():
                z_context = model(X_t, unit_idx, var_mask)
                generated = model.generate(z_context, unit_idx)

            token_ids = generated[0].cpu().tolist()
            tokens = decode_formula(token_ids, strip_special=True)

            if not tokens or not is_valid_rpn(tokens):
                results[n_points] = -np.inf
                continue

            # Fit constants on subsampled data, evaluate on full data
            constants, _ = fit_constants(tokens, eq.var_names, X_sub, y_sub, n_restarts=2)

            # Evaluate on FULL data to measure generalisation
            expr = rpn_to_sympy(tokens)
            if constants:
                for c_name, c_val in constants.items():
                    expr = expr.subs(sympy.Symbol(c_name), sympy.Float(c_val))

            symbols = [sympy.Symbol(f'x{i+1}') for i in range(eq.n_vars)]
            f_eval = sympy.lambdify(symbols, expr, modules='numpy')
            
            with np.errstate(all='ignore'):
                y_pred_full = np.asarray(f_eval(*X_raw.T), dtype=np.float64)

                if np.all(np.isfinite(y_pred_full)):
                    results[n_points] = calculate_r2(y, y_pred_full)
                else:
                    results[n_points] = -np.inf

        except Exception:
            results[n_points] = -np.inf

    return results


def _test_extrapolation(
    tokens: List[str],
    eq,
    constants: Optional[dict],
) -> float:
    """
    Evaluate R² on data sampled outside the original variable ranges.
    Multiplies original ranges by EXTRAP_FACTOR.
    """
    try:
        n_vars = eq.n_vars
        n_points = 500

        # Generate out-of-domain data using extended ranges
        X_ood = np.zeros((n_points, n_vars), dtype=np.float32)
        for j in range(n_vars):
            low = eq.var_lows[j] if hasattr(eq, 'var_lows') else 1.0
            high = eq.var_highs[j] if hasattr(eq, 'var_highs') else 5.0
            range_width = high - low
            # Sample from OUTSIDE the original range
            ood_low = high  # start from original high
            ood_high = high + EXTRAP_FACTOR * range_width
            X_ood[:, j] = np.random.uniform(ood_low, ood_high, n_points).astype(np.float32)

        # Compute ground truth y from the original formula
        local_dict = {name: sympy.Symbol(name) for name in eq.var_names}
        local_dict['pi'] = sympy.pi
        local_dict['e'] = sympy.E
        true_expr = sympy.sympify(eq.formula_str, locals=local_dict)
        true_symbols = [sympy.Symbol(name) for name in eq.var_names]
        f_true = sympy.lambdify(true_symbols, true_expr, modules='numpy')
        y_ood = np.asarray(f_true(*X_ood.T), dtype=np.float64)

        if not np.all(np.isfinite(y_ood)):
            return -np.inf

        # Evaluate predicted formula on OOD data
        pred_expr = rpn_to_sympy(tokens)
        if constants:
            for c_name, c_val in constants.items():
                pred_expr = pred_expr.subs(sympy.Symbol(c_name), sympy.Float(c_val))

        symbols_pred = [sympy.Symbol(f'x{i+1}') for i in range(n_vars)]
        f_pred = sympy.lambdify(symbols_pred, pred_expr, modules='numpy')
        
        with np.errstate(all='ignore'):
            y_pred_ood = np.asarray(f_pred(*X_ood.T), dtype=np.float64)

            if not np.all(np.isfinite(y_pred_ood)):
                return -np.inf

            return calculate_r2(y_ood, y_pred_ood)
    except Exception:
        return -np.inf


# ── Helper Functions ──────────────────────────────────────────────────────────

def _prepare_model_inputs(eq, model, device):
    """Prepare padded tensor inputs for the model from a PreprocessedEquation."""
    import torch

    X_bits = eq.X_bits
    if X_bits.ndim == 2:
        # Unpack compact bits [N, n_vars] -> [N, n_vars, 16]
        nr, nv = X_bits.shape
        X_bits = X_bits.view(np.uint8).reshape(nr, nv, 2)
        X_bits = np.unpackbits(X_bits, axis=-1, bitorder='big').reshape(nr, nv, 16)

    X_t = torch.from_numpy(X_bits).float().unsqueeze(0).to(device)
    unit_idx = torch.from_numpy(eq.unit_matrix_idx).long().unsqueeze(0).to(device)

    max_n = model.max_n_vars
    pad = max_n - eq.n_vars

    if pad > 0:
        pad_x = torch.zeros(1, X_t.shape[1], pad, 16, device=device)
        X_t = torch.cat([X_t, pad_x], dim=2)
        pad_u = torch.full((1, pad, 5), 4, dtype=torch.long, device=device)
        unit_idx = torch.cat([unit_idx, pad_u], dim=1)

    var_mask = torch.zeros(1, max_n, device=device)
    var_mask[:, :eq.n_vars] = 1.0

    return X_t, unit_idx, var_mask


def _reconstruct_X(eq) -> np.ndarray:
    """
    Reconstruct approximate float X from IEEE-754 bits.
    """
    # 1. Handle compact uint16 bit-packed representation [N, n_vars]
    # Each uint16 maps directly to a 16-bit float view
    if eq.X_bits.ndim == 2:
        return eq.X_bits.view(np.float16).astype(np.float32)

    # 2. Fallback for unpacked bits [N, n_vars, 16] (legacy or debugging)
    bits = eq.X_bits.astype(np.uint8)
    uint8 = np.packbits(bits.reshape(-1, 16), axis=-1)
    uint16 = uint8.view(np.uint16)
    f16 = uint16.view(np.float16)
    X = f16.reshape(eq.X_bits.shape[0], eq.n_vars).astype(np.float32)
    return X


def _check_dimensional_validity(tokens: list, var_names: list) -> bool:
    """Check if generated RPN sequence is dimensionally valid."""
    try:
        targets = compute_unit_targets(tokens, var_names)
        return len(targets) == len(tokens)
    except Exception:
        return False


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(results: List[dict]) -> Dict:
    """
    Aggregate per-equation results into dataset-level metrics.
    """
    n = len(results)
    if n == 0:
        return {}

    exact    = [r['exact']          for r in results]
    r2_pre   = [r['r2_pre_bfgs']    for r in results]
    r2_post  = [r['r2_post_bfgs']   for r in results]
    valid    = [r['valid_rpn']       for r in results]
    dim_v    = [r['dim_valid']       for r in results]
    n_vars   = [r['n_vars']          for r in results]
    nodes    = [r['node_count']      for r in results]
    lat_gen  = [r['latency_generate_s'] for r in results]
    lat_bfgs = [r['latency_bfgs_s']     for r in results]

    finite_r2_pre  = [r for r in r2_pre  if np.isfinite(r)]
    finite_r2_post = [r for r in r2_post if np.isfinite(r)]

    # Recovery by variable count
    recovery_by_n = {}
    for nv in sorted(set(n_vars)):
        mask = [i for i, v in enumerate(n_vars) if v == nv]
        if mask:
            recovery_by_n[nv] = np.mean([exact[i] for i in mask])

    # Aggregate Acc_τ
    acc_tau_mean = {}
    for tau in TAU_THRESHOLDS:
        vals = [r['acc_tau'].get(tau, 0.0) for r in results]
        acc_tau_mean[tau] = np.mean(vals)

    # Aggregate noise tolerance
    noise_r2_mean = {}
    for eps in NOISE_LEVELS:
        vals = [r['noise_r2'].get(eps, -np.inf) for r in results]
        finite_vals = [v for v in vals if np.isfinite(v)]
        noise_r2_mean[eps] = np.mean(finite_vals) if finite_vals else -np.inf

    # Aggregate data efficiency
    data_size_r2_mean = {}
    for n_pts in DATA_SIZES:
        vals = [r['data_size_r2'].get(n_pts, -np.inf) for r in results]
        finite_vals = [v for v in vals if np.isfinite(v)]
        data_size_r2_mean[n_pts] = np.mean(finite_vals) if finite_vals else -np.inf

    # Aggregate extrapolation
    extrap_vals = [r['extrap_r2'] for r in results]
    finite_extrap = [v for v in extrap_vals if np.isfinite(v)]

    return {
        # Standard
        'exact_recovery_rate': np.mean(exact),
        'n_exact':             sum(exact),
        'n_equations':         n,
        'mean_r2_pre_bfgs':    np.mean(finite_r2_pre) if finite_r2_pre else -np.inf,
        'mean_r2_post_bfgs':   np.mean(finite_r2_post) if finite_r2_post else -np.inf,
        'valid_rpn_rate':      np.mean(valid),
        'dim_valid_rate':      np.mean(dim_v),
        'recovery_by_n_vars':  recovery_by_n,

        # Complexity
        'mean_node_count':     np.mean(nodes),
        'node_counts':         nodes,

        # Precision
        'acc_tau':             acc_tau_mean,

        # Latency
        'mean_latency_generate_s': np.mean(lat_gen),
        'mean_latency_bfgs_s':     np.mean(lat_bfgs),
        'total_latency_s':         sum(lat_gen) + sum(lat_bfgs),

        # Robustness
        'noise_r2':            noise_r2_mean,
        'data_size_r2':        data_size_r2_mean,
        'mean_extrap_r2':      np.mean(finite_extrap) if finite_extrap else -np.inf,

        # Per-equation details (for TensorBoard histograms)
        'per_eq_r2_pre':       r2_pre,
        'per_eq_r2_post':      r2_post,
        'per_eq_node_count':   nodes,
        'per_eq_results':      results,
    }


# ── Console output ────────────────────────────────────────────────────────────

def print_results(metrics: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Standard
    print(f"\n{'─'*40}")
    print("STANDARD METRICS")
    print(f"{'─'*40}")
    print(f"Equations evaluated:  {metrics.get('n_equations', 0)}")
    print(f"Exact recovery:       {metrics.get('n_exact', 0)} / "
          f"{metrics.get('n_equations', 0)} "
          f"({metrics.get('exact_recovery_rate', 0)*100:.1f}%)")
    print(f"Valid RPN rate:       {metrics.get('valid_rpn_rate', 0)*100:.1f}%")
    print(f"Dimensionally valid:  {metrics.get('dim_valid_rate', 0)*100:.1f}%")

    # Precision
    print(f"\n{'─'*40}")
    print("PRECISION METRICS")
    print(f"{'─'*40}")
    print(f"Mean R² (pre-BFGS):   {metrics.get('mean_r2_pre_bfgs', -np.inf):.4f}")
    print(f"Mean R² (post-BFGS):  {metrics.get('mean_r2_post_bfgs', -np.inf):.4f}")
    r2_delta = metrics.get('mean_r2_post_bfgs', 0) - metrics.get('mean_r2_pre_bfgs', 0)
    print(f"BFGS improvement:     {r2_delta:+.4f}")
    for tau, acc in sorted(metrics.get('acc_tau', {}).items()):
        print(f"Acc_τ (τ={tau}):       {acc*100:.1f}%")

    # Complexity
    print(f"\n{'─'*40}")
    print("COMPLEXITY METRICS")
    print(f"{'─'*40}")
    print(f"Mean node count:      {metrics.get('mean_node_count', 0):.1f}")

    # Robustness
    print(f"\n{'─'*40}")
    print("ROBUSTNESS & STRESS TESTS")
    print(f"{'─'*40}")
    print("Noise tolerance (R² vs noise level):")
    for eps, r2 in sorted(metrics.get('noise_r2', {}).items()):
        print(f"  ε={eps:.3f}: R²={r2:.4f}")
    print("Data efficiency (R² vs N data points):")
    for n, r2 in sorted(metrics.get('data_size_r2', {}).items()):
        print(f"  N={n}: R²={r2:.4f}")
    print(f"Extrapolation R²:     {metrics.get('mean_extrap_r2', -np.inf):.4f}")

    # Latency
    print(f"\n{'─'*40}")
    print("OPERATIONAL METRICS")
    print(f"{'─'*40}")
    print(f"Mean generation time:  {metrics.get('mean_latency_generate_s', 0)*1000:.1f} ms")
    print(f"Mean BFGS time:        {metrics.get('mean_latency_bfgs_s', 0)*1000:.1f} ms")
    print(f"Total eval time:       {metrics.get('total_latency_s', 0):.1f} s")

    # Recovery by variable count
    print(f"\n{'─'*40}")
    print("RECOVERY BY VARIABLE COUNT")
    print(f"{'─'*40}")
    for nv, rate in sorted(metrics.get('recovery_by_n_vars', {}).items()):
        print(f"  {nv} variables: {rate*100:.1f}%")
    print("=" * 60)
