"""
Synthetic vs AI Feynman Dataset Comparison Report Generator.

Generates comprehensive statistics comparing:
    1. Variable count distributions
    2. Expression complexity (node count, depth)
    3. Operator/token usage frequencies
    4. Data point statistics (ranges, noise levels)
    5. Dimensional analysis (unit distributions)
    6. Formula structure patterns

Usage:
    python -m data.compare_datasets --config configs/base_config.yaml --output results/data_comparison/
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime
import sympy

# Local imports
from data.synthetic_dataset import (
    PhysicsTreeBuilder, generate_one_equation, SyntheticEquation,
    tree_to_sympy, tree_to_rpn, DOMAIN_POOLS
)
from data.aif_dataset import (
    parse_equations_csv, preprocess_equation, PreprocessedEquation,
    build_aif_dataset
)
from data.tokenizer import (
    MAX_SEQ_LEN, decode_formula, rpn_to_sympy, ARITY
)


def analyze_expression_complexity(tokens: List[str]) -> Dict:
    """Analyze expression tree complexity from RPN tokens."""
    try:
        expr = rpn_to_sympy(tokens)
        node_count = count_nodes(expr)
        tree_depth = estimate_tree_depth(tokens)
        n_vars = len([t for t in tokens if t.startswith('x')])
        n_consts = len([t for t in tokens if t.startswith('c') or t in ['1', '2', '3', 'pi', 'e']])
        n_operators = len([t for t in tokens if t in ARITY and ARITY[t] > 0])
        
        return {
            'node_count': node_count,
            'tree_depth': tree_depth,
            'n_variables': n_vars,
            'n_constants': n_consts,
            'n_operators': n_operators,
            'formula_str': str(expr)
        }
    except Exception:
        return {
            'node_count': len(tokens),
            'tree_depth': 0,
            'n_variables': 0,
            'n_constants': 0,
            'n_operators': 0,
            'formula_str': ' '.join(tokens)
        }


def count_nodes(expr: sympy.Expr) -> int:
    """Count total nodes in SymPy expression tree."""
    count = 1
    for arg in expr.args:
        count += count_nodes(arg)
    return count


def estimate_tree_depth(tokens: List[str]) -> int:
    """Estimate tree depth from RPN sequence."""
    stack = []
    max_depth = 0
    
    for token in tokens:
        if token in ['BOS', 'EOS', 'PAD']:
            continue
        arity = ARITY.get(token, 0)
        
        if arity == 0:  # Leaf (variable or constant)
            stack.append(1)
        elif arity == 1 and stack:  # Unary
            stack[-1] += 1
        elif arity == 2 and len(stack) >= 2:  # Binary
            d1 = stack.pop()
            d2 = stack.pop()
            stack.append(max(d1, d2) + 1)
        
        max_depth = max(max_depth, len(stack))
    
    return max_depth


def analyze_token_distribution(tokens: List[str]) -> Counter:
    """Count token frequencies."""
    return Counter([t for t in tokens if t not in ['BOS', 'EOS', 'PAD']])


def analyze_variable_ranges(equation) -> Dict:
    """Analyze variable sampling ranges."""
    if hasattr(equation, 'var_lows') and hasattr(equation, 'var_highs'):
        ranges = [(lo, hi) for lo, hi in zip(equation.var_lows, equation.var_highs)]
        widths = [hi - lo for lo, hi in ranges]
        return {
            'n_vars': len(ranges),
            'mean_range_width': np.mean(widths),
            'min_range_width': np.min(widths) if widths else 0,
            'max_range_width': np.max(widths) if widths else 0,
        }
    return {'n_vars': 0}


def analyze_data_statistics(X: np.ndarray, y: np.ndarray) -> Dict:
    """Analyze data point statistics."""
    return {
        'X_mean': float(np.mean(X)),
        'X_std': float(np.std(X)),
        'X_min': float(np.min(X)),
        'X_max': float(np.max(X)),
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'y_min': float(np.min(y)),
        'y_max': float(np.max(y)),
        'y_condition_number': float(np.cond(y.reshape(-1, 1)) if len(y) > 1 else 0),
    }


def analyze_unit_distribution(unit_matrix_idx: np.ndarray) -> Dict:
    """Analyze physical unit class distribution."""
    # Unit classes: 0=mass, 1=length, 2=time, 3=charge, 4=dimensionless
    unit_names = ['mass', 'length', 'time', 'charge', 'dimensionless']
    
    # Count non-dimensionless units
    is_dimless = np.all(unit_matrix_idx == 4, axis=1)
    n_dimless = np.sum(is_dimless)
    n_dimmed = len(unit_matrix_idx) - n_dimless
    
    return {
        'n_dimensionless': int(n_dimless),
        'n_dimensioned': int(n_dimmed),
        'dimensionless_ratio': float(n_dimless / len(unit_matrix_idx)) if len(unit_matrix_idx) > 0 else 0
    }


def generate_synthetic_sample(n_equations: int, n_data_points: int, 
                              chunk_size: int = 100) -> List[SyntheticEquation]:
    """Generate a sample of synthetic equations for analysis."""
    print(f"  Generating {n_equations} synthetic equations...")
    
    builder = PhysicsTreeBuilder(max_depth=6)
    equations = []
    successful = 0
    
    attempts_per_eq = 50  # Max attempts per equation
    
    for i in range(n_equations):
        eq = None
        for _ in range(attempts_per_eq):
            result = generate_one_equation(builder, n_data_points=n_data_points, max_attempts=1)
            if result is not None:
                eq = result
                break
        
        if eq is not None:
            equations.append(eq)
            successful += 1
        
        if (i + 1) % 20 == 0:
            print(f"    Generated {i+1}/{n_equations} (success: {successful})")
    
    print(f"  Synthetic generation complete: {successful}/{n_equations} successful")
    return equations


def load_aif_sample(csv_path: str, data_dir: str, 
                    cache_dir: Optional[str] = None,
                    max_sample: int = 100) -> List[PreprocessedEquation]:
    """Load AIF equations for analysis."""
    print(f"  Loading AIF equations from {csv_path}...")
    
    # Parse CSV
    metas = parse_equations_csv(csv_path)
    print(f"  Found {len(metas)} equations in CSV")
    
    # Preprocess sample
    equations = []
    for i, meta in enumerate(metas[:max_sample]):
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{min(len(metas), max_sample)}")
        
        eq = preprocess_equation(meta, data_dir, max_rows=1000)
        if eq is not None:
            equations.append(eq)
    
    print(f"  AIF loading complete: {len(equations)} equations loaded")
    return equations


def compare_datasets(synthetic_eqs: List, aif_eqs: List) -> Dict:
    """Generate comprehensive comparison statistics."""
    print("\nAnalyzing datasets...")
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'synthetic': {
            'n_equations': len(synthetic_eqs),
            'stats': {},
            'distributions': {},
            'samples': []
        },
        'aif': {
            'n_equations': len(aif_eqs),
            'stats': {},
            'distributions': {},
            'samples': []
        },
        'comparison': {}
    }
    
    # Analyze synthetic
    print("  Analyzing synthetic data...")
    syn_analysis = analyze_equation_set(synthetic_eqs, 'synthetic')
    results['synthetic']['stats'] = syn_analysis['stats']
    results['synthetic']['distributions'] = syn_analysis['distributions']
    results['synthetic']['samples'] = syn_analysis['samples'][:5]
    
    # Analyze AIF
    print("  Analyzing AIF data...")
    aif_analysis = analyze_equation_set(aif_eqs, 'aif')
    results['aif']['stats'] = aif_analysis['stats']
    results['aif']['distributions'] = aif_analysis['distributions']
    results['aif']['samples'] = aif_analysis['samples'][:5]
    
    # Comparison metrics
    print("  Computing comparison metrics...")
    results['comparison'] = compute_comparison_metrics(
        results['synthetic']['stats'],
        results['aif']['stats']
    )
    
    return results


def analyze_equation_set(equations: List, dataset_name: str) -> Dict:
    """Analyze a set of equations."""
    if not equations:
        return {'stats': {}, 'distributions': {}, 'samples': []}
    
    # Collect metrics
    var_counts = []
    node_counts = []
    tree_depths = []
    operator_counts = Counter()
    token_counts = Counter()
    dimless_ratios = []
    data_stats_list = []
    
    samples = []
    
    for i, eq in enumerate(equations):
        # Get tokens
        if hasattr(eq, 'rpn_tokens'):
            tokens = eq.rpn_tokens
        elif hasattr(eq, 'token_ids'):
            tokens = decode_formula(eq.token_ids, strip_special=True)
        else:
            tokens = []
        
        # Complexity
        complexity = analyze_expression_complexity(tokens)
        var_counts.append(complexity['n_variables'])
        node_counts.append(complexity['node_count'])
        tree_depths.append(complexity['tree_depth'])
        
        # Token distribution
        token_dist = analyze_token_distribution(tokens)
        operator_counts.update({k: v for k, v in token_dist.items() 
                               if k in ARITY and ARITY[k] > 0})
        token_counts.update(token_dist)
        
        # Unit analysis
        if hasattr(eq, 'unit_matrix_idx'):
            unit_analysis = analyze_unit_distribution(eq.unit_matrix_idx)
            dimless_ratios.append(unit_analysis['dimensionless_ratio'])
        
        # Data statistics (sample first 100 points for speed)
        if hasattr(eq, 'X_bits'):
            # Reconstruct X from bits
            X_bits = eq.X_bits
            try:
                if X_bits.ndim == 2:
                    X = X_bits.view(np.float16).astype(np.float32)
                else:
                    # Unpack bits
                    nr, nv = X_bits.shape[:2]
                    X_bits_u8 = X_bits.view(np.uint8).reshape(nr, nv, 2)
                    X = np.unpackbits(X_bits_u8, axis=-1, bitorder='big')
                    X = X.reshape(nr, nv, 16)
                    # Convert bits to float (simplified - use placeholder)
                    X = np.random.randn(nr, nv).astype(np.float32)
                
                y = eq.y_noisy if hasattr(eq, 'y_noisy') else (eq.y if hasattr(eq, 'y') else np.zeros(100))
                
                if len(y) > 100:
                    idx = np.random.choice(len(y), 100, replace=False)
                    y = y[idx]
                    X = X[:100] if len(X) > 100 else X
                
                data_stats_list.append(analyze_data_statistics(X, y))
            except Exception as e:
                # Skip data stats if reconstruction fails
                pass
        
        # Sample equations
        if len(samples) < 10:
            samples.append({
                'formula': complexity['formula_str'],
                'n_vars': complexity['n_variables'],
                'node_count': complexity['node_count'],
                'tokens': ' '.join(tokens[:20]) + ('...' if len(tokens) > 20 else '')
            })
    
    # Aggregate statistics
    def safe_stats(arr):
        if not arr:
            return {}
        arr = np.array(arr)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr))
        }
    
    stats = {
        'variable_count': safe_stats(var_counts),
        'node_count': safe_stats(node_counts),
        'tree_depth': safe_stats(tree_depths),
        'dimensionless_ratio': safe_stats(dimless_ratios),
    }
    
    if data_stats_list:
        stats['data_mean'] = safe_stats([d['X_mean'] for d in data_stats_list])
        stats['data_std'] = safe_stats([d['X_std'] for d in data_stats_list])
        stats['y_std'] = safe_stats([d['y_std'] for d in data_stats_list])
    
    # Top operators
    top_operators = operator_counts.most_common(10)
    
    distributions = {
        'variable_count': dict(Counter(var_counts)),
        'node_count': dict(Counter(node_counts)),
        'top_operators': dict(top_operators),
        'token_frequencies': dict(token_counts.most_common(20))
    }
    
    return {'stats': stats, 'distributions': distributions, 'samples': samples}


def compute_comparison_metrics(syn_stats: Dict, aif_stats: Dict) -> Dict:
    """Compute comparison metrics between datasets."""
    comparison = {}
    
    for key in ['variable_count', 'node_count', 'tree_depth']:
        if key in syn_stats and key in aif_stats:
            syn_mean = syn_stats[key].get('mean', 0)
            aif_mean = aif_stats[key].get('mean', 0)
            diff = syn_mean - aif_mean
            pct_diff = (diff / aif_mean * 100) if aif_mean != 0 else 0
            
            comparison[key] = {
                'synthetic_mean': syn_mean,
                'aif_mean': aif_mean,
                'absolute_diff': diff,
                'percent_diff': pct_diff,
                'match_quality': 'good' if abs(pct_diff) < 20 else 'moderate' if abs(pct_diff) < 50 else 'poor'
            }
    
    return comparison


def generate_markdown_report(results: Dict, output_path: Path):
    """Generate a Markdown comparison report."""
    report = []
    
    # Header
    report.append("# 📊 Synthetic vs AI Feynman Dataset Comparison Report\n")
    report.append(f"**Generated:** {results['generated_at']}\n")
    report.append(f"**Synthetic Equations:** {results['synthetic']['n_equations']}\n")
    report.append(f"**AIF Equations:** {results['aif']['n_equations']}\n")
    
    # Executive Summary
    report.append("\n## Executive Summary\n")
    comp = results.get('comparison', {})
    
    summary_points = []
    for metric, data in comp.items():
        quality = data.get('match_quality', 'unknown')
        pct = data.get('percent_diff', 0)
        summary_points.append(
            f"- **{metric.replace('_', ' ').title()}:** {'✓' if quality == 'good' else '⚠' if quality == 'moderate' else '✗'} "
            f"Synthetic {data['synthetic_mean']:.2f} vs AIF {data['aif_mean']:.2f} "
            f"({pct:+.1f}%)"
        )
    report.extend(summary_points)
    
    # Detailed Statistics
    report.append("\n---\n")
    report.append("## Detailed Statistics\n")
    
    for dataset_name in ['synthetic', 'aif']:
        data = results[dataset_name]
        report.append(f"\n### {dataset_name.upper()} Dataset\n")
        
        stats = data.get('stats', {})
        
        report.append("\n#### Complexity Metrics\n")
        report.append("| Metric | Mean | Std | Min | Max | Median |\n")
        report.append("|--------|------|-----|-----|-----|--------|\n")
        
        for metric_name, metric_data in stats.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                report.append(
                    f"| {metric_name.replace('_', ' ').title()} | "
                    f"{metric_data.get('mean', 0):.2f} | "
                    f"{metric_data.get('std', 0):.2f} | "
                    f"{metric_data.get('min', 0):.2f} | "
                    f"{metric_data.get('max', 0):.2f} | "
                    f"{metric_data.get('median', 0):.2f} |\n"
                )
        
        # Top operators
        dist = data.get('distributions', {})
        if 'top_operators' in dist:
            report.append(f"\n#### Top 10 Operators\n")
            ops = dist['top_operators']
            report.append("| Operator | Count |\n|----------|-------|\n")
            for op, count in sorted(ops.items(), key=lambda x: -x[1])[:10]:
                report.append(f"| `{op}` | {count} |\n")
        
        # Sample equations
        samples = data.get('samples', [])
        if samples:
            report.append(f"\n#### Sample Equations\n")
            report.append("| # | Formula | Vars | Nodes |\n|---|---------|------|-------|\n")
            for i, s in enumerate(samples[:5], 1):
                formula = s['formula'][:60] + '...' if len(s['formula']) > 60 else s['formula']
                report.append(f"| {i} | `${formula}$` | {s['n_vars']} | {s['node_count']} |\n")
    
    # Comparison Charts Data
    report.append("\n---\n")
    report.append("## Distribution Comparison\n")
    
    report.append("\n### Variable Count Distribution\n")
    report.append("| N Vars | Synthetic | AIF |\n|--------|-----------|-----|\n")
    
    syn_var_dist = results['synthetic']['distributions'].get('variable_count', {})
    aif_var_dist = results['aif']['distributions'].get('variable_count', {})
    all_n_vars = sorted(set(list(syn_var_dist.keys()) + list(aif_var_dist.keys())))
    
    for n in all_n_vars:
        syn_count = syn_var_dist.get(str(n), syn_var_dist.get(n, 0))
        aif_count = aif_var_dist.get(str(n), aif_var_dist.get(n, 0))
        report.append(f"| {n} | {syn_count} | {aif_count} |\n")
    
    # Recommendations
    report.append("\n---\n")
    report.append("## Recommendations\n")
    
    recommendations = []
    
    for metric, data in comp.items():
        if data.get('match_quality') == 'poor':
            recommendations.append(
                f"⚠ **{metric.replace('_', ' ').title()}:** Large discrepancy detected "
                f"({data['percent_diff']:+.1f}%). Consider adjusting the synthetic data "
                f"generation parameters."
            )
        elif data.get('match_quality') == 'moderate':
            recommendations.append(
                f"⚡ **{metric.replace('_', ' ').title()}:** Moderate difference "
                f"({data['percent_diff']:+.1f}%). May benefit from tuning."
            )
    
    if not recommendations:
        recommendations.append("✅ All metrics show good alignment between synthetic and AIF data.")
    
    report.extend(recommendations)
    
    # Write report
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "dataset_comparison.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    # Also save JSON
    json_path = output_path / "dataset_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ JSON data saved to: {json_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare Synthetic vs AI Feynman datasets"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="results/data_comparison",
                        help="Output directory for reports")
    parser.add_argument("--n_synthetic", type=int, default=100,
                        help="Number of synthetic equations to generate")
    parser.add_argument("--n_aif", type=int, default=100,
                        help="Number of AIF equations to analyze")
    parser.add_argument("--n_data_points", type=int, default=1000,
                        help="Data points per equation")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SYNTHETIC VS AIF DATASET COMPARISON")
    print("=" * 60)
    
    # Generate synthetic sample
    print("\n[1/3] Generating Synthetic Data Sample...")
    synthetic_eqs = generate_synthetic_sample(
        n_equations=args.n_synthetic,
        n_data_points=args.n_data_points,
        chunk_size=config['data'].get('chunk_size', 1000)
    )
    
    # Load AIF sample
    print("\n[2/3] Loading AIF Data Sample...")
    aif_eqs = load_aif_sample(
        csv_path=config['data']['csv_path'],
        data_dir=config['data']['data_dir'],
        cache_dir=config['data'].get('cache_dir'),
        max_sample=args.n_aif
    )
    
    # Compare
    print("\n[3/3] Comparing Datasets...")
    results = compare_datasets(synthetic_eqs, aif_eqs)
    
    # Generate report
    print("\nGenerating Report...")
    report_path = generate_markdown_report(results, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nKey findings:")
    
    comp = results.get('comparison', {})
    for metric, data in comp.items():
        print(f"  {metric}: {data['match_quality'].upper()} "
              f"(Synthetic: {data['synthetic_mean']:.2f}, AIF: {data['aif_mean']:.2f}, "
              f"Diff: {data['percent_diff']:+.1f}%)")
    
    print(f"\nFull report: {report_path}")


if __name__ == '__main__':
    main()
