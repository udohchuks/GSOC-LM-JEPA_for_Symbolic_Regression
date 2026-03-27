"""
Evaluation entry point for LLM-JEPA Symbolic Regression.

Runs the comprehensive evaluation suite and saves all metrics
to a JSON file for later analysis.

Usage:
    python run_eval.py --ckpt checkpoints/best.ckpt
    python run_eval.py --ckpt checkpoints/best.ckpt --output results/eval_v1.json
"""

import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path

from training.trainer import LLMJEPAModule
from predict import load_inference_model
from data.aif_dataset import build_aif_dataloader
from evaluation.evaluate import evaluate_dataset, print_results


def _make_serialisable(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _make_serialisable(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save all evaluation metrics to a JSON file."""
    # Strip per-equation raw results to keep file manageable
    save_dict = {}
    for k, v in metrics.items():
        if k == 'per_eq_results':
            # Save a compact per-equation summary
            save_dict['per_equation'] = [
                {
                    'eq_id':          r['eq_id'],
                    'n_vars':         r['n_vars'],
                    'exact':          r['exact'],
                    'r2_pre_bfgs':    r['r2_pre_bfgs'],
                    'r2_post_bfgs':   r['r2_post_bfgs'],
                    'node_count':     r['node_count'],
                    'valid_rpn':      r['valid_rpn'],
                    'dim_valid':      r['dim_valid'],
                    'acc_tau':        r['acc_tau'],
                    'latency_generate_s': r['latency_generate_s'],
                    'latency_bfgs_s':     r['latency_bfgs_s'],
                    'noise_r2':       r['noise_r2'],
                    'data_size_r2':   r['data_size_r2'],
                    'extrap_r2':      r['extrap_r2'],
                    'predicted':      r.get('predicted'),
                }
                for r in v
            ]
        elif k in ('per_eq_r2_pre', 'per_eq_r2_post', 'per_eq_node_count'):
            continue  # already in per_equation
        else:
            save_dict[k] = v

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(_make_serialisable(save_dict), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-JEPA on AIF dataset with comprehensive metrics"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--output", type=str, default="results/eval_results.json",
                        help="Output JSON file for metrics")
    parser.add_argument("--n_restarts", type=int, default=3,
                        help="BFGS restarts per equation")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading InferenceModel for evaluation on {device}...")
    model = load_inference_model(args.config, args.ckpt, device)

    print("Loading AI Feynman dataset...")
    full_loader = build_aif_dataloader(
        csv_path=config['data']['csv_path'],
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        cache_dir=config['data']['cache_dir'] + "aif_preprocessed.pt",
    )

    # ── Run evaluation ────────────────────────────────────────────────────
    print("Starting comprehensive evaluation...")
    print("  (includes stress tests: noise, data efficiency, extrapolation)")
    metrics = evaluate_dataset(
        model=model,
        dataset=full_loader.dataset,
        device=device,
        n_restarts=args.n_restarts,
        verbose=True,
    )

    # ── Save and display ──────────────────────────────────────────────────
    save_metrics(metrics, args.output)
    print_results(metrics)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
