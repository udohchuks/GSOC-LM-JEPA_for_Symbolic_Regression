"""
Evaluation entry point for LLM-JEPA Symbolic Regression.

Uses the ModelEvaluator module to run 'Goldilocks' tests (R2, SA, NED) 
on AI Feynman and save results.

Usage:
    python run_eval.py --ckpt checkpoints/last.ckpt --n_candidates 50 --temperature 0.1
"""

import os
import argparse
import torch
import yaml
from pathlib import Path
from models.evaluator import ModelEvaluator
from evaluation.evaluate import print_results

def _load_inference_config(config_path="configs/base_config.yaml"):
    """Helper to load default inference params from config."""
    try:
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            return full_cfg.get("inference", {})
    except:
        return {}

def main():
    # Pre-load config for argument defaults
    inf_cfg = _load_inference_config()
    
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-JEPA on AI Feynman using Goldilocks Suite"
    )
    parser.add_argument("--config", type=str, default="configs/small.yaml",
                        help="Path to model config (use configs/small.yaml for small models)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--mode", type=str, choices=["eval", "predict"], default="eval", 
                        help="'eval' for full Feynman suite, 'predict' for single equation")
    parser.add_argument("--id", type=str, default="I.6.2a", help="Equation ID for 'predict' mode")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # ODEFormer / Goldilocks Params (Defaults from config)
    parser.add_argument("--n_candidates", type=int, default=inf_cfg.get("pool_size", 50), 
                        help="Pool size (N) for diversity sampling")
    parser.add_argument("--temperature", type=float, default=inf_cfg.get("temperature", 0.1),
                        help="Sampling temperature")

    args = parser.parse_args()

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Verify Checkpoint
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    # Initialize Evaluator
    evaluator = ModelEvaluator(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=device
    )

    if args.mode == "eval":
        # Run Goldilocks Evaluation
        print(f"📈 Running Goldilocks evaluation (N={args.n_candidates}, T={args.temperature})...")
        metrics = evaluator.run_evaluation(
            output_dir=args.output_dir, 
            verbose=True, 
            n_candidates=args.n_candidates,
            temperature=args.temperature
        )
        print_results(metrics)
        print(f"\nGoldilocks report saved to: {args.output_dir}/goldilocks_report.md")

    elif args.mode == "predict":
        # Run Single Prediction
        print(f"🎯 Predicting equation: {args.id} (N={args.n_candidates}, T={args.temperature})...")
        sample = evaluator.predict_sample_by_id(
            args.id, 
            n_candidates=args.n_candidates,
            temperature=args.temperature
        )
        if sample:
            print(f"\n--- Prediction Result ---")
            print(f"ID:           {sample['id']}")
            print(f"Ground Truth: {sample['gt']}")
            print(f"Prediction:   {sample['pred']}")
            print(f"R²:           {sample['r2']:.6f}")
            print(f"SA (Equiv):   {sample['sa']}")
            print(f"NED:          {sample['ned']:.4f}")
        else:
            print(f"Error: Could not find equation {args.id} in dataset.")

if __name__ == '__main__':
    main()
