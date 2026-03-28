"""
Evaluation entry point for LLM-JEPA Symbolic Regression.

Uses the ModelEvaluator module to run tests and save results.

Usage:
    python run_eval.py --ckpt checkpoints/last.ckpt
"""

import argparse
from pathlib import Path
from models.evaluator import ModelEvaluator
from evaluation.evaluate import print_results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-JEPA on AIF dataset"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="results/latest_eval",
                        help="Directory to save metrics and report")
    args = parser.parse_args()

    # Initialize the unified evaluator
    evaluator = ModelEvaluator(
        config_path=args.config,
        ckpt_path=args.ckpt
    )

    # Run evaluation and save results
    metrics = evaluator.run_evaluation(output_dir=args.output_dir, verbose=True)

    # Final printout
    print_results(metrics)
    print(f"\n✅ All artifacts saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
