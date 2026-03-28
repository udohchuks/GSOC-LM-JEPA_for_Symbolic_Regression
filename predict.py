import argparse
import yaml
import torch
import os
from pathlib import Path

from models.evaluator import ModelEvaluator
from evaluation.evaluate import print_results

def main():
    parser = argparse.ArgumentParser(description="Run Inference on LLM-JEPA")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--mode", type=str, choices=["eval", "predict"], default="predict", 
                        help="'eval' for full Feynman suite, 'predict' for single equation")
    parser.add_argument("--id", type=str, default="I.6.2a", help="Equation ID to predict")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Verify Checkpoint
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    # Initialize Evaluator
    try:
        evaluator = ModelEvaluator(
            config_path=args.config,
            ckpt_path=args.ckpt,
            device=device
        )
    except Exception as e:
        print(f"Failed to initialize evaluator: {e}")
        return

    if args.mode == "predict":
        # Run Single Prediction
        print(f"🎯 Predicting equation: {args.id}...")
        sample = evaluator.predict_sample_by_id(args.id)
        if sample:
            print(f"\n--- Prediction Result ---")
            print(f"ID:           {sample['id']}")
            print(f"Ground Truth: {sample['gt']}")
            print(f"Prediction:   {sample['pred']}")
            print(f"RPN Tokens:   {' '.join(sample['tokens'])}")
            print(f"Exact:        {sample['exact']}")
        else:
            print(f"Error: Could not find equation {args.id} in dataset.")

    elif args.mode == "eval":
        # Run Full AIF Evaluation
        print(f"📈 Running full AI Feynman evaluation...")
        metrics = evaluator.run_evaluation(output_dir=args.output_dir, verbose=True)
        print_results(metrics)
        print(f"Full report saved to: {args.output_dir}/evaluation_report.md")

if __name__ == '__main__':
    main()
