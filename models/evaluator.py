"""
Model Evaluator for LLM-JEPA Symbolic Regression.

Provides a high-level API for loading checkpoints, running evaluations on 
the AI Feynman dataset, and generating human-readable reports.
"""

import yaml
import json
import torch
import sympy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from training.trainer import LLMJEPAModule
from predict import load_inference_model
from data.aif_dataset import build_aif_dataloader
from evaluation.evaluate import evaluate_dataset, print_results
from data.tokenizer import decode_formula, rpn_to_sympy

class ModelEvaluator:
    """
    Unified evaluation and inference handler.
    
    Args:
        config_path: Path to the YAML configuration file.
        ckpt_path:   Path to the .ckpt of a PyTorch Lightning model.
        device:      'cuda' or 'cpu'. Defaults to auto-detection.
    """
    def __init__(self, config_path: str, ckpt_path: str, device: Optional[str] = None):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        print(f"Loading model on {self.device}...")
        self.model = load_inference_model(config_path, ckpt_path, self.device)
        self.model.eval()

    def run_evaluation(self, output_dir: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Runs the full AI Feynman evaluation suite.
        
        Args:
            output_dir: If provided, saves metrics and report to this folder.
            verbose:    Whether to print progress to console.
            
        Returns:
            Dictionary of metrics.
        """
        print("Building AIF Dataloader...")
        loader = build_aif_dataloader(
            csv_path=self.config['data']['csv_path'],
            data_dir=self.config['data']['data_dir'],
            batch_size=1,
            cache_dir=self.config['data']['cache_dir'] + "aif_preprocessed.pt",
        )
        
        print("Starting comprehensive evaluation...")
        metrics = evaluate_dataset(self.model, loader.dataset, device=self.device, verbose=verbose)
        
        if output_dir:
            self._save_results(metrics, Path(output_dir), loader.dataset)
            
        return metrics

    def _save_results(self, metrics: Dict, out_path: Path, dataset):
        """Internal helper to save JSON and Markdown reports."""
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 0. Build an ID-to-Formula mapping to guarantee alignment
        id_to_formula = {eq.eq_id: eq.formula_str for eq in dataset.equations}
        
        # 1. Save Metrics JSON
        with open(out_path / "metrics.json", "w") as f:
            # Handle non-serialisable types
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (torch.Tensor, torch.long)) else x)
            
        # 2. Save Markdown Report
        report_path = out_path / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write("# 🧪 LLM-JEPA Evaluation Report\n\n")
            f.write(f"**Checkpoint:** `{Path(self.ckpt_path).name}`\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Summary Metrics\n")
            f.write(f"- **Equations Evaluated:** {metrics['n_equations']}\n")
            f.write(f"- **Exact Recovery Rate:** {metrics['exact_recovery_rate']*100:.1f}%\n")
            f.write(f"- **Mean R² (post-BFGS):** {metrics.get('mean_r2_post_bfgs', -1):.4f}\n")
            f.write(f"- **Valid RPN Rate:** {metrics['valid_rpn_rate']*100:.1f}%\n\n")
            
            f.write("## 🎨 Prediction Samples\n")
            f.write("| ID | Ground Truth | Predicted (SymPy) | Exact? |\n")
            f.write("|---|---|---|---|\n")
            
            # Show first 10 results
            for i in range(min(10, len(metrics['per_eq_results']))):
                res = metrics['per_eq_results'][i]
                eq_id = res['eq_id']
                gt = id_to_formula.get(eq_id, "Unknown ID")
                pred = res['predicted'] if res['predicted'] else "N/A"
                exact = "✅" if res['exact'] else "❌"
                f.write(f"| {eq_id} | `{gt}` | `${pred}$` | {exact} |\n")
        
        print(f"Results and report saved to: {out_path}")

    def predict_sample_by_id(self, eq_id: str) -> Optional[Dict]:
        """Runs inference on a specific equation ID from the AIF dataset."""
        loader = build_aif_dataloader(
            csv_path=self.config['data']['csv_path'],
            data_dir=self.config['data']['data_dir'],
            batch_size=1,
            cache_dir=self.config['data']['cache_dir'] + "aif_preprocessed.pt",
            shuffle=False
        )
        
        for eq in loader.dataset.equations:
            if eq.eq_id == eq_id:
                # Prepare inputs (similar to evaluation/_evaluate_one)
                from evaluation.evaluate import _prepare_model_inputs
                X_t, unit_idx, var_mask = _prepare_model_inputs(eq, self.model, self.device)
                
                with torch.no_grad():
                    # Explicitly call encode() to avoid JEPA forward-pass trap
                    z_context = self.model.encode(X_t, unit_idx, var_mask)
                    generated = self.model.generate(z_context, unit_idx)
                
                tokens = decode_formula(generated[0].cpu().tolist(), strip_special=True)
                try:
                    expr = rpn_to_sympy(tokens)
                    pred_str = str(sympy.simplify(expr))
                except:
                    pred_str = "Error parsing RPN"
                    
                return {
                    'id': eq_id,
                    'gt': eq.formula_str,
                    'pred': pred_str,
                    'tokens': tokens
                }
        return None

if __name__ == "__main__":
    # Smoke test
    evaluator = ModelEvaluator(
        config_path="configs/base_config.yaml",
        ckpt_path="checkpoints/last.ckpt"
    )
    # evaluator.run_evaluation(output_dir="results/smoke_test")
