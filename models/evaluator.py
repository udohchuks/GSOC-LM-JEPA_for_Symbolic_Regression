"""
Model Evaluator for LLM-JEPA Symbolic Regression.

Provides a high-level API for loading checkpoints, running evaluations on 
the AI Feynman dataset using the 'Goldilocks' suite, and generating 
human-readable reports.
"""

import yaml
import json
import torch
import sympy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from training.trainer import LLMJEPAModule
from inference.generate import InferenceModel
from data.aif_dataset import build_aif_dataloader
from evaluation.evaluate import evaluate_dataset, print_results
from evaluation.metrics import verify_symbolic_accuracy, calculate_ned
from data.tokenizer import decode_formula, rpn_to_sympy, VOCAB_SIZE, MAX_SEQ_LEN

def load_inference_model(config_path: str, ckpt_path: str, device: str) -> InferenceModel:
    """Load trained checkpoint into the InferenceModel for generation."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pl_module = LLMJEPAModule.load_from_checkpoint(ckpt_path, map_location=device)
    base_model = pl_module.model

    inf_model = InferenceModel(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_enc_layers'],
        n_decoder_layers=config['model']['n_dec_layers'],
        max_n_vars=config['data']['max_n_vars'],
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    # Copy weights
    inf_model.data_embedder.load_state_dict(base_model.data_embedder.state_dict())
    inf_model.unit_embedder.load_state_dict(base_model.unit_embedder.state_dict())
    inf_model.context_encoder.load_state_dict(base_model.mix_encoder.state_dict())
    inf_model.decoder.load_state_dict(base_model.decoder.state_dict())

    inf_model.max_n_vars = config['data']['max_n_vars']
    
    # Set to eval mode and ensure float32 for inference
    inf_model.eval()
    inf_model = inf_model.float()  # Ensure float32 for inference
    return inf_model

class ModelEvaluator:
    """
    Unified evaluation and inference handler.
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

    def run_evaluation(
        self, 
        output_dir: Optional[str] = None, 
        verbose: bool = True,
        n_candidates: int = 50,
        temperature: float = 0.1
    ) -> Dict:
        """
        Runs the Goldilocks evaluation suite on AI Feynman.
        """
        print("Building AIF Dataloader...")
        loader = build_aif_dataloader(
            csv_path=self.config['data']['csv_path'],
            data_dir=self.config['data']['data_dir'],
            batch_size=1,
            cache_dir=self.config['data']['cache_dir'],
        )
        
        print(f"Starting Goldilocks evaluation (N={n_candidates}, T={temperature})...")
        # Override config with CLI params if provided
        inf_cfg = self.config.get('inference', {}).copy()
        inf_cfg['pool_size'] = n_candidates
        inf_cfg['temperature'] = temperature
        
        metrics = evaluate_dataset(
            self.model, 
            loader.dataset, 
            device=self.device, 
            verbose=verbose,
            n_candidates=n_candidates,
            temperature=temperature,
            inf_config=inf_cfg
        )
        
        if output_dir:
            self._save_results(metrics, Path(output_dir))
            
        return metrics

    def _save_results(self, metrics: Dict, out_path: Path):
        """Save JSON and Markdown Goldilocks report."""
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Metrics JSON
        with open(out_path / "metrics.json", "w") as f:
            # Handle non-serialisable types
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (torch.Tensor, torch.long)) else x)
            
        # 2. Save Markdown Report
        report_path = out_path / "goldilocks_report.md"
        with open(report_path, "w") as f:
            f.write("# 🧪 LLM-JEPA Goldilocks Evaluation Report\n\n")
            f.write(f"**Checkpoint:** `{Path(self.ckpt_path).name}`\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Summary Metrics\n")
            f.write(f"- **Equations Evaluated:** {metrics['n_equations']}\n")
            f.write(f"- **Symbolic Accuracy (Skeleton Match):** {metrics.get('symbolic_accuracy', 0.0)*100:.1f}%\n")
            f.write(f"- **Constant Recovery (Parameter Match):** {metrics.get('constant_recovery', 0.0)*100:.1f}%\n")
            f.write(f"- **Median R²:** {metrics.get('median_r2', -1):.4f}\n")
            f.write(f"- **Mean NED (Edit Distance):** {metrics.get('mean_ned', 1.0):.4f}\n\n")
            
            f.write("## 🎨 Results Table\n")
            f.write("| ID | Truth | Predicted | R² | SymAcc | ConstRec | NED |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            
            for res in metrics.get('per_eq_results', []):
                eq_id = res['eq_id']
                truth = res['truth']
                pred = res['predicted'] if res['predicted'] else "N/A"
                r2 = f"{res['r2']:.4f}" if torch.isfinite(torch.tensor(res['r2'])) else "-inf"
                sa = "✅" if res['symbolic_accuracy'] else "❌"
                cr = "✅" if res.get('constant_recovery', 0.0) > 0.5 else "❌"
                ned = f"{res['ned']:.4f}"
                f.write(f"| {eq_id} | `{truth}` | `{pred}` | {r2} | {sa} | {cr} | {ned} |\n")
            
            f.write("\n**Legend:** SymAcc = Symbolic Accuracy, ConstRec = Constant Recovery\n")
        
        print(f"Results and report saved to: {out_path}")

    def predict_sample_by_id(self, eq_id: str, n_candidates: int = 50, temperature: float = 0.1) -> Optional[Dict]:
        """Runs inference on a specific equation ID."""
        loader = build_aif_dataloader(
            csv_path=self.config['data']['csv_path'],
            data_dir=self.config['data']['data_dir'],
            batch_size=1,
            cache_dir=self.config['data']['cache_dir'],
            shuffle=False
        )
        
        from evaluation.evaluate import _evaluate_one

        for eq in loader.dataset.equations:
            if eq.eq_id == eq_id:
                res = _evaluate_one(
                    self.model, 
                    eq, 
                    self.device, 
                    n_restarts=3, 
                    n_candidates=n_candidates, 
                    temperature=temperature
                )
                
                return {
                    'id': eq_id,
                    'gt': eq.formula_str,
                    'pred': res['predicted'],
                    'r2': res['r2'],
                    'sa': res['symbolic_accuracy'],
                    'ned': res['ned']
                }
        return None
