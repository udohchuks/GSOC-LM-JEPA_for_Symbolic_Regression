"""
PyTorch Lightning Trainer for LLM-JEPA Symbolic Regression.

Training-specific components:
- Full model with context encoder, target encoder, predictor, decoder
- JEPA loss with perturbation curriculum
- EMA updates for target encoder
- TensorBoard logging with training visualizations

Inference uses a separate pipeline (inference/generate.py).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from typing import Dict, List, Optional, Any
import numpy as np

from models.model import LLMJEPA
from training.losses import LLMJEPALoss
from data.tokenizer import IDX2TOKEN, EOS_IDX, PAD_IDX, BOS_IDX

class LLMJEPAModule(pl.LightningModule):
    """
    Lightning Module for LLM-JEPA training.
    
    SIGReg Note:
    Both context_encoder and target_encoder are trainable.
    Target encoder uses EMA updates after each optimizer step.
    """
    def __init__(
        self,
        d_model:        int = 64,
        n_heads:        int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        max_n_vars:     int = 9,
        vocab_size:     int = 41,
        max_seq_len:    int = 45,
        learning_rate:  float = 1e-4,
        weight_decay:   float = 1e-2,
        warmup_steps:   int = 500,
        # Architecture
        n_isab:         int = 2,
        n_col_attn:     int = 2,
        m_inducing:     int = 32,
        dropout:        float = 0.1,
        # Predictor
        pred_n_heads:          int = 4,
        pred_bottleneck_ratio: float = 0.5,
        pred_dropout:          float = 0.1,
        # Loss weights
        lambda_jepa:    float = 1.0,
        lambda_sigreg:  float = 1.0,
        lambda_lm:      float = 1.0,
        lambda_units:   float = 1.0,
        # SIGReg / LM loss params
        sigreg_num_slices: int = 512,
        sigreg_num_points: int = 17,
        invalid_weight:    float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── 1. Initialize Unified Model ──────────────────────────────────────
        self.model = LLMJEPA(
            d_model=d_model,
            n_heads=n_heads,
            n_isab=n_isab,
            n_col_attn=n_col_attn,
            n_enc_layers=n_encoder_layers,
            n_dec_layers=n_decoder_layers,
            m_inducing=m_inducing,
            max_n_vars=max_n_vars,
            dropout=dropout,
            # Predictor params
            pred_n_heads=pred_n_heads,
            pred_bottleneck_ratio=pred_bottleneck_ratio,
            pred_dropout=pred_dropout,
        )


        # Loss Function
        self.loss_fn = LLMJEPALoss(
            lambda_jepa=lambda_jepa,
            lambda_sigreg=lambda_sigreg,
            lambda_lm=lambda_lm,
            lambda_units=lambda_units,
            sigreg_num_slices=sigreg_num_slices,
            sigreg_num_points=sigreg_num_points,
            invalid_weight=invalid_weight,
        )

        # For logging
        self.validation_outputs = []

        # Example input array for TensorBoard graph logging
        self.example_input_array = ({
            'X_bits': torch.zeros((1, 10, max_n_vars, 16), dtype=torch.float32),
            'unit_idx': torch.zeros((1, max_n_vars, 5), dtype=torch.long),
            'var_mask': torch.ones((1, max_n_vars), dtype=torch.bool),
            'token_ids': torch.zeros((1, max_seq_len), dtype=torch.long),
            'unit_targets_idx': torch.zeros((1, max_seq_len, 5), dtype=torch.long),
        },)
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Full forward pass for training.
        Returns dict containing losses and intermediates for logging.
        """

        # ── 1. Prepare Inputs ──────────────────────────────────────────────
        X_bits      = batch['X_bits']          # [B, N, n_vars, 16]
        unit_idx    = batch['unit_idx']        # [B, n_vars, 5]
        var_mask    = batch['var_mask']        # [B, n_vars]
        token_ids   = batch['token_ids']       # [B, T]
        unit_targets = batch['unit_targets_idx'] # [B, T, 5]

        B, N, V, _ = X_bits.shape

        # ── 2. Forward Pass on Unified Model ───────────────────────────────
        out = self.model(
            X_bits=X_bits,
            unit_idx=unit_idx,
            var_mask=var_mask,
            token_ids=token_ids,
            unit_targets_idx=unit_targets
        )

        # ── 3. Unit Prediction Head ────────────────────────────────────────
        unit_preds = self.model.unit_head(out['h_states'])

        # ── 4. Compute Losses ──────────────────────────────────────────────
        decoder_target = token_ids[:, 1:]
        unit_targets_shifted = unit_targets[:, 1:, :]

        losses = self.loss_fn(
            z_pred=out['z_hat'],
            z_target=out['z_target'],
            z_context=out['z_context'],
            logits=out['logits'],
            token_targets=decoder_target,
            unit_preds=unit_preds,
            unit_targets=unit_targets_shifted,
            token_ids=token_ids[:, :-1],   # decoder input [BOS, t1 .. t_{T-2}], same length as targets
        )

        return {
            'loss': losses['total'],
            'losses': losses,
            'logits': out['logits'],
            'targets': decoder_target,
            'z_context': out['z_context'],
            'z_target': out['z_target'],
        }
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Single training step."""
        outputs = self.forward(batch)
        loss = outputs['loss']
        losses = outputs['losses']
        
        # Log all loss components
        self.log('train/total', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/jepa', losses['jepa'], on_step=True, on_epoch=True)
        self.log('train/sigreg', losses['sigreg'], on_step=True, on_epoch=True)
        self.log('train/lm', losses['lm'], on_step=True, on_epoch=True)
        self.log('train/units', losses['units'], on_step=True, on_epoch=True)
        
        return loss
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Single validation step."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            loss = outputs['loss']
            losses = outputs['losses']

        # Log validation losses
        self.log('val/total', loss, on_epoch=True, prog_bar=True)
        self.log('val/jepa', losses['jepa'], on_epoch=True)
        self.log('val/lm', losses['lm'], on_epoch=True)

        return outputs

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Single test step — run on the AIF (Feynman) dataset after training."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            loss = outputs['loss']
            losses = outputs['losses']

        # Log test losses (all components for thorough reporting)
        self.log('test/total',  loss,             on_epoch=True, prog_bar=True)
        self.log('test/jepa',   losses['jepa'],   on_epoch=True)
        self.log('test/sigreg', losses['sigreg'], on_epoch=True)
        self.log('test/lm',     losses['lm'],     on_epoch=True)
        self.log('test/units',  losses['units'],  on_epoch=True)

        return outputs


    def _lr_lambda(self, step: int) -> float:
        """Linear warmup + cosine decay helper for pickle compatibility."""
        import math
        warmup_steps = self.hparams.warmup_steps
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # In some PL versions, accessing this before/during setup can be tricky
        try:
            max_steps = self.trainer.estimated_stepping_batches
        except Exception:
            # Safe fallback: epochs * batches_per_epoch (approx)
            max_steps = 100000 
        
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def configure_optimizers(self) -> Dict:
        """Configure AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self._lr_lambda
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

