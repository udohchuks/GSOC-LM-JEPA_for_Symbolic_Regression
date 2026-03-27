"""
LLM-JEPA Full Model Assembly.
 
Combines all components into one nn.Module:
    DataEmbedder    → per-scalar IEEE-754 embeddings
    UnitEmbedder    → per-variable unit embeddings
    MixEncoder      → z_context + var_summaries
    TargetEncoder   → z_target from formula tokens
    JEPAPredictor   → z_hat from z_context + var_summaries
    RPNDecoder      → RPN token logits from z_context
    UnitPredHead    → unit class logits from decoder hidden states
 
Training:
    L_jepa   = MSE(z_hat, z_target)
    L_sigreg = SIGReg(z_context) + SIGReg(z_target)
    L_lm     = validity-weighted CrossEntropy
    L_units  = mean CE over 5 unit dimensions
    L_total  = L_jepa + L_sigreg + L_lm + L_units
 
Inference:
    Only DataEmbedder, UnitEmbedder, MixEncoder, RPNDecoder are used.
    TargetEncoder, JEPAPredictor, UnitPredHead are discarded.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
 
from models.embedders     import DataEmbedder, UnitEmbedder
from models.encoder       import MixEncoder
from models.target_encoder import TargetEncoder
from models.predictor     import JEPAPredictor
from models.decoder       import RPNDecoder, UnitPredictionHead
from data.tokenizer       import (
    PAD_IDX, ARITY, IDX2TOKEN, VOCAB_SIZE,
    get_valid_next_tokens, MAX_SEQ_LEN, BOS_IDX, EOS_IDX,
)


class LLMJEPA(nn.Module):
    """
    Full LLM-JEPA model for symbolic regression.

    Args:
        d_model:      embedding dimension
        n_heads:      attention heads (all components)
        n_isab:       ISAB blocks in MixEncoder
        n_col_attn:   column attention layers in MixEncoder
        n_enc_layers: transformer layers in TargetEncoder
        n_dec_layers: transformer decoder layers
        m_inducing:   inducing points per ISAB block
        max_n_vars:   maximum number of variables (padding size)
        dropout:      dropout rate
    """

    def __init__(
        self,
        d_model:      int = 256,
        n_heads:      int = 8,
        n_isab:       int = 2,
        n_col_attn:   int = 2,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        m_inducing:   int = 32,
        max_n_vars:   int = 9,
        dropout:      float = 0.1,
        # Predictor params
        pred_n_heads:          int = 4,
        pred_bottleneck_ratio: float = 0.5,
        pred_dropout:          float = 0.1,
    ):
        super().__init__()

        self.d_model    = d_model
        self.max_n_vars = max_n_vars

        # ── Encoder side ───────────────────────────────────────────────────
        self.data_embedder = DataEmbedder(
            d_model=d_model,
            max_n_vars=max_n_vars,
        )
        self.unit_embedder = UnitEmbedder(d_model=d_model)
        self.mix_encoder   = MixEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_isab=n_isab,
            n_col_attn=n_col_attn,
            m_inducing=m_inducing,
            max_n_vars=max_n_vars,
            dropout=dropout,
        )

        # ── JEPA components ────────────────────────────────────────────────
        self.target_encoder = TargetEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
        )

        self.predictor = JEPAPredictor(
            d_model=d_model,
            n_heads=pred_n_heads,
            bottleneck_ratio=pred_bottleneck_ratio,
            dropout=pred_dropout,
        )

        # ── Decoder side ───────────────────────────────────────────────────
        self.decoder = RPNDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            dropout=dropout,
        )
        self.unit_head = UnitPredictionHead(d_model=d_model)

    def encode(
        self,
        X_bits:   torch.Tensor,
        unit_idx: torch.Tensor,
        var_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode data table to z_context and var_summaries.

        Args:
            X_bits:   [B, N, n_vars, 16]
            unit_idx: [B, n_vars, 5]
            var_mask: [B, n_vars]

        Returns:
            z_context:     [B, d_model]
            var_summaries: [B, n_vars, d_model]
        """
        data_emb = self.data_embedder(X_bits)
        unit_emb = self.unit_embedder(unit_idx)
        return self.mix_encoder(data_emb, unit_emb, var_mask)

    def forward(
        self,
        X_bits:           torch.Tensor,
        unit_idx:         torch.Tensor,
        var_mask:         torch.Tensor,
        token_ids:        torch.Tensor,
        unit_targets_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full training forward pass.

        Args:
            X_bits:           [B, N, n_vars, 16]
            unit_idx:         [B, n_vars, 5]
            var_mask:         [B, n_vars]
            token_ids:        [B, T]  ground truth RPN (includes BOS)
            unit_targets_idx: [B, T, 5]

        Returns:
            dict with keys: z_context, z_target, z_hat,
                            logits, h_states
        """
        # ── Encode data ────────────────────────────────────────────────────
        z_context, var_summaries = self.encode(X_bits, unit_idx, var_mask)

        # ── Encode formula (target) ────────────────────────────────────────
        # target_encoder sees the full unmasked formula
        # z_target gradients only come from SIGReg — not from MSE
        z_target = self.target_encoder(token_ids, unit_idx)

        # ── JEPA prediction ────────────────────────────────────────────────
        z_hat = self.predictor(z_context, var_summaries, var_mask)

        # ── Decode formula ────────────────────────────────────────────────
        # Teacher forcing: feed ground truth tokens shifted right
        # Input:  [BOS, t1, t2, ..., t_{T-1}]
        # Target: [t1, t2, ..., t_{T-1}, EOS]
        decoder_input  = token_ids[:, :-1]           # drop last
        logits, h_states = self.decoder(
            decoder_input, z_context, unit_idx
        )

        return {
            'z_context':     z_context,      # [B, d_model]
            'z_target':      z_target,        # [B, d_model]
            'z_hat':         z_hat,           # [B, d_model]
            'logits':        logits,          # [B, T-1, vocab_size]
            'h_states':      h_states,        # [B, T-1, d_model]
        }


if __name__ == '__main__':
    B, N, n_vars, d_model = 2, 50, 4, 64
    T = MAX_SEQ_LEN

    model = LLMJEPA(
        d_model=d_model, n_heads=4, n_isab=1,
        n_col_attn=1, n_enc_layers=2, n_dec_layers=2,
        m_inducing=8, max_n_vars=9,
    )

    X_bits   = torch.randint(0, 2, (B, N, 9, 16)).float()
    unit_idx = torch.randint(0, 9, (B, 9, 5))
    var_mask = torch.ones(B, 9)
    var_mask[:, n_vars:] = 0.0

    token_ids        = torch.zeros(B, T, dtype=torch.long)
    token_ids[:, 0]  = BOS_IDX
    token_ids[:, 1]  = 4   # x1
    token_ids[:, 2]  = 12  # sin
    token_ids[:, 3]  = EOS_IDX

    unit_targets = torch.randint(0, 9, (B, T, 5))

    model.train()
    out = model(X_bits, unit_idx, var_mask, token_ids, unit_targets)

    assert out['z_context'].shape == (B, d_model)
    assert out['z_target'].shape  == (B, d_model)
    assert out['z_hat'].shape     == (B, d_model)
    assert out['logits'].shape    == (B, T - 1, VOCAB_SIZE)
    print("Training forward pass: OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
    print("Full model test passed.")
