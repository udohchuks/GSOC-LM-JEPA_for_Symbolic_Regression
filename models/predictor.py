"""
JEPA Predictor for LLM-JEPA Symbolic Regression.

Purpose:
Bridges the Context Encoder (data) and Target Encoder (formula).
Predicts z_target (formula representation) from z_context (data representation).

Architecture Constraints (CRITICAL):
1. SMALL: Must be significantly smaller than the Context Encoder.
   - Why? To prevent the "lazy encoder" problem. If the Predictor is too
     powerful, it can learn the mapping without z_context being informative.
     We force the Context Encoder to extract rich physics features.

2. Inputs:
   - z_context:      [B, d_model]      Global data summary
   - var_summaries:  [B, n_vars, d_model]  Per-variable features
   - var_mask:       [B, n_vars]       To ignore padded variables

3. Output:
   - z_pred:         [B, d_model]      Predicted formula representation
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class JEPAPredictor(nn.Module):
    """
    Narrow bottleneck predictor for JEPA objective.
    
    Takes context encoder outputs and predicts the target encoder's
    representation. Deliberately constrained to force the context
    encoder to learn meaningful representations.
    
    Args:
        d_model:   Embedding dimension (input/output size)
        n_heads:   Attention heads
        dropout:   Attention dropout
    """
    def __init__(
        self,
        d_model:          int,
        n_heads:          int = 4,
        bottleneck_ratio: float = 0.5,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = max(1, int(d_model * bottleneck_ratio))

        self.z_proj = nn.Sequential(
            nn.Linear(d_model, self.bottleneck_dim),
            nn.GELU(),
            nn.RMSNorm(self.bottleneck_dim),
        )
        
        self.var_proj = nn.Sequential(
            nn.Linear(d_model, self.bottleneck_dim),
            nn.GELU(),
            nn.RMSNorm(self.bottleneck_dim),
        )

        # ── Cross-Attention ──────────────────────────────────────────────────

        self.attn = nn.MultiheadAttention(
            embed_dim=self.bottleneck_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── Output Projection ───────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(self.bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
        )
        
        # ── Gating Mechanism─────────────────────────
        self.gate = nn.Sequential(
            nn.Linear(self.bottleneck_dim, d_model),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        z_context:     torch.Tensor,
        var_summaries: torch.Tensor,
        var_mask:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict z_target from context encoder outputs.
        
        Args:
            z_context:     [B, d_model]      Global equation summary
            var_summaries: [B, n_vars, d_model]  Per-variable representations
            var_mask:      [B, n_vars]       1.0=real, 0.0=padding
            
        Returns:
            z_pred: [B, d_model]  Predicted formula representation
        """
        B = z_context.shape[0]
        
        # ── 1. Project to Bottleneck Space ───────────────────────────────────
        q = self.z_proj(z_context).unsqueeze(1)  # [B, 1, bottleneck_dim]
        k = self.var_proj(var_summaries)         # [B, n_vars, bottleneck_dim]
        v = k                                    # Same as key for standard attn


        # ── 2. Cross-Attention ───────────────────────────────────────────────
        key_padding_mask = None
        if var_mask is not None:
            key_padding_mask = (var_mask == 0.0)  # [B, n_vars]
        
        # attn_out: [B, 1, bottleneck_dim]
        attn_out, _ = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
        )

        # Squeeze the sequence dimension
        x = attn_out.squeeze(1)  # [B, bottleneck_dim]
        
        # ── 3. Gated Output Projection ───────────────────────────────────────
        z_pred = self.out_proj(x)  # [B, d_model]
        
        gate = self.gate(x)        # [B, d_model]
        z_pred = z_pred * gate
        
        return z_pred



# ── Tests ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    from models.encoder import MixEncoder
    from models.embedders import DataEmbedder, UnitEmbedder
    
    print("Testing JEPAPredictor (Small Bottleneck Design)...")
    
    B, N, n_vars, d_model = 4, 200, 9, 128  # Use 128 for clear bottleneck check
    predictor = JEPAPredictor(d_model=d_model, n_heads=4)
    
    # ── Test 1: Parameter Count Check (CRITICAL) ─────────────────────────────
    # Predictor must be significantly smaller than the encoder
    predictor_params = sum(p.numel() for p in predictor.parameters())
    
    # Build a small encoder for comparison
    encoder = MixEncoder(d_model=d_model, n_heads=4, n_isab=1, n_col_attn=1)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    
    ratio = predictor_params / encoder_params
    print(f"Predictor params: {predictor_params:,}")
    print(f"Encoder params:   {encoder_params:,}")
    print(f"Ratio:            {ratio:.2f}x")
    
    # Predictor should be < 50% of encoder size
    assert ratio < 0.5, f"Predictor too large! Ratio {ratio:.2f} should be < 0.5"
    print("✓ Test 1: Predictor is significantly smaller than Encoder")
    
    # ── Test 2: Forward Pass Shapes ──────────────────────────────────────────
    # Create fake inputs matching MixEncoder output
    z_context     = torch.randn(B, d_model)
    var_summaries = torch.randn(B, n_vars, d_model)
    var_mask      = torch.ones(B, n_vars)
    var_mask[:, 5:] = 0.0  # Pad last 4 variables
    
    z_pred = predictor(z_context, var_summaries, var_mask)
    
    assert z_pred.shape == (B, d_model), f"Expected {(B, d_model)}, got {z_pred.shape}"
    assert torch.isfinite(z_pred).all(), "Output contains NaN/Inf"
    print(f"✓ Test 2: Forward pass output shape {z_pred.shape}")
    
    # ── Test 3: Gradient Flow ────────────────────────────────────────────────
    # Verify gradients flow through predictor to inputs
    z_context.requires_grad_(True)
    var_summaries.requires_grad_(True)
    
    z_pred = predictor(z_context, var_summaries, var_mask)
    loss = z_pred.sum()
    loss.backward()
    
    assert z_context.grad is not None, "No gradient for z_context"
    assert var_summaries.grad is not None, "No gradient for var_summaries"
    assert torch.isfinite(z_context.grad).all(), "Gradient NaN/Inf"
    print("✓ Test 3: Gradients flow correctly through predictor")
    
    # ── Test 4: Variable Masking ─────────────────────────────────────────────
    # Padded variables should not affect output
    predictor.eval()  # Disable dropout for consistency check
    var_summaries_modified = var_summaries.clone()
    var_summaries_modified[:, 5:, :] = torch.randn_like(var_summaries[:, 5:, :])
    
    with torch.no_grad():
        z_pred_original = predictor(z_context, var_summaries, var_mask)
        z_pred_modified = predictor(z_context, var_summaries_modified, var_mask)
    
    # Outputs should be identical (padded vars are masked)
    assert torch.allclose(z_pred_original, z_pred_modified, atol=1e-5), \
        "Variable masking not working — padded vars affect output"
    print("✓ Test 4: Variable masking works correctly")
    predictor.train() # Reset to train mode
    
    # ── Test 5: Bottleneck Verification ──────────────────────────────────────
    # Check internal dimension is actually smaller
    assert predictor.bottleneck_dim == d_model // 2, \
        f"Bottleneck dim {predictor.bottleneck_dim} should be {d_model // 2}"
    print(f"✓ Test 5: Bottleneck dimension verified ({d_model} → {predictor.bottleneck_dim})")
    
    print("\n✅ All JEPAPredictor tests passed.")
    print(f"Predictor is ready for integration with {'SIGReg' if True else 'EMA'} loss.")