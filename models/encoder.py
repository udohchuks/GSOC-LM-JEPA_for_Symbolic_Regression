"""
Mix Encoder for LLM-JEPA Symbolic Regression.

Produces z_context [B, d_model] from data and unit embeddings.

Architecture:
    1. Fuse:            add unit embeddings to data embeddings
    2. ISAB (row):      set attention over N observations per variable
                        O(Nm) — permutation invariant over rows
    3. PMA (pool):      compress N rows → 1 vector per variable
    4. Column attn:     variables attend to each other
                        no positional encoding — equivariant over variables
    5. PMA (aggregate): n_vars vectors → z_context [B, d_model]

Why this order?
    Unit info must be injected before ISAB so the row-level
    attention knows what physical type each variable is.
    Column attention comes after row aggregation because
    inter-variable relationships emerge from the per-variable
    summaries, not from individual scalar values.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── ISAB: Induced Set Attention Block ─────────────────────────────────────────

class ISAB(nn.Module):
    """
    Induced Set Attention Block from Set Transformer (Lee et al. 2019).

    Reduces O(N²) self-attention to O(Nm) using m learnable inducing points.

    Forward pass:
        H = MultiheadAttention(query=I, key=X, value=X)
            inducing points I attend over input set X
            compresses N inputs to m summaries
            shape: [B, m, d_model]

        Z = MultiheadAttention(query=X, key=H, value=H)
            input set X attends over compressed summaries H
            each element of X gets updated with global context
            shape: [B, N, d_model]

    Args:
        d_model:   embedding dimension
        n_heads:   attention heads
        m:         number of inducing points (m << N)
        dropout:   attention dropout
    """
    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        m:        int = 32,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.m = m

        # Learnable inducing points
        # Shape [1, m, d_model] — shared across the batch
        self.inducing = nn.Parameter(torch.randn(1, m, d_model))

        # First attention: inducing points → X
        self.attn1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Second attention: X → inducing summaries
        self.attn2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)

        # Feed-forward after second attention
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, N, d_model]  input set (N elements)

        Returns:
            [B, N, d_model]  each element updated with global context
        """
        B = X.shape[0]

        # Expand inducing points to batch size
        I = self.inducing.expand(B, -1, -1)   # [B, m, d_model]

        # First attention: I queries X
        # Inducing points summarise the full set
        H, _ = self.attn1(query=I, key=X, value=X)
        H    = self.norm1(H + I)               # residual + norm
        # H: [B, m, d_model]

        # Second attention: X queries H
        # Each element gets context from the compressed summary
        Z, _ = self.attn2(query=X, key=H, value=H)
        Z    = self.norm2(Z + X)               # residual + norm
        # Z: [B, N, d_model]
        
        # Feed-forward
        Z = self.norm3(Z + self.ff(Z))

        return Z

# ── PMA: Pooling by Multihead Attention ───────────────────────────────────────

class PMA(nn.Module):
    """
    Pooling by Multihead Attention from Set Transformer (Lee et al. 2019).

    Aggregates a set of N vectors into k summary vectors using
    k learnable seed vectors as queries.

    For producing z_context (k=1): one summary vector for the whole equation.
    For producing per-variable summaries after ISAB (k=1 per variable):
        this is called per-variable inside MixEncoder.

    Args:
        d_model:  embedding dimension
        n_heads:  attention heads
        k:        number of output vectors (seed vectors)
        dropout:  attention dropout
    """
    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        k:        int = 1,
        dropout:  float = 0.1,
    ):
        super().__init__()

        # Learnable seed vectors — the queries
        self.seeds = nn.Parameter(torch.randn(1, k, d_model))

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    
    def forward(
        self,
        X: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            X: [B, N, d_model]  set of N vectors
            key_padding_mask: [B, N] indicates which elements are padding (True)

        Returns:
            [B, k, d_model]  k summary vectors
        """
        B = X.shape[0]

        seeds = self.seeds.expand(B, -1, -1)   # [B, k, d_model]

        out, _ = self.attn(
            query=seeds,
            key=X,
            value=X,
            key_padding_mask=key_padding_mask,
        )
        out    = self.norm(out + seeds)         # residual + norm

        out    = self.norm2(out + self.ff(out))

        return out   # [B, k, d_model]


# ── Mix Encoder ───────────────────────────────────────────────────────────────

class MixEncoder(nn.Module):
    """
    Mix Encoder: fuses data and unit embeddings, applies ISAB over rows
    then self-attention over variables to produce z_context.

    Input:
        data_emb:  [B, N, n_vars, d_model]  from DataEmbedder
        unit_emb:  [B, n_vars, d_model]     from UnitEmbedder
        var_mask:  [B, n_vars]              1.0=real, 0.0=padding

    Output:
        z_context:     [B, d_model]         global equation summary
        var_summaries: [B, n_vars, d_model] per-variable summaries
                       (used by decoder cross-attention and JEPA predictor)

    Steps:
        1. Fuse unit embeddings into data embeddings
        2. ISAB over rows (per variable independently)
        3. PMA: compress N rows → 1 vector per variable
        4. Column self-attention: variables attend to each other
        5. PMA: aggregate n_vars → z_context
    """
    def __init__(
        self,
        d_model:    int,
        n_heads:    int = 8,
        n_isab:     int = 2,       # number of ISAB blocks
        n_col_attn: int = 2,       # number of column attention layers
        m_inducing: int = 32,      # inducing points per ISAB
        max_n_vars: int = 9,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_n_vars = max_n_vars

        # Step 1: project fused embedding to d_model
        # Input: data_emb + broadcast unit_emb = d_model (same dim, just add)
        # A LayerNorm stabilises after addition
        self.fuse_norm = nn.RMSNorm(d_model)

        # Step 2: ISAB stack (row-level, per variable)
        self.isab_blocks = nn.ModuleList([
            ISAB(d_model=d_model, n_heads=n_heads,
                 m=m_inducing, dropout=dropout)
            for _ in range(n_isab)
        ])

        # Step 3: PMA to compress N rows → 1 per variable
        self.row_pma = PMA(
            d_model=d_model, n_heads=n_heads, k=1, dropout=dropout
        )

        #Step 4: column-level self-attention
        # Standard transformer encoder layers, NO positional encoding
        col_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,    # pre-norm: more stable training
        )
        self.col_attn = nn.TransformerEncoder(
            col_layer,
            num_layers=n_col_attn,
        )

        # Step 5: PMA to aggregate n_vars → z_context
        self.context_pma = PMA(
            d_model=d_model, n_heads=n_heads, k=1, dropout=dropout
        )

        self.output_norm = nn.RMSNorm(d_model)
    
    def forward(
        self,
        data_emb:  torch.Tensor,
        unit_emb:  torch.Tensor,
        var_mask:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data_emb:  [B, N, n_vars, d_model]
            unit_emb:  [B, n_vars, d_model]
            var_mask:  [B, n_vars] optional

        Returns:
            z_context:     [B, d_model]
            var_summaries: [B, n_vars, d_model]
        """
        B, N, V, D = data_emb.shape
        # ── Step 1: Fuse unit info into data embeddings ───────────────────
        # unit_emb: [B, n_vars, d_model] → [B, 1, n_vars, d_model]
        # Broadcast adds same unit info to all N rows of each variable
        fused = data_emb + unit_emb.unsqueeze(1)
        fused = self.fuse_norm(fused)

        # ── Step 2: ISAB over rows — process each variable independently ──
        # Reshape: treat each variable as a separate set of N elements
        # [B, N, n_vars, d_model] → [B*n_vars, N, d_model]
        x = fused.permute(0, 2, 1, 3)          # [B, n_vars, N, d_model]
        x = x.reshape(B * V, N, D)             # [B*n_vars, N, d_model]


        for isab in self.isab_blocks:
            x = isab(x)
        # x: [B*n_vars, N, d_model]

        # ── Step 3: PMA — compress N rows → 1 per variable ───────────────
        var_vecs = self.row_pma(x)              # [B*n_vars, 1, d_model]
        var_vecs = var_vecs.squeeze(1)          # [B*n_vars, d_model]
        var_vecs = var_vecs.reshape(B, V, D)    # [B, n_vars, d_model]

        # Apply variable mask: zero out padded variable slots
        if var_mask is not None:
            var_vecs = var_vecs * var_mask.unsqueeze(-1)
        
        # ── Step 4: Column-level self-attention ───────────────────────────
        # Variables attend to each other
        key_padding_mask = None
        if var_mask is not None:
            key_padding_mask = (var_mask == 0.0)
            # [B, n_vars] where True means "this variable is padding"
        
        var_summaries = self.col_attn(
            var_vecs,
            src_key_padding_mask=key_padding_mask,
        )
        # var_summaries: [B, n_vars, d_model]

        # ── Step 5: PMA — aggregate n_vars → z_context ───────────────────
        z = self.context_pma(var_summaries, key_padding_mask=key_padding_mask) # [B, 1, d_model]
        z = z.squeeze(1)                        # [B, d_model]
        z = self.output_norm(z)

        return z, var_summaries

if __name__ == '__main__':
    from models.embedders import DataEmbedder, UnitEmbedder

    B, N, n_vars, d_model = 4, 200, 9, 64

    # Create embedders and encoder
    data_embedder = DataEmbedder(d_model=d_model, max_n_vars=n_vars)
    unit_embedder = UnitEmbedder(d_model=d_model)
    encoder       = MixEncoder(d_model=d_model, n_heads=4,
                                n_isab=2, n_col_attn=2, m_inducing=16)

    # Fake inputs
    X_bits   = torch.randint(0, 2, (B, N, n_vars, 16)).float()
    unit_idx = torch.randint(0, 9, (B, n_vars, 5))
    var_mask = torch.ones(B, n_vars)
    var_mask[:, 3:] = 0.0   # only 3 real variables

    # Forward pass through embedders
    data_emb = data_embedder(X_bits)
    unit_emb = unit_embedder(unit_idx)

    assert data_emb.shape == (B, N, n_vars, d_model)
    assert unit_emb.shape == (B, n_vars, d_model)
    print(f"data_emb: {data_emb.shape} — OK")
    print(f"unit_emb: {unit_emb.shape} — OK")

    # Forward pass through encoder
    encoder.eval()
    data_embedder.eval()
    unit_embedder.eval()

    with torch.no_grad():
        z_context, var_summaries = encoder(data_emb, unit_emb, var_mask)

    assert z_context.shape     == (B, d_model)
    assert var_summaries.shape == (B, n_vars, d_model)
    print(f"z_context:     {z_context.shape} — OK")
    print(f"var_summaries: {var_summaries.shape} — OK")

    # ── Row permutation invariance ────────────────────────────────────────
    perm           = torch.randperm(N)
    with torch.no_grad():
        data_emb_perm  = data_embedder(X_bits[:, perm, :, :])
        z_perm, _      = encoder(data_emb_perm, unit_emb, var_mask)

    assert torch.allclose(z_context, z_perm, atol=1e-4), \
        "MixEncoder is not row-permutation invariant"
    print("Row permutation invariance: OK")

    # ── Column shuffle equivariance ───────────────────────────────────────
    # Shuffling variable columns should change z_context
    # (equivariance not invariance — the output changes accordingly)
    # But shuffling should not crash or produce NaN
    col_perm     = torch.randperm(n_vars)
    with torch.no_grad():
        data_emb_col = data_embedder(X_bits[:, :, col_perm, :])
        unit_emb_col = unit_embedder(unit_idx[:, col_perm, :])
        mask_col     = var_mask[:, col_perm]

        z_col, _ = encoder(data_emb_col, unit_emb_col, mask_col)
    assert torch.isfinite(z_col).all(), "Column shuffle produced NaN/Inf"
    print("Column shuffle: no NaN/Inf — OK")

    # ── No NaN in outputs ─────────────────────────────────────────────────
    assert torch.isfinite(z_context).all()
    assert torch.isfinite(var_summaries).all()
    print("No NaN/Inf in outputs: OK")

    print(f"\nAll encoder tests passed.")
    print(f"Parameter count: "
          f"{sum(p.numel() for p in encoder.parameters()):,}")