"""
Embedders for LLM-JEPA Symbolic Regression.

DataEmbedder:  IEEE-754 scalars → per-scalar embeddings + variable identity
               Output: [B, N, n_vars, d_model]
               No row aggregation here — that happens in MixEncoder via ISAB.

UnitEmbedder:  unit class indices → per-variable unit embeddings
               Output: [B, n_vars, d_model]
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from data.unit_table import N_UNIT_DIMS, N_UNIT_CLASSES

class DataEmbedder(nn.Module):
    """
    Embeds IEEE-754 encoded scalars into per-scalar representations.

    Input:
        X_bits: [B, N, n_vars, 16]  IEEE-754 encoded data table

    Output:
        [B, N, n_vars, d_model]  one embedding per scalar value

    Steps:
        1. Linear(16, d_model): embed each scalar independently
        2. Add variable identity embedding: which column is this?


    Why variable identity embedding?
        After IEEE-754 encoding, x1=2.0 and x2=2.0 produce
        identical bit patterns. The variable identity embedding
        tells the model which column a scalar came from.
        This is added here so ISAB receives variable-aware inputs.
    """

    def __init__(
        self,
        d_model:    int,
        max_n_vars: int = 9,
    ):
        super().__init__()
        self.d_model = d_model

        # Embed each IEEE-754 scalar (16 bits → d_model)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Variable identity: which column is this scalar from?
        # x1 gets embedding 0, x2 gets embedding 1, etc.
        # No row positional encoding — rows are an unordered set
        self.var_embed = nn.Embedding(max_n_vars, d_model)


    def forward(self, X_bits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_bits: [B, N, n_vars, 16]

        Returns:
            [B, N, n_vars, d_model]
        """

        B, N, V, _ = X_bits.shape

        # Embed each scalar: reshape to apply Linear to all scalars at once
        x = X_bits.reshape(B * N * V, 16)
        x = self.scalar_encoder(x)              # [B*N*V, d_model]
        x = x.reshape(B, N, V, self.d_model)    # [B, N, n_vars, d_model]


        # Add variable identity embedding
        var_ids = torch.arange(V, device=X_bits.device)
        var_emb = self.var_embed(var_ids)        # [n_vars, d_model]

        # Broadcast: same identity added to all N rows of each variable
        x = x + var_emb.unsqueeze(0).unsqueeze(0)
        # [B, N, n_vars, d_model]

        return x

class UnitEmbedder(nn.Module):
    """
    Embeds physical unit class indices into per-variable representations.

    Input:
        unit_idx: [B, n_vars, 5]  class indices in [0, 8]

    Output:
        [B, n_vars, d_model]

    5 separate embedding tables, one per unit dimension [m, s, kg, K, V].
    Output = sum of all 5 embeddings.
    """
    def __init__(
        self,
        d_model:        int,
        n_unit_dims:    int = N_UNIT_DIMS,
        n_unit_classes: int = N_UNIT_CLASSES,
    ):
        super().__init__()
        self.n_unit_dims = n_unit_dims

        self.unit_embeds = nn.ModuleList([
            nn.Embedding(n_unit_classes, d_model)
            for _ in range(n_unit_dims)
        ])

        self.norm = nn.LayerNorm(d_model)


    def forward(self, unit_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            unit_idx: [B, n_vars, 5]

        Returns:
            [B, n_vars, d_model]
        """
        emb = self.unit_embeds[0](unit_idx[..., 0]) # meters emb
        for i in range(1, self.n_unit_dims):
            emb = emb + self.unit_embeds[i](unit_idx[..., i])
        return self.norm(emb)