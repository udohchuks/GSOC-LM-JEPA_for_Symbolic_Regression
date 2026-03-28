"""
Inference Pipeline for LLM-JEPA Symbolic Regression.

Generates formulas from trained model and logs to TensorBoard.
Separate from training trainer.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from models.embedders import DataEmbedder, UnitEmbedder
from models.encoder import MixEncoder
from models.decoder import RPNDecoder
from data.tokenizer import (
    IDX2TOKEN, TOKEN2IDX, EOS_IDX, PAD_IDX, BOS_IDX,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    get_valid_next_tokens, ARITY, MAX_SEQ_LEN, VOCAB_SIZE,
)
from data.unit_table import N_UNIT_DIMS, N_UNIT_CLASSES
from data.aif_dataset import build_aif_dataloader

class InferenceModel(nn.Module):
    """
    Inference-only model (no target encoder, no predictor, no unit head).
    """
    def __init__(
        self,
        d_model:          int,
        n_heads:          int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        max_n_vars:       int,
        vocab_size:       int,
        max_seq_len:      int,
    ):
        super().__init__()
        self.data_embedder = DataEmbedder(d_model=d_model, max_n_vars=max_n_vars)
        self.unit_embedder = UnitEmbedder(d_model=d_model)

        self.context_encoder = MixEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_isab=2,
            n_col_attn=2,
            m_inducing=32,
            max_n_vars=max_n_vars,
        )
        
        self.decoder = RPNDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=0.0,  # No dropout at inference
        )

    def forward(
        self,
        X_bits:    torch.Tensor,
        unit_idx:  torch.Tensor,
        var_mask:  torch.Tensor,
    ) -> torch.Tensor:
        """Alias for encode() to maintain standard nn.Module call syntax."""
        return self.encode(X_bits, unit_idx, var_mask)

    def encode(
        self,
        X_bits:    torch.Tensor,
        unit_idx:  torch.Tensor,
        var_mask:  torch.Tensor,
    ) -> torch.Tensor:
        """Encode data table to z_context latent vector."""
        data_emb = self.data_embedder(X_bits)
        unit_emb = self.unit_embedder(unit_idx)
        z_context, _ = self.context_encoder(data_emb, unit_emb, var_mask)
        return z_context
    
    @torch.no_grad()
    def generate(
        self,
        z_context:   torch.Tensor,
        unit_idx:    torch.Tensor,
        max_len:     int = 30,
        greedy:      bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Batched, autoregressive generation with RPN validity masking.
        Supports both greedy and temperature-based sampling.
        """
        self.eval()
        B = z_context.shape[0]
        device = z_context.device
        
        # 1. Pre-build Arity Map for insanely fast, vectorized stack updates
        # +1 for variables/constants, 0 for unary, -1 for binary
        arity_map = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=device) # [Max size]
        for idx, tok in IDX2TOKEN.items():
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
                continue
            arity = ARITY.get(tok, 0)
            if arity == 0: arity_map[idx] = 1
            elif arity == 2: arity_map[idx] = -1

        # 2. Initialize State Tensors (No more Python lists!)
        generated = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device) # Bx1
        stack_depths = torch.zeros(B, dtype=torch.long, device=device) # B
        finished = torch.zeros(B, dtype=torch.bool, device=device) # B

        for step in range(max_len - 1):
            # Decoder forward pass
            logits, _ = self.decoder(generated, z_context, unit_idx)
            next_logits = logits[:, -1, :]  # [B, VOCAB_SIZE]

            # 3. Apply Validity Mask
            # We still need a short loop here because get_valid_next_tokens has complex rules
            mask = torch.full_like(next_logits, float('-inf'))
            for b in range(B):
                if finished[b]:
                    continue
                valid_idxs = get_valid_next_tokens(
                    stack_depth=stack_depths[b].item(),
                    seq_len=step,
                    max_len=max_len,
                )
                mask[b, valid_idxs] = 0.0
            
            next_logits = next_logits + mask

            # 4. Token Selection (Greedy or Sample)
            if greedy:
                next_tokens = next_logits.argmax(dim=-1) # [B]
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # [B]

            # 5. Handle Finished Sequences
            # If a sequence is already finished, force its next token to be PAD
            next_tokens = torch.where(finished, torch.tensor(PAD_IDX, device=device), next_tokens)
            
            # Update finished tracker (if they just predicted EOS, mark them finished)
            is_eos = (next_tokens == EOS_IDX)
            finished = finished | is_eos

            # 6. Vectorized Stack Depth Update
            # Instantly apply the arity math to the whole batch using the map
            deltas = arity_map[next_tokens]
            stack_depths = torch.clamp(stack_depths + deltas, min=0)

            # 7. Append and Check
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Pad to max_len if it finished early
        if generated.shape[1] < max_len:
            pad = torch.full(
                (B, max_len - generated.shape[1]),
                PAD_IDX, dtype=torch.long, device=device
            )
            generated = torch.cat([generated, pad], dim=1)

        return generated[:, :max_len]