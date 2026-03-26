"""
RPN Decoder and Unit Prediction Head for LLM-JEPA Symbolic Regression.

RPNDecoder:
    Autoregressive causal transformer decoder.
    Cross-attends to z_context at every layer.
    Generates RPN token sequences token by token.

UnitPredictionHead:
    5 linear probes on decoder hidden state h_t.
    Predicts required units at each generation step.
    Training scaffold only — discarded at inference.
    Deliberately small (single Linear per dimension) so that
    dimensional reasoning is learned in h_t, not in the probes.

PhysicsTokenEmbedding:
    Combines learned token embedding with fixed unit signatures.
    Makes dimensional structure part of every variable representation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from data.tokenizer import (
    VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX,
    TOKEN2IDX, IDX2TOKEN, ARITY,
    VARIABLE_TOKENS, CONSTANT_TOKENS, BINARY_TOKENS, UNARY_TOKENS,
    MAX_SEQ_LEN, get_valid_next_tokens,
)
from data.unit_table import N_UNIT_DIMS, N_UNIT_CLASSES, UNIT_OFFSET
from models.embedders import UnitEmbedder

class PhysicsTokenEmbedding(nn.Module):
    """
    Token embedding augmented with variable unit embeddings.

    For variable tokens (x1...x9):
        embedding = token_embed(id) + pos_embed(pos) + unit_embed(units)

    For all other tokens (operators, constants):
        embedding = token_embed(id) + pos_embed(pos)

    Unit embeddings come from the same UnitEmbedder design:

    Args:
        d_model:     embedding dimension
        vocab_size:  vocabulary size
        max_seq_len: max sequence length
        dropout:     embedding dropout
    """

    def __init__(self, d_model, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.unit_embedder = UnitEmbedder(d_model=d_model) 
        
        self.norm = nn.RMSNorm(d_model) 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids, unit_matrix=None):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embed(token_ids) + self.pos_embed(positions)

        if unit_matrix is not None:
            x = self._add_variable_units(x, token_ids, unit_matrix)
        return self.dropout(self.norm(x))
    
    def _add_variable_units(self, x, token_ids, unit_matrix):
        x = x.clone()
        for var_idx, var_tok in enumerate(VARIABLE_TOKENS):
            if var_tok not in TOKEN2IDX: continue
            tok_id = TOKEN2IDX[var_tok]

            if var_idx >= unit_matrix.shape[1]: continue # this variable not in this equation

            mask = (token_ids == tok_id)   # [B, T]
            if not mask.any(): continue

            var_unit_idx = unit_matrix[:, var_idx, :]   # [B, 5]

            unit_emb = self.unit_embedder(var_unit_idx.unsqueeze(1)) # [B, 1, d_model]

            x = x + mask.unsqueeze(-1).float() * unit_emb
        return x
    
# ── RPN Decoder ───────────────────────────────────────────────────────────────

class RPNDecoder(nn.Module):
    """
    Autoregressive causal transformer decoder for RPN formula generation.

    At each step t, generates the next token conditioned on:
        - all previously generated tokens (causal self-attention)
        - the full equation context (cross-attention to z_context)

    Architecture:
        4 transformer decoder layers
        Each layer: causal self-attention + cross-attention + FFN
        Cross-attention memory: z_context unsqueezed to [B, 1, d_model]

    Weight tying:
        LM head shares weights with token embedding matrix.
        This is standard practice — reduces parameters and
        improves generalisation by tying input and output spaces.

    z_context dropout:
        During training, small dropout on z_context prepares the
        decoder for LSO (latent space optimisation at inference)
        where z_context is perturbed to find better formulas.

    Args:
        d_model:       embedding dimension
        n_heads:       attention heads
        n_layers:      number of decoder layers
        vocab_size:    output vocabulary size
        max_seq_len:   maximum sequence length
        dropout:       attention and embedding dropout
    """
    def __init__(
        self,
        d_model:     int,
        n_heads:     int = 8,
        n_layers:    int = 4,
        vocab_size:  int = VOCAB_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size

        # Physics-aware token embedding
        self.embedding = PhysicsTokenEmbedding(
            d_model=d_model,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Casual transformer

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True, 
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: lm_head and token_embed share weights
        self.lm_head.weight = self.embedding.token_embed.weight

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_ids:   torch.Tensor,
        z_context:   torch.Tensor,
        unit_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing forward pass for training.

        Args:
            token_ids:   [B, T]           input token sequence 
            z_context:   [B, d_model]     encoder output
            unit_matrix: [B, n_vars, 5]   unit class indices (optional)

        Returns:
            logits:   [B, T, vocab_size]  next-token predictions
            h_states: [B, T, d_model]     decoder hidden states
                      (used by UnitPredictionHead)
        """

        B, T = token_ids.shape
        # ── Embed tokens with physics-aware embeddings ────────────────────
        tgt = self.embedding(token_ids, unit_matrix)
        # tgt: [B, T, d_model]

        # ── Prepare memory for cross-attention ────────────────────────────
        # z_context: [B, d_model] → [B, 1, d_model]
        # The decoder cross-attends to this single vector at every layer
        memory = z_context.unsqueeze(1)

        # ── Causal mask ───────────────────────────────────────────────────
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=token_ids.device
        )
        

        # ── Padding mask ──────────────────────────────────────────────────
        # Ignore PAD tokens in the input sequence
        pad_mask_bool = (token_ids == PAD_IDX)   # [B, T] True = ignore
        pad_mask = torch.zeros(
            (B, T), 
            dtype=torch.float32, 
            device=token_ids.device
        )
        #    Fill -inf where padding exists (if any)
        pad_mask.masked_fill_(pad_mask_bool, float('-inf'))


        # ── Transformer decoder ───────────────────────────────────────────
        h_states = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
            tgt_is_causal=True,
        )
        # h_states: [B, T, d_model]



        h_states = self.norm(h_states)

        # ── LM head ───────────────────────────────────────────────────────
        logits = self.lm_head(h_states)
        # logits: [B, T, vocab_size]

        return logits, h_states



class UnitPredictionHead(nn.Module):
    """
    Predicts required physical units at each decoder step.

    5 independent linear classifiers, one per unit dimension [m, s, kg, K, V].
    Each classifier: d_model → 9 classes (unit exponents -4 to +4).

    Why single Linear layers (no hidden layers)?
        The learning must happen in the decoder hidden states h_t.
        

    This is a training scaffold — discarded at inference.
    Its role is to force h_t to linearly encode dimensional structure.

    Args:
        d_model:        decoder hidden state dimension
        n_unit_dims:    number of unit dimensions (5)
        n_unit_classes: number of classes per dimension (9)
    """
    def __init__(
        self,
        d_model:        int,
        n_unit_dims:    int = N_UNIT_DIMS,
        n_unit_classes: int = N_UNIT_CLASSES,
    ):
        super().__init__()
        # 5 separate single-layer classifiers
        self.heads = nn.ModuleList([
            nn.Linear(d_model, n_unit_classes)
            for _ in range(n_unit_dims)
        ])
    
    def forward(
        self,
        h_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            h_states: [B, T, d_model]  decoder hidden states

        Returns:
            List of 5 tensors, each [B, T, n_unit_classes]
            One per unit dimension [m, s, kg, K, V]
        """
        return [head(h_states) for head in self.heads]

    def loss(
        self,
        h_states:          torch.Tensor,
        unit_targets_idx:  torch.Tensor,
        ignore_index:      int = -100,
    ) -> torch.Tensor:
        """
        Compute mean cross-entropy loss over all 5 unit dimensions.

        Args:
            h_states:         [B, T, d_model]
            unit_targets_idx: [B, T, 5]  class indices in [0, 8]
            ignore_index:     class index to ignore in loss

        Returns:
            scalar loss
        """
        predictions = self.forward(h_states)
        # predictions: list of 5 tensors [B, T, 9]

        total_loss = torch.tensor(0.0, device=h_states.device)

        for dim_idx, pred in enumerate(predictions):
            target = unit_targets_idx[:, :, dim_idx]  # [B, T]

            # Verify shapes before cross_entropy
            assert pred.shape[:2] == target.shape, f"Shape mismatch: {pred.shape[:2]} vs {target.shape}"

            loss = F.cross_entropy(pred.reshape(-1, 9), target.reshape(-1))
            total_loss += loss
        return total_loss / len(predictions)


if __name__ == '__main__':
    from models.embedders import DataEmbedder, UnitEmbedder
    from models.encoder   import MixEncoder

    B, N, n_vars, d_model = 2, 100, 4, 64
    T = MAX_SEQ_LEN

    # Build full pipeline
    data_embedder = DataEmbedder(d_model=d_model, max_n_vars=9)
    unit_embedder = UnitEmbedder(d_model=d_model)
    encoder       = MixEncoder(d_model=d_model, n_heads=4,
                                n_isab=1, n_col_attn=1, m_inducing=8)
    decoder       = RPNDecoder(d_model=d_model, n_heads=4,
                                n_layers=2)
    unit_head     = UnitPredictionHead(d_model=d_model)

    # ── Fake inputs ───────────────────────────────────────────────────────
    X_bits   = torch.randint(0, 2, (B, N, 9, 16)).float()
    unit_idx = torch.randint(0, 9, (B, 9, 5))
    var_mask = torch.ones(B, 9)
    var_mask[:, n_vars:] = 0.0

    # Fake token sequence: BOS + some tokens + EOS + PAD
    token_ids = torch.zeros(B, T, dtype=torch.long)
    token_ids[:, 0] = BOS_IDX
    token_ids[:, 1] = TOKEN2IDX.get('x1', 4)
    token_ids[:, 2] = TOKEN2IDX.get('sin', 10)
    token_ids[:, 3] = EOS_IDX
    # rest stay 0 = PAD_IDX

    # Fake unit targets
    unit_targets = torch.randint(0, 9, (B, T, 5))

    # ── Encoder forward pass ──────────────────────────────────────────────
    data_emb         = data_embedder(X_bits)
    unit_emb         = unit_embedder(unit_idx)
    z_context, var_s = encoder(data_emb, unit_emb, var_mask)

    assert z_context.shape == (B, d_model)
    print(f"z_context: {z_context.shape} — OK")

    # ── Decoder forward pass ──────────────────────────────────────────────
    decoder.train()   # enable z_dropout
    logits, h_states = decoder(token_ids, z_context, unit_idx)

    assert logits.shape   == (B, T, VOCAB_SIZE)
    assert h_states.shape == (B, T, d_model)
    print(f"logits:   {logits.shape} — OK")
    print(f"h_states: {h_states.shape} — OK")

    # ── Unit prediction head ──────────────────────────────────────────────
    unit_preds = unit_head(h_states)
    assert len(unit_preds) == 5
    assert unit_preds[0].shape == (B, T, N_UNIT_CLASSES)
    print(f"unit_preds: 5 x {unit_preds[0].shape} — OK")

    # Unit loss
    u_loss = unit_head.loss(h_states, unit_targets)
    assert u_loss.item() > 0
    assert torch.isfinite(u_loss)
    print(f"unit_loss: {u_loss.item():.4f} — OK")

    # ── Weight tying check ────────────────────────────────────────────────
    assert decoder.lm_head.weight is decoder.embedding.token_embed.weight, \
        "LM head and token embedding should share weights"
    print("Weight tying: OK")

    # ── No NaN in outputs ─────────────────────────────────────────────────
    assert torch.isfinite(logits).all()
    assert torch.isfinite(h_states).all()
    print("No NaN/Inf: OK")

    # ── Validity mask at inference ────────────────────────────────────────
    valid = get_valid_next_tokens(stack_depth=0, seq_len=0, max_len=MAX_SEQ_LEN)
    assert len(valid) > 0
    print(f"Valid tokens at depth 0: {len(valid)} — OK")

    print(f"\nAll decoder tests passed.")
    print(f"Decoder params: "
          f"{sum(p.numel() for p in decoder.parameters()):,}")
    print(f"Unit head params: "
          f"{sum(p.numel() for p in unit_head.parameters()):,}")