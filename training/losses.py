"""
Loss Functions for LLM-JEPA with SIGReg (LeJEPA).

SIGReg replaces EMA-based collapse prevention with a theoretically
grounded regularization term that enforces isotropic Gaussian
embeddings. Both encoders are trainable — gradients flow through
both context and target encoders.

context encoder:   trains on MSE + SIGReg(z_context) + L_lm + L_units
formula encoder:   trains on SIGReg(z_target) only
predictor:         cross-attention, z_context queries var_summaries → z_hat
MSE loss:          MSE(z_hat, z_target.detach())
SIGReg:            sigreg_fn(z_context) + sigreg_fn(z_target)

Components:
1. SIGRegLoss:      LeJEPA's Sketched Isotropic Gaussian Regularization
2. JEPALoss:        MSE between predicted and target representations
3. ValidityWeightedCE: Cross-entropy with RPN validity mask
4. UnitLoss:        Cross-entropy over 5 unit dimensions
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import lejepa
from data.tokenizer import (
    PAD_IDX, ARITY, VOCAB_SIZE, IDX2TOKEN, 
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    MAX_SEQ_LEN, BOS_IDX, EOS_IDX, TOKEN2IDX,
    get_valid_next_tokens
)


# ── SIGReg Loss (LeJEPA) ──────────────────────────────────────────────────────
class SIGRegLoss(nn.Module):
    """
    SIGReg regularization from LeJEPA (https://github.com/galilai-group/lejepa).
    
    Constrains concatenated embeddings to isotropic Gaussian distribution
    via variance and covariance terms. Prevents representational collapse
    without requiring EMA or gradient detachment.
    
    Args:
        num_slices:   number of random projections for multivariate test
        num_points:   evaluation points for univariate Epps-Pulley test
        lambda_reg:   regularization weight (default: 0.1)
    """
    def __init__(
        self,
        num_slices:   int = 1024,
        num_points:   int = 17,
        lambda_reg:   float = 0.1,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        univariate_test = lejepa.univariate.EppsPulley(n_points=num_points)
        self.sigreg_fn = lejepa.multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test,
               num_slices=num_slices,
            )
    

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, d_model] or [2*B, d_model] for concatenated z_context/z_target
        Returns:
            scalar SIGReg loss
        """
        return self.lambda_reg * self.sigreg_fn(embeddings)


# ── JEPA Prediction Loss ──────────────────────────────────────────────────────
class JEPALoss(nn.Module):
    """
    MSE loss between predicted and target representations.
    
    SIGReg Note: z_target is NOT detached — gradients flow through
    both encoders. Collapse is prevented by SIGRegLoss, not by
    detaching the target.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        z_pred:   torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_pred:   [B, d_model]  from predictor
            z_target: [B, d_model]  from target encoder (trainable for SIGReg)
        Returns:
            scalar MSE loss
        """
        return F.mse_loss(z_pred, z_target)

#── Unit Loss ─────────────────────────────────────────────────────────────────
class UnitLoss(nn.Module):
    """
    Cross-entropy loss over 5 unit dimensions.
    
    Each dimension is an independent classifier over 9 classes
    (unit exponents -4 to +4, offset by +4).
    
    Args:
        n_unit_dims:    number of unit dimensions (default: 5)
        n_unit_classes: classes per dimension (default: 9)
        ignore_index:   class index to ignore (default: -100)
    """
    def __init__(
        self,
        n_unit_dims:    int = 5,
        n_unit_classes: int = 9,
        ignore_index:   int = -100,
    ):
        super().__init__()
        self.n_unit_dims = n_unit_dims
        self.n_unit_classes = n_unit_classes
        self.ignore_index = ignore_index
    
    def forward(
        self,
        unit_preds:  List[torch.Tensor],
        unit_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            unit_preds:  list of 5 tensors, each [B, T, n_unit_classes]
            unit_targets: [B, T, 5] class indices in [0, 8]
            
        Returns:
            scalar mean loss over 5 dimensions
        """
        total_loss = torch.tensor(0.0, device=unit_targets.device)
        
        for dim_idx, pred in enumerate(unit_preds):
            target = unit_targets[:, :, dim_idx]
            loss = F.cross_entropy(
                pred.reshape(-1, self.n_unit_classes),
                target.reshape(-1),
                ignore_index=self.ignore_index,
            )
            total_loss = total_loss + loss
        
        return total_loss / self.n_unit_dims

class ValidityWeightedCE(nn.Module):
    """
    Cross-entropy loss weighted by RPN validity at each step.
    Vectorized for speed using torch.cumsum.
    """
    def __init__(
        self,
        invalid_weight: float = 2.0,
        ignore_index: int = PAD_IDX,
    ):
        super().__init__()
        self.invalid_weight = invalid_weight
        self.ignore_index = ignore_index
        self.eos_idx = EOS_IDX
        self.max_len = MAX_SEQ_LEN
        
        # Pre-compute arity map for vectorization
        arity_map = torch.zeros(VOCAB_SIZE, dtype=torch.long)
        for idx, tok in IDX2TOKEN.items():
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
                continue
            
            arity = ARITY.get(tok, 0)
            if arity == 2:
                arity_map[idx] = -1 
            elif arity == 0:
                arity_map[idx] = 1
        
        self.register_buffer('arity_map', arity_map)

        # Full arity map (2, 1, or 0) for validity check
        full_arity_map = torch.zeros(VOCAB_SIZE, dtype=torch.long)
        for idx, tok in IDX2TOKEN.items():
            full_arity_map[idx] = ARITY.get(tok, 0)
        self.register_buffer('full_arity_map', full_arity_map)

    def _get_batch_depths(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes stack depths for the entire batch at once.
        O(T) instead of O(B*T) loops.
        """
        # Map tokens to their stack delta (-1, 0, or 1)
        deltas = self.arity_map[token_ids]  # [B, T]

        # NO SHIFT: cumsum directly gives stack state AFTER processing token_ids[:t+1]
        # This is correct because:
        #   - logits[t] is computed after seeing input_ids[:t+1] (causal attention)
        #   - targets[t] is the next token to predict
        #   - Validity of targets[t] depends on stack state AFTER input_ids[:t+1]
        depths = torch.cumsum(deltas, dim=1)  # [B, T]
        return torch.clamp(depths, min=0)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:    [B, T, V] prediction scores
            targets:   [B, T] ground truth token indices
            token_ids: [B, T] input tokens (includes BOS). 
                       If None, targets is used as the context.
        """
        B, T, V = logits.shape
        input_ids = token_ids if token_ids is not None else targets
        device = logits.device
        
        # 1. Compute Base Loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
            reduction='none'
        ).reshape(B, T)  # [B, T]
        
        # 2. Get depths (Fast Vectorized Way)
        depths = self._get_batch_depths(input_ids)  # [B, T]
        
        # 3. Apply Validity Weights (Fully Vectorized)
        arities = self.full_arity_map[targets]  # [B, T]
        is_eos = (targets == self.eos_idx)      # [B, T] mask
        is_pad = (targets == self.ignore_index) # [B, T] mask
        
        # Sequence position for max_len check
        t_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T) # [B, T]
        
        # Parallel Validity Check:
        # Instead of 'if targets[b,t] == EOS...', we evaluate all conditions
        # across the entire [B, T] grid in one GPU kernel call.
        
        # - EOS is valid ONLY if stack depth is exactly 1 (complete expression)
        valid_eos = is_eos & (depths == 1)
        
        # - Non-EOS tokens are valid IF:
        #   a) We haven't hit the hard max length limit
        #   b) The current stack depth can satisfy the operator's arity
        valid_non_eos = (~is_eos) & (t_indices < self.max_len - 1) & (
            (arities == 0) |                   # Leaf: always valid if not full
            ((arities == 1) & (depths >= 1)) | # Unary: needs 1 operand
            ((arities == 2) & (depths >= 2))   # Binary: needs 2 operands
        )
        
        is_valid = valid_eos | valid_non_eos  # [B, T] boolean mask
        
        # 4. Final Loss Calculation
        weights = torch.ones_like(ce_loss)
        # Penalize tokens that violate RPN grammar
        weights[~is_valid & ~is_pad] = self.invalid_weight
        
        weighted_loss = ce_loss * weights
        mask = (targets != self.ignore_index).float()
        return (weighted_loss * mask).sum() / mask.sum().clamp(min=1.0)


# ── Combined Loss Wrapper ─────────────────────────────────────────────────────
class LLMJEPALoss(nn.Module):
    """
    Wrapper that combines all loss components for SIGReg-based LLM-JEPA training.
    
    Usage:
        loss_fn = LLMJEPALoss()
        losses = loss_fn(
            z_pred=z_pred,
            z_target=z_target,
            z_context=z_context,
            logits=logits,
            token_targets=token_targets,
            unit_preds=unit_preds,
            unit_targets=unit_targets,
            token_ids=token_ids,  # for validity weighting
        )
    
    All loss terms weighted equally (1.0) by default.
    """
    def __init__(
        self,
        # SIGReg params
        sigreg_num_slices:  int = 1024,
        sigreg_num_points:  int = 17,
        sigreg_lambda:      float = 1.0,
        # Validity-weighted CE params
        invalid_weight:     float = 2.0,
        # Unit loss params
        n_unit_dims:        int = 5,
        n_unit_classes:     int = 9,
        # Loss weights (all default to 1.0)
        lambda_jepa:        float = 1.0,
        lambda_sigreg:      float = 1.0,
        lambda_lm:          float = 1.0,
        lambda_units:       float = 1.0,
    ):
        super().__init__()
        self.jepa_loss = JEPALoss()
        self.sigreg_loss = SIGRegLoss(
            num_slices=sigreg_num_slices,
            num_points=sigreg_num_points,
            lambda_reg=sigreg_lambda,
        )
        self.lm_loss = ValidityWeightedCE(
            invalid_weight=invalid_weight,
        )
        self.unit_loss = UnitLoss(
            n_unit_dims=n_unit_dims,
            n_unit_classes=n_unit_classes,
        )
        
        # Loss weights
        self.lambda_jepa = lambda_jepa
        self.lambda_sigreg = lambda_sigreg
        self.lambda_lm = lambda_lm
        self.lambda_units = lambda_units
    
    def forward(
        self,
        z_pred:        torch.Tensor,
        z_target:      torch.Tensor,
        z_context:     torch.Tensor,
        logits:        torch.Tensor,
        token_targets: torch.Tensor,
        unit_preds:    List[torch.Tensor],
        unit_targets:  torch.Tensor,
        token_ids:     Optional[torch.Tensor] = None,  # for validity weighting
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Returns:
            dict with 'total' loss and individual components for logging
        """
        # 1. JEPA prediction loss (target detached)
        L_jepa = self.jepa_loss(z_pred, z_target)
        
        # 2. SIGReg loss applied independently to context and target
        L_sigreg = self.sigreg_loss(z_context) + self.sigreg_loss(z_target)
        
        # 3. LM loss with validity weighting (uses get_valid_next_tokens)
        L_lm = self.lm_loss(logits, token_targets, token_ids)
        
        # 4. Unit loss
        L_units = self.unit_loss(unit_preds, unit_targets)
        
        # Total loss
        L_total = (
            self.lambda_jepa * L_jepa +
            self.lambda_sigreg * L_sigreg +
            self.lambda_lm * L_lm +
            self.lambda_units * L_units
        )
        
        return {
            'total': L_total,
            'jepa': L_jepa.detach(),
            'sigreg': L_sigreg.detach(),
            'lm': L_lm.detach(),
            'units': L_units.detach(),
        }


# ── Tests ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    from data.tokenizer import VOCAB_SIZE, PAD_IDX, BOS_IDX, EOS_IDX, TOKEN2IDX
    
    print("Testing loss functions (SIGReg version)...")
    
    B, T, d_model = 4, 20, 64
    n_vars = 9
    vocab_size = VOCAB_SIZE
    
    # ── Test 1: SIGReg Loss ──────────────────────────────────────────────────
    sigreg = SIGRegLoss(lambda_reg=1.0)
    
    # Random tensors should have reasonable variance/covariance
    z_combined = torch.randn(2 * B, d_model)
    L_sig = sigreg(z_combined)
    
    assert torch.isfinite(L_sig) and L_sig > 0
    print(f"✓ SIGReg loss: {L_sig.item():.4f}")
    
    # Collapse test: constant tensor should have high loss
    z_collapsed = torch.zeros(2 * B, d_model)
    L_sig_collapsed = sigreg(z_collapsed)
    assert L_sig_collapsed > L_sig, "SIGReg should penalize collapse"
    print(f"✓ SIGReg detects collapse: {L_sig_collapsed.item():.4f} > {L_sig.item():.4f}")
    
    # ── Test 2: JEPA Loss ───────────────────────────────────────────────────
    jepa = JEPALoss()
    z_pred = torch.randn(B, d_model)
    z_target = torch.randn(B, d_model)
    L_j = jepa(z_pred, z_target)
    
    assert torch.isfinite(L_j) and L_j >= 0
    print(f"✓ JEPA loss: {L_j.item():.4f}")
    
    # ── Test 3: Validity-Weighted CE ────────────────────────────────────────
    lm_loss = ValidityWeightedCE(invalid_weight=2.0)
    
    # Create a valid RPN sequence for testing
    token_ids = torch.zeros(B, T, dtype=torch.long)
    token_ids[:, 0] = BOS_IDX
    token_ids[:, 1] = TOKEN2IDX.get('x1', 4)
    token_ids[:, 2] = TOKEN2IDX.get('x2', 5)
    token_ids[:, 3] = TOKEN2IDX.get('+', 20)
    token_ids[:, 4] = EOS_IDX
    
    logits = torch.randn(B, T, vocab_size)
    targets = token_ids.clone()
    
    L_lm_plain = lm_loss(logits, targets, token_ids=None)
    assert torch.isfinite(L_lm_plain)
    print(f"✓ LM loss (plain): {L_lm_plain.item():.4f}")
    
    L_lm_weighted = lm_loss(logits, targets, token_ids=token_ids)
    assert torch.isfinite(L_lm_weighted)
    print(f"✓ LM loss (weighted): {L_lm_weighted.item():.4f}")
    
    # ── Test 4: Unit Loss ───────────────────────────────────────────────────
    unit_loss = UnitLoss(n_unit_dims=5, n_unit_classes=9)
    
    unit_preds = [torch.randn(B, T, 9) for _ in range(5)]
    unit_targets = torch.randint(0, 9, (B, T, 5))
    
    L_u = unit_loss(unit_preds, unit_targets)
    assert torch.isfinite(L_u) and L_u > 0
    print(f"✓ Unit loss: {L_u.item():.4f}")
    
    # ── Test 5: Combined Loss ───────────────────────────────────────────────
    combined = LLMJEPALoss()
    
    losses = combined(
        z_pred=torch.randn(B, d_model),
        z_target=torch.randn(B, d_model),
        z_context=torch.randn(B, d_model),
        logits=torch.randn(B, T, vocab_size),
        token_targets=targets,
        unit_preds=unit_preds,
        unit_targets=unit_targets,
        token_ids=token_ids,
    )
    
    assert 'total' in losses
    assert torch.isfinite(losses['total'])
    assert all(k in losses for k in ['jepa', 'sigreg', 'lm', 'units'])
    print(f"✓ Combined loss: {losses['total'].item():.4f}")
    print(f"  Components: jepa={losses['jepa'].item():.3f}, "
          f"sigreg={losses['sigreg'].item():.3f}, "
          f"lm={losses['lm'].item():.3f}, "
          f"units={losses['units'].item():.3f}")
    
    print("\n✅ All loss function tests passed (SIGReg version).")