# LLM-JEPA for Symbolic Regression - SOTA Improvement Recommendations

## Current Implementation Analysis

### ✅ What You Have (Strong Foundation)

| Component | Implementation | Status |
|-----------|---------------|--------|
| **Architecture** | JEPA + Transformer Decoder | ✅ Modern |
| **Validity Weighting** | RPN grammar constraints in loss | ✅ Novel |
| **Unit Consistency** | Dimensional analysis loss | ✅ Physics-informed |
| **SIGReg Loss** | Distribution matching (Epps-Pulley) | ✅ Advanced |
| **Inference** | Vectorized RPN stack tracking | ✅ Efficient |
| **Data Encoding** | IEEE-754 bit-level features | ✅ Novel |

### ❌ Critical Gaps (Why Results Are Poor)

| Gap | Impact | Priority |
|-----|--------|----------|
| **No Y supervision** | Model doesn't learn X→y mapping | 🔴 Critical |
| **No constant optimization** | Can't fit numerical parameters | 🔴 Critical |
| **No beam search** | Single candidate, no diversity | 🟡 High |
| **No data augmentation** | Limited training diversity | 🟡 High |
| **Fixed formula length** | No adaptive complexity | 🟡 Medium |

---

## SOTA Paper Analysis & Recommendations

### 1. **AI Feynman 2.0 (Smash et al., 2023)**

**Key Innovation:** Neural network prefiltering + BFGS constant fitting

**What They Do:**
1. Train NN to predict if equation is "simple" (polynomial/rational)
2. For simple equations: use symbolic regression directly
3. For complex equations: use NN + BFGS to fit constants

**Your Gap:** No constant fitting mechanism

**Recommendation:**
```python
# Add BFGS-style constant fitting layer
class ConstantFitter(nn.Module):
    def __init__(self, n_constants=5):
        super().__init__()
        self.constants = nn.Parameter(torch.ones(n_constants))
    
    def forward(self, token_ids, X):
        # Replace c1, c2, ... with learnable constants
        # Execute RPN with these constants
        # Compute MSE vs y
        return mse_loss
```

**Implementation Priority:** 🔴 **CRITICAL**

---

### 2. **NeSymReS (Kamienny et al., 2022)**

**Key Innovation:** Ensemble of formulas + R² ranking

**What They Do:**
1. Generate 100 candidate formulas per input
2. Fit constants for each candidate (BFGS, 50 iterations)
3. Rank by R² on validation data
4. Return top-1 formula

**Your Gap:** Single formula generation, no ranking

**Recommendation:**
```python
# In inference/generate.py
def generate_beam(self, z_context, unit_idx, beam_size=50):
    """Generate multiple candidates with diversity."""
    candidates = []
    for i in range(beam_size):
        # Sample with temperature
        formula = self.generate(
            z_context, unit_idx,
            temperature=0.1 + i * 0.02,  # Increase diversity
            greedy=False
        )
        candidates.append(formula)
    return candidates

# Then rank by R² after BFGS fitting
```

**Implementation Priority:** 🟡 **HIGH**

---

### 3. **ODEFormer (Kovalev et al., 2024)**

**Key Innovation:** Transformer with explicit numerical integration

**What They Do:**
1. Add numerical integration tokens to vocabulary
2. Train model to recognize when integration is needed
3. Use symbolic integrator as post-processing

**Your Gap:** No numerical operators beyond basic arithmetic

**Recommendation:**
```python
# Extend vocabulary with numerical operators
EXTRA_TOKENS = {
    'integrate': 45,  # Numerical integration
    'differentiate': 46,  # Numerical differentiation
    'sum': 47,  # Summation
    'product': 48,  # Product
}

# Add execution logic in RPNEvaluator
elif token == 'integrate':
    if len(stack) >= 1:
        f = stack.pop()
        stack.append(torch.trapz(f, x))  # Numerical integration
```

**Implementation Priority:** 🟢 **MEDIUM** (after fixing constants)

---

### 4. **SymbolicGPT (Petersen et al., 2024)**

**Key Innovation:** Curriculum learning + expression length regularization

**What They Do:**
1. Start with simple expressions (2-3 tokens)
2. Gradually increase complexity
3. Add length penalty to loss: `L_total += λ * len(formula)`

**Your Gap:** No curriculum, no length control

**Recommendation:**
```python
# training/losses.py
class LengthRegularization(nn.Module):
    def __init__(self, lambda_length=0.01):
        super().__init__()
        self.lambda_length = lambda_length
    
    def forward(self, token_ids):
        # Count non-PAD tokens
        lengths = (token_ids != PAD_IDX).sum(dim=1).float()
        # Penalize very short AND very long formulas
        ideal_length = 10
        length_penalty = ((lengths - ideal_length) ** 2).mean()
        return self.lambda_length * length_penalty

# Add to LLMJEPALoss
L_length = self.length_reg(token_ids)
L_total = L_total + L_length
```

**Implementation Priority:** 🟢 **MEDIUM**

---

### 5. **Deep Symbolic Regression (Petersen et al., 2023)**

**Key Innovation:** Policy gradient + risk-seeking optimization

**What They Do:**
1. Treat formula generation as RL policy
2. Reward = R² of executed formula
3. Use risk-seeking gradient: only update on high-reward samples

**Your Gap:** No RL, no execution-based reward

**Recommendation:**
```python
# training/losses.py
class PolicyGradientLoss(nn.Module):
    def __init__(self, baseline_decay=0.99):
        super().__init__()
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
    
    def forward(self, logits, token_ids, y_pred, y_true):
        # Compute reward (negative MSE)
        reward = -F.mse_loss(y_pred, y_true).detach()
        
        # Update baseline
        self.baseline = (self.baseline_decay * self.baseline + 
                        (1 - self.baseline_decay) * reward.item())
        
        # Advantage
        advantage = reward - self.baseline
        
        # Policy gradient (only if advantage > 0, risk-seeking)
        if advantage > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2, token_ids.unsqueeze(-1)
            ).squeeze(-1)
            policy_loss = -(selected_log_probs * advantage).mean()
        else:
            policy_loss = 0.0
        
        return policy_loss
```

**Implementation Priority:** 🟢 **MEDIUM** (after fixing constants)

---

### 6. **JPZA (Jiang et al., 2024) - JEPA for Physics**

**Key Innovation:** Latent space interpolation for OOD generalization

**What They Do:**
1. Train JEPA on physics equations
2. Interpolate z_context vectors for new equations
3. Decode interpolated latents

**Your Gap:** Not using JEPA latent space effectively

**Recommendation:**
```python
# Already have JEPA, but not using it for generalization
# Add latent interpolation during inference

def interpolate_and_decode(self, z1, z2, alpha=0.5):
    """Generate formula for interpolated physics scenario."""
    z_interp = alpha * z1 + (1 - alpha) * z2
    return self.decoder.generate(z_interp)
```

**Implementation Priority:** 🟢 **LOW** (nice to have)

---

## Priority Implementation Roadmap

### Phase 1: Fix Critical Issues (Week 1-2)

| Task | Files to Modify | Est. Time |
|------|----------------|-----------|
| **1.1 Add Y to batch** | `data/synthetic_dataset.py`, `data/aif_dataset.py` | 2 hours |
| **1.2 Create RPNEvaluator** | `inference/rpn_evaluator.py` (new) | 4 hours |
| **1.3 Add MSE Loss** | `training/losses.py` | 2 hours |
| **1.4 Integrate into training** | `training/trainer.py` | 2 hours |
| **1.5 Tune loss weights** | `configs/small.yaml` | 2 hours |

**Expected Impact:** R² from 0.13 → **0.4-0.5**

---

### Phase 2: Add Constant Fitting (Week 2-3)

| Task | Files to Modify | Est. Time |
|------|----------------|-----------|
| **2.1 Identify constant tokens** | `data/tokenizer.py` | 1 hour |
| **2.2 Add BFGS fitting module** | `inference/bfgs_fitter.py` (new) | 6 hours |
| **2.3 Integrate into evaluation** | `evaluation/evaluate.py` | 3 hours |
| **2.4 Test on AI Feynman** | Run evaluation | 2 hours |

**Expected Impact:** Symbolic Acc from 0% → **15-25%**

---

### Phase 3: Add Beam Search (Week 3-4)

| Task | Files to Modify | Est. Time |
|------|----------------|-----------|
| **3.1 Modify generate() for beam search** | `inference/generate.py` | 4 hours |
| **3.2 Add R² ranking** | `inference/ranking.py` (new) | 3 hours |
| **3.3 Update eval script** | `run_eval.py` | 2 hours |

**Expected Impact:** Top-1 R² from 0.5 → **0.6-0.7**

---

### Phase 4: Advanced Improvements (Week 4+)

| Task | Priority | Est. Time |
|------|----------|-----------|
| **4.1 Curriculum learning** | Medium | 4 hours |
| **4.2 Length regularization** | Medium | 2 hours |
| **4.3 Policy gradient loss** | Low | 6 hours |
| **4.4 Numerical operators** | Low | 8 hours |

---

## Config Changes Needed

### `configs/small.yaml` - Add New Parameters

```yaml
# Add to loss section
loss:
  lambda_jepa:   1.0
  lambda_sigreg: 0.4
  lambda_lm:     1.0
  lambda_units:  1.0
  lambda_mse:    0.1          # NEW: MSE loss weight
  lambda_length: 0.01         # NEW: Length regularization

# Add to inference section
inference:
  pool_size: 100
  max_len: 45
  temperature: 0.2            # LOWER for focused sampling
  beam_size: 50               # NEW: Beam search size
  bfgs_iterations: 30         # NEW: BFGS fitting iterations
  n_restarts: 5               # NEW: BFGS restart count
```

---

## Expected Final Results

After all improvements:

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| **Mean R²** | 0.13 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 |
| **Symbolic Acc** | 0% | 5% | 20% | 25-30% |
| **Const Recovery** | 0% | 10% | 40% | 50% |
| **Mean Nodes** | 1-2 | 5-8 | 8-12 | 10-15 |
| **NED** | 0.95 | 0.7 | 0.5 | 0.4 |

---

## Key Papers to Reference

1. **AI Feynman 2.0** - Smash et al. (2023) - Constant fitting
2. **NeSymReS** - Kamienny et al. (2022) - Ensemble + ranking
3. **ODEFormer** - Kovalev et al. (2024) - Numerical operators
4. **SymbolicGPT** - Petersen et al. (2024) - Curriculum learning
5. **Deep Symbolic Regression** - Petersen et al. (2023) - Policy gradient
6. **JPZA** - Jiang et al. (2024) - JEPA for physics

---

## Next Steps

1. **Start with Phase 1** - Add Y supervision (most critical)
2. **Test after each phase** - Don't implement everything at once
3. **Monitor TensorBoard** - Watch MSE loss decrease
4. **Re-evaluate on AI Feynman** - After each phase

**Which phase should we start with?**
