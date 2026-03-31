# LLM-JEPA for Symbolic Regression - Codebase Documentation

**Last Updated:** 2026-03-30  
**Status:** Production-Ready with Commutative Augmentation

---

## Overview

This document outlines the end-to-end architecture, data flow, and training paradigms of the LLM-JEPA Symbolic Regression system. The model learns to generate mathematical formulas from tabular data using a Joint-Embedding Predictive Architecture (JEPA) with physics-informed constraints.

### Key Features

- **Physics-Informed Generation:** All synthetic equations respect dimensional homogeneity
- **Commutative Data Augmentation:** Automatic generation of algebraically equivalent formulas (a+b ↔ b+a)
- **Validity-Weighted Loss:** RPN grammar constraints enforced during training
- **Unit Consistency:** Dimensional analysis ensures physically meaningful predictions
- **JEPA + SIGReg:** No EMA needed - both encoders trainable with isotropic Gaussian regularization

---

## 1. Data Generation & Handling

Data generation is the foundation of the system. Equations must be **mathematically valid**, **dimensionally consistent**, and **structurally diverse**.

### `data/synthetic_dataset.py` (The Generator)

#### Physics Variable Pooling

```python
DOMAIN_POOLS = {
    'mechanics': [
        ('m1', 'mass1', (0.5, 5.0)),      # kg
        ('x1', 'pos_x1', (1.0, 5.0)),     # m
        ('v', 'vel', (0.1, 3.0)),         # m/s
        ('t', 'time', (0.1, 5.0)),        # s
    ],
    'electromagnetism': [...],
    'thermodynamics': [...],
    'dimensionless': [...]
}
```

Variables are sampled from physics domains with **realistic bounds**:
- Temperatures: 200-500 Kelvin
- Masses: 0.5-5.0 kg
- Velocities: 0.1-3.0 m/s

#### Unit Propagation & Dimensional Homogeneity

```python
def propagate_units(operator, child_units):
    if operator in ('+', '-'):
        # Addition requires identical units
        if child_units[0] != child_units[1]:
            return None  # Reject: can't add mass to time
        return child_units[0]
    
    elif operator == '*':
        # Multiplication adds unit exponents
        return [child_units[0][i] + child_units[1][i] for i in range(5)]
    
    elif operator == '/':
        # Division subtracts unit exponents
        return [child_units[0][i] - child_units[1][i] for i in range(5)]
```

As `PhysicsTreeBuilder` constructs expressions recursively, **every operator validates** the intrinsic units of its child nodes. Invalid trees are rejected immediately.

#### Tabular Data Creation

```python
# Convert symbolic tree to numerical function
f_lambda = sympy.lambdify(symbols, expr, 'numpy')

# Sample input data from variable ranges
X = np.column_stack([
    np.random.uniform(lo, hi, n_data_points)
    for lo, hi in ranges
])

# Evaluate to get output
y = f_lambda(*[X[:, i] for i in range(n_vars)])
```

Each equation is evaluated across `n_data_points` (default: 500-2000) to create the training dataset.

#### Commutative & Structural Augmentations

**NEW (2026-03-30):** Three augmentation strategies increase diversity without recomputing tables:

```python
def augment_commutative(rpn_tokens):
    """Generate algebraically equivalent variants."""
    augmented = [rpn_tokens]
    for i, tok in enumerate(rpn_tokens):
        if tok in ('+', '*'):
            swapped = swap_rpn_operands(rpn_tokens, i)
            if swapped != rpn_tokens:
                augmented.append(swapped)
    return augmented  # Returns [original, swapped] for each + or *
```

**Augmentation Types:**
1. **Affine Transforms:** Linear scale stretching (x → a*x + b)
2. **Negation Augmentation:** Random sub-expression negation (30% probability)
3. **Commutative Swapping:** Operand swapping for `+` and `*` operators

**Impact:** Each generated equation produces **1-2 training samples** automatically.

#### IEEE-754 Bit Encoding

```python
def to_ieee754_16bit(X):
    """Convert floats to 16-bit IEEE-754 representation."""
    # Each float16 → 16 binary features
    # Stored as uint8 to save RAM
    return X.view(np.uint8).reshape(-1, n_vars, 2)
```

Instead of passing raw floats to the Transformer (which LLMs struggle to interpret), inputs are **explicitly decomposed** into:
- Sign bits
- Exponent bits  
- Fraction/mantissa bits

This enables the model to learn numerical patterns at the **bit level**.

---

### `data/aif_dataset.py` (The Evaluation Dataloader)

Both Synthetic and AI Feynman datasets use identical interfaces:

```python
class LazySyntheticDataset(Dataset):
    def __getitem__(self, idx):
        # Load equation from cached .pt file
        eq = self.equations[idx]
        
        # Subsample n_rows (e.g., 200 points) per equation
        indices = np.random.choice(len(eq.y_noisy), self.n_rows, replace=False)
        
        # Return batch with padding
        return {
            'X_bits': eq.X_bits[indices],
            'y_noisy': eq.y_noisy[indices],
            'token_ids': eq.token_ids,
            'var_mask': self.var_mask(eq.n_vars),
        }
```

**Key Design:**
- **Lazy Loading:** 100k+ row equations are **never fully loaded** into memory
- **On-the-fly Subsampling:** Each `__getitem__` call samples fresh `n_rows` data points
- **Variable Padding:** Equations with 3 vars are padded to `max_n_vars=9` with zeros
- **Mask Tracking:** `var_mask` tells the model which variables are real vs padding

> [!TIP]
> **Contiguous Chunk Sampling:** The `LazySyntheticDataset` uses `ContiguousChunkSampler` during training. This forces the PyTorch DataLoader to read **sequentially** across `.pt` chunk files rather than jumping randomly, solving Google Drive / cloud I/O throttling issues.

---

## 2. Architecture & Forward Pass (`training/trainer.py`)

Training uses PyTorch Lightning via `LLMJEPAModule`, based on JEPA but with **SIGReg** replacing EMA for collapse prevention.

### Forward Pass Flow

```python
def forward(self, batch):
    # ── 1. Prepare Inputs ──────────────────────────────────────────────
    X_bits      = batch['X_bits']       # [B, N, n_vars, 16] IEEE-754 bits
    unit_idx    = batch['unit_idx']     # [B, n_vars, 5] unit vectors
    var_mask    = batch['var_mask']     # [B, n_vars] variable mask
    token_ids   = batch['token_ids']    # [B, T] RPN formula tokens
    unit_targets = batch['unit_targets']# [B, T, 5] unit labels per token
    
    # ── 2. Context Encoding ────────────────────────────────────────────
    data_emb = self.data_embedder(X_bits)      # [B, N, d_model]
    unit_emb = self.unit_embedder(unit_idx)    # [B, n_vars, d_model]
    z_context, var_summaries = self.mix_encoder(
        data_emb, unit_emb, var_mask
    )  # z_context: [B, d_model]
    
    # ── 3. Target Encoding ─────────────────────────────────────────────
    z_target = self.target_encoder(token_ids)  # [B, d_model]
    
    # ── 4. JEPA Prediction ─────────────────────────────────────────────
    z_pred = self.predictor(z_context, var_summaries)  # [B, d_model]
    
    # ── 5. Decoding ────────────────────────────────────────────────────
    logits = self.decoder(token_ids[:, :-1], z_context, unit_idx)
    
    # ── 6. Unit Prediction ─────────────────────────────────────────────
    unit_preds = self.unit_head(decoder_hidden_states)
    
    return {
        'z_pred': z_pred,
        'z_target': z_target,
        'z_context': z_context,
        'logits': logits,
        'unit_preds': unit_preds,
    }
```

### Component Roles

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **DataEmbedder** | IEEE-754 bits | [B, N, d_model] | Learn bit-level patterns |
| **UnitEmbedder** | Unit vectors | [B, n_vars, d_model] | Encode physics dimensions |
| **MixEncoder** | Data + Unit embeddings | z_context + var_summaries | Cross-attention fusion |
| **TargetEncoder** | RPN tokens | z_target | Formula representation |
| **JEPAPredictor** | z_context + var_summaries | z_pred | Latent space prediction |
| **RPNDecoder** | z_context + tokens | logits | Autoregressive generation |
| **UnitHead** | Decoder hidden states | unit classes | Dimensional prediction |

---

## 3. Loss Assessment (`training/losses.py`)

Four independent losses are aggregated in `LLMJEPALoss`:

### A. JEPALoss (Semantic Proximity)

```python
class JEPALoss(nn.Module):
    def forward(self, z_pred, z_target):
        return F.mse_loss(z_pred, z_target)
```

**Key Design:** Unlike BYOL, SIMCLR, or classic JEPA:
- ❌ **NO gradient detachment** on `z_target`
- ✅ Gradients flow through **both encoders**
- ✅ Both encoders are **jointly trainable**

This enables direct semantic alignment without EMA complexity.

---

### B. SIGRegLoss (Collapse Regularization)

```python
class SIGRegLoss(nn.Module):
    """Skewed Isotropic Gaussian Regularization."""
    def forward(self, embeddings):
        # Force embeddings to match N(0, I) distribution
        # via variance + covariance terms
        return lambda_reg * EppsPulley_test(embeddings)
```

**Problem Solved:** Without Stop-Gradients or EMA, Siamese networks collapse (predicting constant 0 gives 0 MSE loss).

**Solution:** SIGReg forces `z_context` and `z_target` to:
1. Maintain **unit variance** across dimensions
2. Maintain **zero covariance** between dimensions
3. Spread continuously across latent space

**Applied Independently:** `L_sigreg = SIGReg(z_context) + SIGReg(z_target)`

---

### C. ValidityWeightedCE (Structural Token Language Modeling)

```python
class ValidityWeightedCE(nn.Module):
    def __init__(self, ignore_index=-100, invalid_weight=2.0):
        super().__init__()
        self.ignore_index = ignore_index  # PAD tokens ignored
        self.invalid_weight = invalid_weight
    
    def forward(self, logits, targets, token_ids):
        # 1. Base Cross Entropy (PAD tokens masked to -100)
        ce_loss = F.cross_entropy(logits, targets, ignore_index=-100)
        
        # 2. Compute RPN Stack Depths via cumsum
        deltas = arity_map[token_ids]  # +1 for leaves, -1 for binary
        depths = torch.cumsum(deltas, dim=1)
        
        # 3. Build Validity Mask
        # Binary ops only valid if depth >= 2
        # Unary ops only valid if depth >= 1
        # EOS only valid if depth == 1 AND seq_len >= 5
        is_valid_logit = build_validity_mask(depths, arities)
        
        # 4. Penalize Invalid Probability Mass
        probs = F.softmax(logits, dim=-1)
        invalid_prob_mass = (probs * ~is_valid_logit).sum(dim=-1)
        
        # 5. Final Loss
        return ce_loss + (invalid_weight * invalid_prob_mass)
```

**Innovations:**

1. **PAD Masking:** `decoder_target[decoder_target == PAD_IDX] = -100`
   - Prevents model from learning to output padding
   - PyTorch's `cross_entropy` natively ignores `-100` indices

2. **Dynamic Grammar Penalty:**
   - Maps RPN tokens to stack deltas: `+1` (leaves), `0` (unary), `-1` (binary)
   - Uses `cumsum` to track stack depth at each position
   - Builds validity mask based on depth constraints

3. **Complexity Enforcement:**
   ```python
   # EOS only valid if:
   valid_eos = (depth == 1) & (seq_len >= 5)
   ```
   - Prevents "lazy" 1-token equations
   - Forces minimum formula complexity

4. **Soft Continuous Penalty:**
   - Doesn't hard-mask invalid tokens
   - Instead, penalizes **probability mass** assigned to invalid tokens
   - Gradient flows through invalid_prob_mass

**Impact:** Model learns RPN grammar **implicitly** through loss, not just memorization.

---

### D. UnitLoss (Dimensional Mapping)

```python
class UnitLoss(nn.Module):
    def forward(self, unit_preds, unit_targets):
        # 5 independent classifiers: (kg, m, s, A, K)
        losses = [
            F.cross_entropy(pred[:, i], target[:, i], ignore_index=-100)
            for i in range(5)
        ]
        return sum(losses) / 5
```

**Purpose:** Every token prediction includes **5 unit classifiers** predicting:
- Kilograms (mass)
- Meters (length)
- Seconds (time)
- Amperes (charge)
- Dimensionless

**Effect:** Forces the Transformer to use **physically grounded** tokens, not abstract placeholders.

---

### Total Loss Composition

```python
def forward(self, z_pred, z_target, z_context, logits, 
            token_targets, unit_preds, unit_targets, token_ids):
    
    L_jepa   = self.jepa_loss(z_pred, z_target)
    L_sigreg = self.sigreg_loss(z_context) + self.sigreg_loss(z_target)
    L_lm     = self.lm_loss(logits, token_targets, token_ids)
    L_units  = self.unit_loss(unit_preds, unit_targets)
    
    L_total = (
        self.lambda_jepa * L_jepa +
        self.lambda_sigreg * L_sigreg +
        self.lambda_lm * L_lm +
        self.lambda_units * L_units
    )
    
    return {
        'total': L_total,
        'jepa': L_jepa,
        'sigreg': L_sigreg,
        'lm': L_lm,
        'units': L_units,
    }
```

**Default Weights:** All `lambda_* = 1.0` (equal weighting)

---

## 4. Modularity & Scaling

The architecture is designed for **easy scaling** via configuration changes.

### Configuration Example (`configs/small.yaml`)

```yaml
model:
  d_model:      128          # Embedding dimension
  n_heads:      4            # Must divide d_model (128/4=32)
  n_enc_layers: 4            # Target encoder depth
  n_dec_layers: 4            # RPN decoder depth
  n_isab:       2            # ISA blocks in MixEncoder
  n_col_attn:   2            # Column attention layers
  
  predictor:
    pred_n_heads:          2
    pred_bottleneck_ratio: 0.50  # 128 * 0.5 = 64
    pred_dropout:          0.20

training:
  lr:            4e-4
  max_epochs:    3
  batch_size:    128
  val_check_interval: 200

data:
  n_synthetic:   250000     # 250k equations
  n_data_points: 500        # Per equation
  max_n_vars:    9          # Padding size
```

### Dimensional Constraints

**Must Satisfy:**
```python
assert d_model % n_heads == 0
assert (d_model * bottleneck_ratio) % pred_n_heads == 0
```

**Example (small.yaml):**
- `d_model=128`, `n_heads=4` → `head_dim=32` ✅
- `bottleneck_dim = 128 * 0.5 = 64`, `pred_n_heads=2` → `64/2=32` ✅

### Scaling Guidelines

| Dataset Size | Recommended Config | Parameters | Training Time (T4) |
|--------------|-------------------|------------|-------------------|
| 25k-50k | `d_model=80, 4 layers` | ~1M | 4-8 hours |
| 100k-250k | `d_model=128, 4 layers` | ~2M | 8-15 hours |
| 500k-1M | `d_model=256, 6 layers` | ~3.4M | 20-40 hours |

**Rule of Thumb:** 5-10 parameters per training equation for optimal generalization.

---

## 5. Inference Pipeline

### Formula Generation (`inference/generate.py`)

```python
@torch.no_grad()
def generate(self, z_context, unit_idx, max_len=45, 
             temperature=0.1, greedy=True):
    
    # Initialize with BOS token
    generated = torch.full((B, 1), BOS_IDX)
    stack_depths = torch.zeros(B)
    
    for step in range(max_len - 1):
        # Decoder forward
        logits = self.decoder(generated, z_context, unit_idx)
        
        # Apply RPN validity mask
        mask = build_validity_mask(stack_depths, step, max_len)
        logits = logits + mask * (-1e9)  # Penalize invalid
        
        # Sample or greedy select
        if greedy:
            next_token = logits.argmax(dim=-1)
        else:
            probs = softmax(logits / temperature)
            next_token = multinomial(probs)
        
        # Update stack depths
        deltas = arity_map[next_token]
        stack_depths += deltas
        
        # Check for EOS
        if (next_token == EOS_IDX).all():
            break
        
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

**Key Features:**
- **Vectorized RPN Tracking:** Stack depths updated for entire batch simultaneously
- **Validity Masking:** Invalid tokens penalized with `-1e9` before softmax
- **Temperature Sampling:** Higher temperature → more diverse candidates
- **Early Stopping:** Stops when all sequences predict EOS

### Beam Search & Ranking

```python
# Generate multiple candidates
candidates = []
for temp in [0.1, 0.2, 0.3, 0.5, 0.8]:
    formula = model.generate(z_context, temperature=temp)
    candidates.append(formula)

# Rank by R² (after BFGS constant fitting)
ranked = rank_by_r2(candidates, X, y)
return ranked[0]  # Best formula
```

**Default:** `pool_size=100` candidates generated per input.

---

## 6. Training Workflow

### Step 1: Generate Synthetic Data

```bash
python -m data.generate_data --config configs/small.yaml
```

**Output:** `cache/synthetic_small/part_*.pt` (100 equations per file)

### Step 2: Preprocess AI Feynman

```bash
python -m data.preprocess_aif --config configs/small.yaml
```

**Output:** `cache/aif_preprocessed.pt` (evaluation dataset)

### Step 3: Train Model

```bash
python -m training.train --config configs/small.yaml
```

**Monitors:**
- `train/total` - Combined training loss
- `val/total` - Validation loss (every 200 steps)
- `train/lm`, `train/jepa`, `train/sigreg`, `train/units` - Component losses

### Step 4: Evaluate

```bash
python run_eval.py --ckpt checkpoints/last.ckpt --n_candidates 100
```

**Metrics:**
- Exact recovery rate
- Mean R² (pre/post BFGS)
- Symbolic Accuracy (NED < 0.1)
- Constant Recovery
- Mean formula nodes

---

## 7. Debugging & Monitoring

### TensorBoard Logs

```bash
tensorboard --logdir tb_logs/
```

**Key Metrics to Watch:**

| Metric | Healthy Range | Warning Signs |
|--------|---------------|---------------|
| `train/total` | Decreasing | Plateau or increase |
| `train/lm` | 2.0-4.0 | > 6.0 (grammar issues) |
| `train/jepa` | 0.1-0.5 | > 1.0 (latent mismatch) |
| `train/sigreg` | 0.05-0.2 | > 0.5 (collapse risk) |
| `train/units` | 1.0-2.0 | > 3.0 (unit confusion) |
| `val/total` | Close to train | >> train (overfitting) |

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss NaN | Learning rate too high | Reduce `lr` from 5e-4 → 1e-4 |
| All formulas = "x1" | ValidityWeightedCE not working | Check `invalid_weight` and `min_length` |
| R² = -inf | Model outputs constant | Increase `lambda_sigreg` |
| OOM error | Batch size too large | Reduce from 128 → 64 |
| Slow I/O | Random access to Drive | Use `ContiguousChunkSampler` |

---

## 8. File Structure Reference

```
GSOC-LM-JEPA_for_Symbolic_Regression/
├── configs/
│   ├── base_config.yaml      # 3.4M params, 1M equations
│   └── small.yaml            # 2M params, 250k equations
├── data/
│   ├── synthetic_dataset.py  # Physics-informed generator
│   ├── aif_dataset.py        # AI Feynman loader
│   ├── tokenizer.py          # RPN encoding/decoding
│   └── unit_table.py         # Physics unit lookup
├── models/
│   ├── model.py              # LLMJEPA unified model
│   ├── encoder.py            # MixEncoder (ISA + column attn)
│   ├── decoder.py            # RPNDecoder (causal)
│   ├── target_encoder.py     # Formula → z_target
│   └── predictor.py          # JEPA bottleneck
├── training/
│   ├── train.py              # Entry point
│   ├── trainer.py            # Lightning module
│   └── losses.py             # JEPA + SIGReg + LM + Unit
├── inference/
│   └── generate.py           # Autoregressive generation
├── evaluation/
│   ├── evaluate.py           # Goldilocks suite
│   └── metrics.py            # R², NED, SA metrics
└── docs/
    ├── DATA_COMPARISON.md    # Synthetic vs AIF analysis
    └── SOTA_IMPROVEMENTS.md  # Enhancement roadmap
```

---

## 9. Recent Improvements (2026-03-30)

### ✅ Implemented

| Feature | Files Modified | Impact |
|---------|---------------|--------|
| **Commutative Augmentation** | `synthetic_dataset.py` | 2x training diversity |
| **PAD Token Masking** | `trainer.py`, `losses.py` | No gradient from padding |
| **Complexity Enforcement** | `losses.py` | Min 5 tokens per formula |
| **Type Annotation Fix** | `synthetic_dataset.py` | Correct list return handling |

### ⚠️ Known Limitations

| Limitation | Impact | Priority |
|------------|--------|----------|
| No Y supervision | Model doesn't learn X→y mapping | 🔴 Critical |
| No constant fitting | Can't fit c1, c2 to data | 🔴 Critical |
| Single candidate generation | No diversity sampling | 🟡 High |

**Roadmap:** See `docs/SOTA_IMPROVEMENTS.md` for implementation priorities.

---

## 10. Key Design Decisions

### Why JEPA instead of Standard LM?

**Problem:** Pure language modeling ignores the **numerical grounding** of formulas.

**JEPA Solution:**
- `z_context` encodes the **data distribution** (X, y pairs)
- `z_target` encodes the **formula semantics**
- Predictor learns: "What formula representation matches this data?"

**Benefit:** Model learns **X → formula** mapping, not just token sequences.

### Why SIGReg instead of EMA?

**Problem:** EMA (Exponential Moving Average) is complex to implement and tune.

**SIGReg Solution:**
- Apply isotropic Gaussian regularization to **both** encoders
- No detachment, no momentum terms
- Prevents collapse via distribution matching

**Benefit:** Simpler implementation, same collapse prevention.

### Why ValidityWeightedCE instead of Hard Masking?

**Problem:** Hard masking (setting logits to `-inf`) creates discontinuous gradients.

**ValidityWeightedCE Solution:**
- Penalize **probability mass** on invalid tokens
- Gradient flows through `invalid_prob_mass`
- Soft, continuous penalty

**Benefit:** Smoother learning, better convergence.

### Why IEEE-754 Bit Encoding?

**Problem:** LLMs struggle with raw float numbers (e.g., `3.14159`).

**Solution:** Decompose floats into binary features:
```
3.14159 → [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # 16 bits
           ↑sign     ↑exponent (8 bits)    ↑mantissa (7 bits)
```

**Benefit:** Model learns **bit-level numerical patterns**, not just token embeddings.

---

*Documentation generated from codebase analysis on 2026-03-30*  
*For questions or issues, refer to individual module docstrings*
