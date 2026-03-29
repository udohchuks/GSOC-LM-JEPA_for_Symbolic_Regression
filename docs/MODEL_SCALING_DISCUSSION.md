# Model Scaling Discussion - 20k Synthetic Data

## Current Situation

**Your Requirements:**
- Dataset size: ~20k synthetic equations
- Target: <1M parameters (currently 3.4M)
- Goal: Avoid overfitting on small dataset

---

## Analysis of Current Base Model (3.4M params)

### Configuration
```yaml
d_model:      128
n_heads:      4
n_enc_layers: 4
n_dec_layers: 5
n_isab:       2
n_col_attn:   2
m_inducing:   32
```

### Problem
Training a 3.4M parameter model on only 20k equations will **severely overfit**:
- **Parameter-to-data ratio:** 170 params per equation (way too high)
- **Rule of thumb:** Should be 5-10 params per data point for good generalization
- **Expected behavior:** Training loss → 0, Validation loss → high

---

## Proposed Tiny Model (~800K params)

### Configuration (in `configs/small_20k_config.yaml`)
```yaml
d_model:      48          # 128 → 48 (62% reduction)
n_heads:      4           # Keep same (attention diversity important)
n_enc_layers: 2           # 4 → 2 (50% reduction)
n_dec_layers: 2           # 5 → 2 (60% reduction)
n_isab:       1           # 2 → 1 (50% reduction)
n_col_attn:   1           # 2 → 1 (50% reduction)
m_inducing:   16          # 32 → 16 (50% reduction)
max_n_vars:   7           # 9 → 7 (less padding)
```

### Expected Parameter Distribution

Based on architecture analysis:

| Component | Base (3.4M) | Tiny (~800K) | Reduction |
|-----------|-------------|--------------|-----------|
| **Embeddings** | ~400K | ~100K | 75% ↓ |
| **MixEncoder** | ~1.2M | ~250K | 79% ↓ |
| **RPNDecoder** | ~1.5M | ~350K | 77% ↓ |
| **Predictor** | ~200K | ~80K | 60% ↓ |
| **Heads** | ~100K | ~20K | 80% ↓ |
| **TOTAL** | **3.4M** | **~800K** | **76% ↓** |

### Why These Specific Changes?

1. **d_model: 128 → 48**
   - Biggest impact on all linear layers
   - Quadratic effect on attention: `(d_model × d_model)`
   - Safe reduction: embedding dimension doesn't need to be huge for symbolic regression

2. **n_enc_layers: 4 → 2**
   - Encoder processes tabular data (simpler than sequences)
   - 2 layers sufficient for feature extraction

3. **n_dec_layers: 5 → 2**
   - Decoder generates RPN sequences (needs more capacity)
   - But 2 layers still enough for short sequences (max 50 tokens)

4. **n_isab: 2 → 1, n_col_attn: 2 → 1**
   - These are attention mechanisms within encoder
   - For 20k equations, single pass is sufficient

5. **m_inducing: 32 → 16**
   - Inducing points for set attention
   - Fewer points = less computation, slightly less expressivity

6. **max_n_vars: 9 → 7**
   - AIF max is 10, but 99% have ≤7 variables
   - Reduces embedding table size

---

## Alternative Configurations to Consider

### Option A: Ultra-Tiny (~400K params) - FOR 10K DATA
```yaml
d_model:      32
n_heads:      4
n_enc_layers: 2
n_dec_layers: 2
n_isab:       1
n_col_attn:   1
m_inducing:   16
```
**Use when:** Only 10k equations, very limited compute

### Option B: Balanced (~1.2M params) - FOR 50K DATA
```yaml
d_model:      64
n_heads:      4
n_enc_layers: 3
n_dec_layers: 3
n_isab:       2
n_col_attn:   1
m_inducing:   24
```
**Use when:** 50k equations, want better performance

### Option C: Current Tiny (~800K params) - FOR 20K DATA ⭐
```yaml
d_model:      48
n_heads:      4
n_enc_layers: 2
n_dec_layers: 2
n_isab:       1
n_col_attn:   1
m_inducing:   16
```
**Use when:** 20k equations (YOUR CASE)

---

## Training Strategy for Small Models

### Key Adjustments in `small_20k_config.yaml`

1. **Higher Learning Rate:** `5e-4` (vs 3e-4)
   - Small models need stronger gradients
   - Faster convergence

2. **More Epochs:** `15` (vs 1 for base)
   - Small data → can train longer without overfitting
   - But monitor validation loss!

3. **Increased SIGReg Weight:** `0.5` (vs 0.1)
   - More regularization for small model
   - Prevents collapse

4. **Smaller Batches:** `64` (vs 128)
   - Better gradient estimates for small data
   - More frequent updates

5. **Shorter Sequences:** `n_rows: 100` (vs 200)
   - Less data per equation
   - Faster training, less memory

---

## Expected Performance

### Tiny Model (800K) on 20k Equations

| Metric | Expected | Notes |
|--------|----------|-------|
| **Training Time** | 2-3 hrs | Colab T4, 15 epochs |
| **GPU Memory** | ~4GB | Well within Colab Free |
| **Exact Recovery** | 15-25% | Reasonable for small data |
| **Mean R²** | 0.6-0.75 | Good baseline |
| **Overfitting** | Low | If validated properly |

### Comparison to Base Model

| Metric | Base (3.4M) | Tiny (800K) |
|--------|-------------|-------------|
| Params | 3.4M | 0.8M |
| Train Time | 8-10 hrs | 2-3 hrs |
| GPU Memory | 10-12GB | 4GB |
| Overfitting (20k data) | Severe | Minimal |
| Performance (20k data) | Poor (overfits) | Better (fits well) |

---

## Recommendations

### For Your 20k Dataset:

1. **Use `configs/small_20k_config.yaml`** as-is
   - Already tuned for your use case
   - ~800K params, good balance

2. **Monitor During Training:**
   ```bash
   tensorboard --logdir tb_logs/small_20k/
   ```
   - Watch val/total vs train/total
   - Stop early if gap widens

3. **If Overfitting:**
   - Reduce epochs (15 → 10)
   - Increase dropout (0.1 → 0.2)
   - Reduce d_model further (48 → 40)

4. **If Underfitting:**
   - Increase epochs (15 → 20)
   - Increase d_model (48 → 56)
   - Add more synthetic data (20k → 30k)

---

## Next Steps

1. **Review the config:** `configs/small_20k_config.yaml`
2. **Discuss adjustments:** Any specific constraints?
3. **Generate 20k data:** Using the small config
4. **Train and evaluate:** Monitor for overfitting

---

**Questions to Consider:**

1. Is 800K params acceptable, or do you need even smaller?
2. What's your target exact recovery rate on AIF?
3. Are you limited by training time or model performance?
4. Do you plan to scale up to 50k-100k equations later?
