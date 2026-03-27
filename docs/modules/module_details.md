# Module: `models/embedders.py`

## DataEmbedder

**Purpose:** Converts raw IEEE-754 bit-encoded scalars into dense per-scalar embeddings.

**Input:** `X_bits [B, N, n_vars, 16]` — each scalar is a 16-bit binary vector.

**Output:** `[B, N, n_vars, d_model]` — one embedding per scalar value.

**How it works:**
1. A `Linear(16 → d_model)` layer projects each 16-bit vector into embedding space.
2. A learnable **variable identity embedding** is added so the model can distinguish which column (x1, x2, …) a scalar belongs to. Without this, `x1=2.0` and `x2=2.0` would produce identical representations.
3. No row positional encoding is used — rows are treated as an unordered set (permutation invariant).

---

## UnitEmbedder

**Purpose:** Encodes the physical unit signature of each variable into a dense vector.

**Input:** `unit_idx [B, n_vars, 5]` — class indices for 5 SI dimensions `[m, s, kg, K, V]`. Each index is in `[0, 8]` representing exponents from -4 to +4.

**Output:** `[B, n_vars, d_model]`

**How it works:**
- 5 independent `nn.Embedding(9, d_model)` tables — one per unit dimension.
- The final embedding is the **sum** of all 5 lookups, followed by LayerNorm.
- Same unit vector → same embedding (verified by tests).

---

# Module: `models/encoder.py`

## MixEncoder

**Purpose:** The core context encoder. Fuses data and unit embeddings, then produces `z_context` (a single vector summarizing the entire data table) and `var_summaries` (per-variable representations).

**Architecture (5 stages):**

| Stage | Operation | Shape Change |
|-------|-----------|-------------|
| 1. Fuse | `data_emb + unit_emb` (broadcast add) | `[B,N,V,d]` |
| 2. ISAB | Row-level set attention per variable | `[B*V,N,d]` → `[B*V,N,d]` |
| 3. PMA (row) | Compress N rows → 1 per variable | `[B*V,N,d]` → `[B,V,d]` |
| 4. Column Attn | Variables attend to each other | `[B,V,d]` → `[B,V,d]` |
| 5. PMA (aggregate) | All variables → z_context | `[B,V,d]` → `[B,d]` |

**Key design choices:**
- **ISAB** (Induced Set Attention Block): Uses `m` learnable inducing points to reduce self-attention from O(N²) to O(Nm). Critical for handling 100k rows per equation.
- **No positional encoding** in column attention — the model is equivariant over variable ordering.
- **Row permutation invariance** is verified by the built-in test.

---

# Module: `models/decoder.py`

## RPNDecoder

**Purpose:** Autoregressive causal transformer decoder that generates RPN token sequences.

**Inputs:** `token_ids [B, T]`, `z_context [B, d_model]`, `unit_matrix [B, n_vars, 5]`

**Output:** `logits [B, T, vocab_size]`, `h_states [B, T, d_model]`

**Key features:**
- **PhysicsTokenEmbedding:** Augments standard token + positional embeddings with unit information for variable tokens. This makes dimensional structure part of every variable's representation.
- **Weight tying:** The LM head shares weights with the token embedding, reducing parameters.
- **Causal mask:** Standard autoregressive masking to prevent the decoder from seeing future tokens during training.
- At inference, a **validity mask** based on the RPN stack counter is applied before softmax so that only grammatically valid tokens can be generated.

## UnitPredictionHead

**Purpose:** 5 single-layer classifiers (one per SI dimension) applied to decoder hidden states. Forces `h_t` to linearly encode dimensional information.

**Training scaffold only** — this head is discarded at inference time.

---

# Module: `models/target_encoder.py`

## TargetEncoder

**Purpose:** Encodes the ground-truth RPN formula into `z_target`, the latent representation that the Predictor tries to match.

**Architecture:**
1. `PhysicsTokenEmbedding` (same as decoder)
2. Bidirectional `TransformerEncoder` (pre-norm, GELU)
3. `PMA(k=1)` to pool the sequence into a single vector

**SIGReg note:** All parameters are trainable. Gradients flow through this module — collapse prevention comes from SIGReg loss, not from detaching.

---

# Module: `models/predictor.py`

## JEPAPredictor

**Purpose:** Predicts `z_target` from `z_context` and `var_summaries`. The JEPA prediction objective.

**Deliberately small** — uses a half-dimension bottleneck (`d_model // 2`) to prevent the "lazy encoder" problem where a powerful predictor learns the mapping without forcing the encoder to extract good features.

**Architecture:**
1. Project `z_context` and `var_summaries` to bottleneck dimension
2. Cross-attention: `z_context` queries `var_summaries`
3. Gated output projection back to `d_model`

---

# Module: `training/losses.py`

## Loss Components

| Loss | Formula | Purpose |
|------|---------|---------|
| `JEPALoss` | `MSE(z_hat, z_target.detach())` | Prediction objective |
| `SIGRegLoss` | LeJEPA's Epps-Pulley test | Collapse prevention |
| `ValidityWeightedCE` | CE with 2× penalty for invalid tokens | Language modeling |
| `UnitLoss` | Mean CE over 5 unit dimensions | Dimensional awareness |

The `ValidityWeightedCE` is fully vectorized using `torch.cumsum` to compute stack depths for the entire batch at once — no Python loops over positions.

---

# Module: `training/trainer.py`

## LLMJEPAModule

**Purpose:** PyTorch Lightning module that orchestrates the full training loop.

**Key features:**
- Wraps the `LLMJEPA` model and `LLMJEPALoss`
- AdamW optimizer with linear warmup + cosine decay LR schedule
- Logs all 4 loss components to TensorBoard (`train/jepa`, `train/sigreg`, `train/lm`, `train/units`)

---

# Module: `data/tokenizer.py`

## RPN Tokenizer

**Vocabulary:** ~41 tokens total:
- 4 special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`
- 9 variable tokens: `x1` – `x9`
- 11 constant tokens: `0, 1, 2, 3, pi, e, c1–c5`
- 4 binary operators: `+, -, *, /`
- 13 unary operators: `sqrt, sq, exp, log, sin, cos, tan, arcsin, arccos, arctan, inv, abs, neg`

**Key functions:**
- `formula_string_to_rpn()`: SymPy expression → RPN token list
- `get_valid_next_tokens()`: Returns valid token indices at each generation step based on stack depth
- `is_valid_rpn()`: O(n) stack-based validation

---

# Module: `inference/generate.py`

## InferenceModel

**Purpose:** Stripped-down model for inference only. Contains only the encoder path (`DataEmbedder → MixEncoder`) and `RPNDecoder`. No target encoder, predictor, or unit head.

**`generate()` method:**
- Batched autoregressive generation with validity masking
- Supports greedy and temperature-based sampling
- Vectorized stack depth updates using a pre-built arity map
