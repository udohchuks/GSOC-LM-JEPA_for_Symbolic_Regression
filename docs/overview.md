# LLM-JEPA for Symbolic Regression — Project Overview

> **What it does:** Given a table of numerical observations (`X`, `y`), the model generates a concise mathematical formula that describes the relationship — like `F = m * a` or `E = q / (4πε₀r²)` — in a physically consistent, human-interpretable form.

---

## The Problem

Symbolic Regression (SR) is the task of discovering the underlying mathematical law governing a dataset, not just fitting a curve. This is harder than standard regression because the output is a symbolic expression with arbitrary structure, not a fixed-length vector.

Classical SR (genetic programming, MCTS) is slow — O(hours per equation). Neural approaches must be fast enough to generate a formula in milliseconds from raw data.

---

## The Approach: LLM-JEPA + ODEFormer Inference

This project combines three ideas:

**JEPA (Joint Embedding Predictive Architecture):**
Instead of reconstructing the input, the model learns a *representation* of the data (`z_context`) and separately encodes the formula (`z_target`), then trains a predictor to bridge them. This produces rich, abstract data representations that don't over-fit to surface-level patterns.

**LLM-style Decoder:**
The data representation is fed to an autoregressive decoder that generates formulas token-by-token in Reverse Polish Notation (RPN). RPN eliminates parentheses and makes the grammar checkable at every step with a simple stack counter — invalid tokens can be hard-blocked during generation.

**ODEFormer-Style Inference (Sampling & Ranking):**
At inference, the model generates a diverse pool of N candidates (default 50) using temperature-controlled sampling. Each candidate is skeletonized (constants replaced with placeholders), deduplicated, and constants are fitted using BFGS optimization. Candidates are ranked by R² on the full dataset, and the best is selected. This "diversity search" dramatically improves recovery rates compared to greedy decoding.

Together: the encoder learns *what the data means*, the decoder *writes the formula*, and ODEFormer inference *finds the best candidate*.

---

## How It Works (End-to-End)

```
Input: X [observations × variables], units for each variable

    ┌─────────────────────────────────────────────────────┐
    │  IEEE-754 Encoding: each float → 16 binary bits     │
    └──────────────────────────┬──────────────────────────┘
                               ↓
    ┌──────────────────────────────────────────────────────┐
    │  MixEncoder (ISAB + Column Attention)                │
    │  Reads all data points for all variables at once     │
    │  → z_context [d_model]      (global summary)         │
    │  → var_summaries [n_vars, d_model]  (per-variable)  │
    └────┬───────────────────────────┬────────────────────┘
         │                           │
         ↓ (training only)           ↓ (training only)
    ┌──────────────┐      ┌─────────────────────────┐
    │TargetEncoder │      │     JEPAPredictor        │
    │ formula→     │      │ z_context + var_summaries│
    │ z_target     │      │ → z_hat                  │
    └──────┬───────┘      └──────────┬──────────────┘
           │                         │
           └─── MSE Loss ────────────┘
                (L_jepa)

          z_context
               ↓
     ┌──────────────────────────────────────────┐
     │  RPNDecoder (Temperature Sampling × N)   │
     │  Generates diverse candidate pool        │
     │  → N RPN sequences                       │
     └──────────────────┬───────────────────────┘
                        ↓
     ┌──────────────────────────────────────────┐
     │  ODEFormer Pipeline (Goldilocks Eval)   │
     │  1. Skeletonize: replace constants      │
     │     c1,c2,... → CONST placeholders      │
     │  2. Deduplicate: unique skeletons only  │
     │  3. BFGS Optimization: fit constants    │
     │     on FULL data (max_iter=100)         │
     │  4. Rank by R², select top-k            │
     └──────────────────┬───────────────────────┘
                        ↓
Output: "x1 x2 * x3 +" → sympy.parse → F = x1*x2 + x3
        with fitted constants: F = 2.5*x1*x2 + 3.0*x3
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **IEEE-754 bit encoding** | Represents any float exactly as 16 binary features. No normalisation needed — normalisation would destroy the symbolic structure (e.g. `2πr` becomes `r` after standardisation). |
| **RPN (postfix) notation** | Parenthesis-free, stack-checkable grammar. Validity of any token can be determined in O(1) from the stack depth alone — no parser needed. |
| **SIGReg instead of EMA** | Standard JEPA uses an Exponential Moving Average (EMA) teacher to prevent representational collapse. SIGReg (LeJEPA) instead constrains embeddings to be isotropic Gaussian, allowing both encoders to be fully trainable. Simpler, no momentum hyperparameter. |
| **Unit dimensional analysis** | Physics formulas are dimensionally consistent. The model encodes SI units (mass, length, time, current, temperature) for every variable and token. Unit violations are penalised during training and can be checked at inference. |
| **BFGS Post-processing** | Numerical constants (e.g. `2.718`, `9.81`) are represented as placeholder tokens (`c1`...`c5`) during generation, then fitted to data with SciPy's L-BFGS-B optimizer (100 iterations, 5 restarts). This cleanly separates symbolic structure from numerical optimization. |
| **Diversity Pool Sampling & R² Ranking** | To maximize recovery, the model samples N candidates (default 50) using temperature-controlled sampling (T=0.1). Unique skeletons are deduplicated, constants are fitted with BFGS, and candidates are ranked by R² on the FULL dataset. This "diversity search" is critical for recovering complex formulas. |
| **Goldilocks Evaluation Metrics** | Three core metrics provide complete performance picture: (1) R² for numeric precision, (2) Symbolic Accuracy for functional equivalence via random point evaluation, (3) Normalized Edit Distance (NED) for structural similarity via prefix trees. |
| **ISAB encoder** | Induced Self-Attention Block uses M=32 inducing points. Reduces O(N²) attention to O(N·M) for N data points — critical for tables with thousands of rows. |
| **Synthetic pretraining + AIF fine-tuning** | Pretraining on physics-informed synthetic equations (dimensionally valid by construction) gives the model a strong prior. The AI Feynman dataset is used for evaluation/fine-tuning. |
| **Centralized Configuration** | All hyperparameters in `configs/base_config.yaml`. Tune `pool_size`, `temperature`, `max_iter` (BFGS), and `n_workers` without touching Python code. |
| **Chunked Data Loading** | Synthetic data saved in chunks (2000 equations/part) with lazy loading. Supports num_workers=4+ on Colab with thread-safe cache and LRU eviction (~32 MB memory). |

---

## Project Structure

```
GSOC-LM-JEPA_for_Symbolic_Regression/
│
├── data/
│   ├── tokenizer.py          # RPN vocabulary, encode/decode, validity mask
│   ├── unit_table.py         # SI unit vectors, class index mapping
│   ├── aif_dataset.py        # AI Feynman dataset loader (evaluation)
│   ├── synthetic_dataset.py  # Physics-informed synthetic data (Lazy loading)
│   ├── generate_data.py      # Standalone generation CLI
│   └── utils.py              # IEEE-754 encoding (uint8 optimized), noise
│
├── models/
│   ├── embedders.py          # DataEmbedder (IEEE-754→embedding), UnitEmbedder
│   ├── encoder.py            # MixEncoder = ISAB + Column Attention
│   ├── target_encoder.py     # Formula → z_target (training only)
│   ├── predictor.py          # JEPAPredictor (z_context → z_hat)
│   ├── decoder.py            # RPNDecoder + UnitPredictionHead
│   ├── model.py              # LLMJEPA: assembles all components
│   └── evaluator.py          # ModelEvaluator: high-level eval API
│
├── training/
│   ├── losses.py             # SIGRegLoss, JEPALoss, ValidityWeightedCE, UnitLoss
│   ├── trainer.py            # PyTorch Lightning module (LLMJEPAModule)
│   └── train.py              # CLI entry point, YAML config, data splitting
│
├── inference/
│   ├── generate.py           # InferenceModel: encode + autoregressive generate
│   └── beam_search.py        # ODEFormer pipeline: skeletonize, fit_and_score
│
├── evaluation/
│   ├── metrics.py            # R², Symbolic Accuracy, NED, Constant Recovery
│   └── evaluate.py           # Goldilocks eval suite with BFGS fitting
│
├── tests/
│   ├── test_tokenizer.py     # Tokenizer tests (17 tests)
│   ├── test_metrics.py       # Metrics tests (21 tests)
│   ├── test_inference.py     # Inference tests (11 tests)
│   ├── test_integration.py   # Integration tests (16 tests)
│   └── test_models_and_loss.py # Model/loss tests (13 tests)
│
├── configs/
│   ├── base_config.yaml      # Full training/inference hyperparameters
│   └── smoke_test.yaml       # Fast local validation (1 epoch, CPU)
│
├── predict.py                # CLI: load checkpoint → generate formula
├── run_eval.py               # CLI: run Goldilocks evaluation suite
└── docs/
    ├── overview.md           # This file
    └── technical_reference.md # Module-by-module documentation
```

---

## Training

### 1. Generate Synthetic Data (Required)
Large-scale pretraining requires a synthetic corpus. Generate this **before** training:

```bash
# Generate 1M equations with chunking (Colab-optimized)
python -m data.generate_data --config configs/base_config.yaml

# This creates:
#   cache/synthetic_1M/metadata_manifest.pt  (index for instant startup)
#   cache/synthetic_1M/part_0.pt             (2000 equations)
#   cache/synthetic_1M/part_1.pt             (2000 equations)
#   ...
#   cache/synthetic_1M/part_499.pt           (2000 equations)
```

**Chunking benefits:**
- Saves incrementally to disk (no RAM exhaustion)
- Supports parallel generation (num_workers=4)
- Manifest file enables instant dataset startup

### 2. Start Training
The training script uses `LazySyntheticDataset` to load chunks on-demand:

```bash
# Training with chunked data loading
python -m training.train --config configs/base_config.yaml
```

**DataLoader configuration (from base_config.yaml):**
```yaml
data:
  num_workers: 4           # Colab: 4, Cloud: 8+
  chunk_size: 2000         # Equations per chunk file
  max_cache_size: 16       # Max chunks in memory (~32 MB)
```

**Features:**
- **Lazy loading:** Only loads required chunks, not entire dataset
- **Thread-safe cache:** Uses locks for multi-worker safety
- **LRU eviction:** Automatically removes old chunks when cache is full
- **Persistent workers:** Workers persist across epochs (reduces spawn overhead)

### 3. Monitor
```bash
tensorboard --logdir tb_logs
```

**Logged scalars:** `train/total`, `train/jepa`, `train/sigreg`, `train/lm`, `train/units`, `val/total`, `val/jepa`, `val/lm`

---

## Memory Usage

| Component | Memory |
|-----------|--------|
| **Per chunk file** | ~1-2 MB (2000 equations) |
| **Max cache (16 chunks)** | ~16-32 MB |
| **DataLoader prefetch (4 workers)** | ~8 MB |
| **Total overhead** | ~24-40 MB |

This is far more efficient than loading 1M equations into RAM (~500 MB).

---

## Inference

### Single Equation Prediction

```bash
# From a custom CSV file (last column = target y)
python predict.py --config configs/base_config.yaml \
                  --ckpt checkpoints/best.ckpt \
                  --csv my_data.csv

# From the Feynman dataset by equation ID
python predict.py --config configs/base_config.yaml \
                  --ckpt checkpoints/best.ckpt \
                  --eq_id I.12.1
```

**Output:** RPN tokens + SymPy expression (with BFGS-fitted constants)

### Inference Configuration

All inference parameters are controlled via `configs/base_config.yaml`:

```yaml
inference:
  pool_size: 50              # Number of candidates to sample
  max_len: 45                # Max RPN tokens per formula
  temperature: 0.1           # Sampling temperature for diversity
  max_iter: 100              # BFGS iterations per skeleton
  n_restarts: 5              # BFGS restarts for better convergence
  n_workers: 8               # CPU workers for parallel fitting
  top_k: 5                   # Number of top candidates to return
```

---

## Evaluation

### Goldilocks Evaluation Suite

```bash
python run_eval.py --config configs/base_config.yaml \
                   --ckpt checkpoints/best.ckpt
```

This runs the full evaluation pipeline on the AI Feynman dataset and generates:
- `results/metrics.json` - Raw metrics
- `results/goldilocks_report.md` - Human-readable report

**Metrics reported:**
- **R² (Numeric Precision):** Quality of fit on original data points
- **Symbolic Accuracy (SA):** Functional equivalence via random point evaluation
- **Normalized Edit Distance (NED):** Structural similarity via prefix trees
- **Constant Recovery (CR):** Accuracy of fitted constants vs ground truth
- **Node Count:** Formula complexity
- **Latency:** Inference time (generation + BFGS)

**Output format:**
```
==============================================================================================
Equation                                 |     R² | SymAcc | ConstRec |    NED | Nodes
---------------------------------------------------------------------------------------------
I.12.1                                   |   0.99 |    1.0 |      1.0 |   0.00 |     5
I.6.2a                                   |   0.95 |    0.0 |      0.0 |   0.15 |     7
...
---------------------------------------------------------------------------------------------
MEAN                                     |   0.85 |   0.75 |     0.60 |   0.12 |     6
==============================================================================================
```

### Evaluation Configuration

Override inference parameters via CLI:

```bash
python run_eval.py --ckpt checkpoints/best.ckpt \
                   --n_candidates 100 \
                   --temperature 0.2
```

For challenging periodic functions (sin, cos), increase BFGS budget:
```bash
# Edit configs/base_config.yaml:
inference:
  max_iter: 200      # More iterations for complex functions
  n_restarts: 10     # More restarts for better coverage
```
