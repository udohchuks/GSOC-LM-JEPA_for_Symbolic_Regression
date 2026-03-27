# LLM-JEPA for Symbolic Regression — Project Overview

> **What it does:** Given a table of numerical observations (`X`, `y`), the model generates a concise mathematical formula that describes the relationship — like `F = m * a` or `E = q / (4πε₀r²)` — in a physically consistent, human-interpretable form.

---

## The Problem

Symbolic Regression (SR) is the task of discovering the underlying mathematical law governing a dataset, not just fitting a curve. This is harder than standard regression because the output is a symbolic expression with arbitrary structure, not a fixed-length vector.

Classical SR (genetic programming, MCTS) is slow — O(hours per equation). Neural approaches must be fast enough to generate a formula in milliseconds from raw data.

---

## The Approach: LLM-JEPA

This project combines two ideas:

**JEPA (Joint Embedding Predictive Architecture):**  
Instead of reconstructing the input, the model learns a *representation* of the data (`z_context`) and separately encodes the formula (`z_target`), then trains a predictor to bridge them. This produces rich, abstract data representations that don't over-fit to surface-level patterns.

**LLM-style Decoder:**  
The data representation is fed to an autoregressive decoder that generates formulas token-by-token in Reverse Polish Notation (RPN). RPN eliminates parentheses and makes the grammar checkable at every step with a simple stack counter — invalid tokens can be hard-blocked during generation.

Together: the encoder learns *what the data means*, the decoder *writes the formula*.

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
    ┌─────────────────────────────────────┐
    │  RPNDecoder (autoregressive)         │
    │  Generates formula token by token    │
    │  Validity mask blocks bad tokens     │
    │  → token logits → formula string    │
    └─────────────────────────────────────┘

Output: "x1 x2 * x3 +" → sympy.parse → F = x1*x2 + x3
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **IEEE-754 bit encoding** | Represents any float exactly as 16 binary features. No normalisation needed — normalisation would destroy the symbolic structure (e.g. `2πr` becomes `r` after standardisation). |
| **RPN (postfix) notation** | Parenthesis-free, stack-checkable grammar. Validity of any token can be determined in O(1) from the stack depth alone — no parser needed. |
| **SIGReg instead of EMA** | Standard JEPA uses an Exponential Moving Average (EMA) teacher to prevent representational collapse. SIGReg (LeJEPA) instead constrains embeddings to be isotropic Gaussian, allowing both encoders to be fully trainable. Simpler, no momentum hyperparameter. |
| **Unit dimensional analysis** | Physics formulas are dimensionally consistent. The model encodes SI units (mass, length, time, current, temperature) for every variable and token. Unit violations are penalised during training and can be checked at inference. |
| **BFGS post-processing** | Numerical constants (e.g. `2.718`, `9.81`) are represented as placeholder tokens (`c1`...`c5`) during generation, then fitted to data with L-BFGS-B after generation. This cleanly separates symbolic structure from numerical optimisation. |
| **ISAB encoder** | Induced Self-Attention Block uses M=32 inducing points. Reduces O(N²) attention to O(N·M) for N data points — critical for tables with thousands of rows. |
| **Synthetic pretraining + AIF fine-tuning** | Pretraining on physics-informed synthetic equations (dimensionally valid by construction) gives the model a strong prior. The AI Feynman dataset is used for evaluation/fine-tuning. |

---

## Project Structure

```
GSOC-LM-JEPA_for_Symbolic_Regression/
│
├── data/
│   ├── tokenizer.py          # RPN vocabulary, encode/decode, validity mask
│   ├── unit_table.py         # SI unit vectors, class index mapping
│   ├── aif_dataset.py        # AI Feynman dataset loader (evaluation)
│   ├── synthetic_dataset.py  # Physics-informed synthetic data generator
│   └── utils.py              # IEEE-754 encoding, noise, unit targets
│
├── models/
│   ├── embedders.py          # DataEmbedder (IEEE-754→embedding), UnitEmbedder
│   ├── encoder.py            # MixEncoder = ISAB + Column Attention
│   ├── target_encoder.py     # Formula → z_target (training only)
│   ├── predictor.py          # JEPAPredictor (z_context → z_hat)
│   ├── decoder.py            # RPNDecoder + UnitPredictionHead
│   └── model.py              # LLMJEPA: assembles all components
│
├── training/
│   ├── losses.py             # SIGRegLoss, JEPALoss, ValidityWeightedCE, UnitLoss
│   ├── trainer.py            # PyTorch Lightning module (LLMJEPAModule)
│   └── train.py              # CLI entry point, YAML config, data splitting
│
├── inference/
│   └── generate.py           # InferenceModel: encode + autoregressive generate
│
├── evaluation/
│   ├── metrics.py            # R², Acc_τ, node count, BFGS, exact check
│   └── evaluate.py           # Full eval suite: noise, data efficiency, OOD
│
├── configs/
│   ├── base_config.yaml      # Full training hyperparameters
│   └── smoke_test.yaml       # Fast local validation (1 epoch, CPU)
│
├── predict.py                # CLI: load checkpoint → generate formula
├── run_eval.py               # CLI: run full evaluation suite
└── tests/                    # pytest: data, model, pipeline
```

---

## Training

```bash
# Full training (from base config)
python -m training.train --config configs/base_config.yaml

# Smoke test (1 epoch, CPU, fast)
python -m training.train --config configs/smoke_test.yaml

# Monitor
tensorboard --logdir tb_logs
```

**Logged scalars:** `train/total`, `train/jepa`, `train/sigreg`, `train/lm`, `train/units`, `val/total`, `val/jepa`, `val/lm`

---

## Inference

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

---

## Evaluation

```bash
python run_eval.py --config configs/base_config.yaml \
                   --ckpt checkpoints/best.ckpt
```

**Metrics reported:**
- Exact symbolic recovery rate
- Mean R² (pre and post BFGS constant fitting)
- Acc_τ at τ = 0.1%, 1%, 10%
- Valid RPN rate / Dimensional validity rate
- Noise tolerance (R² vs additive noise level)
- Data efficiency (R² vs number of data points seen)
- Out-of-distribution extrapolation R²
- Mean formula complexity (node count)
- Inference latency (generation + BFGS)
