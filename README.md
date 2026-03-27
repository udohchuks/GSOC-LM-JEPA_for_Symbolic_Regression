# LLM-JEPA for Symbolic Regression

A Joint Embedding Predictive Architecture (JEPA) for Symbolic Regression. Parses tabular data into IEEE-754 bit-level embeddings and trains a decoder to generate Reverse Polish Notation (RPN) mathematical formulas, validated by unit-dimensional constraints.

## Features
- **IEEE-754 Data Encoding** — Floating-point inputs tokenised at the bit level
- **MixEncoder + JEPA** — Predicts formulas in latent space via set-attention and column-attention
- **Unit-Validated Decoding** — Masks invalid tokens based on RPN stack depth and dimensional analysis
- **SIGReg Collapse Prevention** — No EMA needed; both encoders are trainable
- **Comprehensive Evaluation** — Noise tolerance, data efficiency, extrapolation, complexity analysis

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-username/GSOC-LM-JEPA_for_Symbolic_Regression.git
cd GSOC-LM-JEPA_for_Symbolic_Regression
pip install -r requirements.txt
```

### 2. Download Dataset

Download the AI Feynman dataset with units:

```bash
python -c "import tarfile; tar = tarfile.open('Feynman_with_units.tar.gz'); tar.extractall('data/'); tar.close()"
```

> The `data/Feynman_with_units/` directory should contain ~100 data files (e.g. `I.6.2a`, `I.12.1`, etc.)

### 3. Verify Data Loading

```bash
python run_test_load.py
```

Expected output:
```
Parsing AIF equations from data/FeynmanEquations.csv
Found 100 equations
...
Preprocessed 99/100 equations (1 failed)
Loaded 99 equations
```

---

## Training

All hyperparameters are in `configs/base_config.yaml`. Train with:

```bash
python -m training.train --config configs/base_config.yaml
```

**Key config options** (`configs/base_config.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `training.max_epochs` | 30 | Training epochs |
| `training.lr` | 3e-4 | Learning rate |
| `model.d_model` | 256 | Hidden dimension |
| `model.n_enc_layers` | 4 | Encoder transformer layers |
| `model.n_dec_layers` | 4 | Decoder transformer layers |
| `training.use_synthetic` | false | Use synthetic pretraining data |
| `hardware.accelerator` | gpu | `gpu` or `cpu` |

**Training outputs:**
- Checkpoints saved to `checkpoints/` (top-3 by validation loss + `last.ckpt`)
- TensorBoard logs saved to `tb_logs/`

**Monitor training:**
```bash
tensorboard --logdir tb_logs
```

---

## Inference

Run inference on a trained checkpoint to generate symbolic formulas.

### From a custom CSV file

The CSV should have variable columns followed by a target column (last column = output):

```bash
python predict.py --ckpt checkpoints/last.ckpt --csv path/to/your/data.csv
```

### From an AI Feynman equation

```bash
python predict.py --ckpt checkpoints/last.ckpt --eq_id I.6.2a
```

**Example output:**
```
Found equation I.6.2a: Ground truth is exp(-theta**2/2)/sqrt(2*pi)
Generated RPN Tokens: x1 sq neg 2 / exp pi 2 * sqrt inv *
SymPy Expression: exp(-x1**2/2)/sqrt(2*pi)
```

---

## Evaluation

Run the comprehensive evaluation suite on the full AI Feynman dataset:

```bash
python run_eval.py --ckpt checkpoints/last.ckpt
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/base_config.yaml` | Config file path |
| `--ckpt` | *(required)* | Checkpoint path |
| `--output` | `results/eval_results.json` | Output JSON file |
| `--n_restarts` | 3 | BFGS optimisation restarts |

**Metrics computed:**

| Category | Metrics |
|---|---|
| **Standard** | Exact recovery rate, valid RPN rate, dimensional validity |
| **Precision** | Pre-BFGS R², Post-BFGS R², BFGS improvement delta, Acc_τ at τ ∈ {0.1, 0.01, 0.001} |
| **Complexity** | Mean formula node count |
| **Robustness** | R² vs noise level (ε ∈ {0.001, 0.01, 0.1}), R² vs data size (N ∈ {10, 50, 100, 200}), extrapolation R² |
| **Operational** | Generation latency (ms), BFGS fitting latency (ms) |

Results are saved to `results/eval_results.json` and printed to the console.

---

## Running Tests

```bash
pytest tests/
```

Or without pytest:
```bash
python -m unittest discover tests
```

---

## Project Structure

```
├── configs/
│   └── base_config.yaml       # All hyperparameters
├── data/
│   ├── FeynmanEquations.csv   # Equation metadata
│   ├── Feynman_with_units/    # Raw data files (100 equations)
│   ├── aif_dataset.py         # AIF dataset loader
│   ├── synthetic_dataset.py   # Synthetic pretraining data
│   ├── tokenizer.py           # RPN tokenizer (~38 tokens)
│   ├── unit_table.py          # Physical unit lookup
│   └── utils.py               # IEEE-754 encoding, unit targets
├── models/
│   ├── model.py               # LLMJEPA unified model
│   ├── encoder.py             # MixEncoder (ISAB + column attention)
│   ├── decoder.py             # RPNDecoder (causal transformer)
│   ├── embedders.py           # Data + Unit embedders
│   └── predictor.py           # JEPA predictor (bottleneck)
├── training/
│   ├── train.py               # Training entry point
│   ├── trainer.py             # Lightning module
│   └── losses.py              # JEPA + SIGReg + LM + Unit losses
├── inference/
│   └── generate.py            # Autoregressive generation
├── evaluation/
│   ├── evaluate.py            # Full evaluation suite
│   └── metrics.py             # Metric functions (R², BFGS, etc.)
├── predict.py                 # CLI inference
├── run_eval.py                # CLI evaluation
├── run_test_load.py           # Data loading smoke test
└── requirements.txt
```

---

## Configuration

Edit `configs/base_config.yaml` to adjust:
- **Model size**: Scale `d_model`, `n_enc_layers`, `n_dec_layers` for larger models
- **Training**: Adjust `lr`, `max_epochs`, `warmup_steps`
- **Hardware**: Set `accelerator: cpu` for local testing, `accelerator: gpu` for training
- **Data**: Toggle `use_synthetic: true` for synthetic pretraining

For experiment-specific configs, create a new YAML file and pass it via `--config`:
```bash
python -m training.train --config configs/my_experiment.yaml
```