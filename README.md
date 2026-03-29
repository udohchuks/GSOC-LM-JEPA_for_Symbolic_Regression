# LLM-JEPA for Symbolic Regression

A Joint Embedding Predictive Architecture (JEPA) for Symbolic Regression. Parses tabular data into IEEE-754 bit-level embeddings and trains a decoder to generate Reverse Polish Notation (RPN) mathematical formulas, validated by unit-dimensional constraints.

## Features
- **Direct float16 bit encoding** — Represents any float as 16 binary features stored as `float16`. Removes `unpackbits` CPU bottleneck during training; data is fed directly to the model as a tensor.
- **MixEncoder + JEPA** — Predicts formulas in latent space via set-attention and column-attention
- **Unit-Validated Decoding** — Masks invalid tokens based on RPN stack depth and dimensional analysis
- **SIGReg Collapse Prevention** — No EMA needed; both encoders are trainable
- **BFGS Post-processing** — Numerical constants (e.g. `2.718`, `9.81`) are represented as placeholder tokens (`c1`...`c5`) during generation, then fitted to data with SciPy's BFGS optimizer (30 iterations). This cleanly separates symbolic structure from numerical optimization.
- **Beam Sampling & R² Ranking** — Explores top-N skeletons via temperature-based decoding to maximize exact recovery rate
- **Goldilocks Evaluation Suite** — Comprehensive metrics (R², NED, SA) + noise tolerance and data efficiency tests

---

## 📊 Dataset Documentation

### Synthetic Data Generation

The synthetic data generator creates physics-informed mathematical expressions for pretraining:

- **Dimensional Homogeneity:** Every equation is physically valid (mass, length, time, charge units)
- **Pattern-Based (80%):** 70 physics equation templates (inverse-square, relativistic, energy, waves, distance)
- **AI Feynman Match:** Optimized complexity distribution (mean 3.9-4.1 vars vs 4.09 in AIF)
- **Affine Transformations:** Increases diversity without changing structure

**Configuration Guide:**
- **20k-50k equations (~1M params, tiny predictor):** Use `configs/small.yaml`
- **100k+ equations (~3.4M params):** Use `configs/base_config.yaml`

**Full Documentation:** 
- [`docs/SYNTHETIC_DATA_GENERATION.md`](docs/SYNTHETIC_DATA_GENERATION.md) - Generation details
- [`docs/SMALL_MODEL_CONFIG.md`](docs/SMALL_MODEL_CONFIG.md) - Model scaling guide

### AI Feynman Comparison

Comprehensive comparison between synthetic pretraining data and AI Feynman evaluation data:

| Metric | Synthetic | AI Feynman | Gap |
|--------|-----------|------------|-----|
| Mean Variables | 3.9-4.1 | 4.09 | -5% to 0% |
| Mean Nodes | 12-14 | 12.47 | -4% to +12% |
| Mean Depth | 2.9-3.0 | 2.92 | ±0% |
| Division Freq | 0.20-0.30/eq | 0.4/eq | -25% to -50% |
| Negation Freq | 0.10-0.15/eq | 0.3/eq | -50% to -67% |

**Pattern Coverage:** 70 physics templates covering inverse-square, relativistic, thermodynamics, waves, electromagnetism

**Full Report:** [`docs/DATA_COMPARISON.md`](docs/DATA_COMPARISON.md)

**Run your own comparison:**
```bash
python -m data.compare_datasets \
    --config configs/base_config.yaml \
    --output results/data_comparison/ \
    --n_synthetic 500 \
    --n_aif 500
```

---

## Documentation

| Document | Description |
|---|---|
| [Overview](docs/overview.md) | High-level: motivation, architecture diagram, design decisions |
| [Technical Reference](docs/technical_reference.md) | Detailed: every module, class, implementation details |
| [Synthetic Data Generation](docs/SYNTHETIC_DATA_GENERATION.md) | Physics-informed data generation, AI Feynman comparison |
| [Dataset Comparison](docs/DATA_COMPARISON.md) | Synthetic vs AI Feynman analysis report |
| [Small Model Config](docs/SMALL_MODEL_CONFIG.md) | Model scaling guide for <100k equations |
| [Colab Notebooks](notebooks/README.md) | Google Colab workflow (generate → train → evaluate) |

---

## 🚀 Google Colab Notebooks

Three ready-to-run Colab notebooks for the complete workflow:

| Notebook | Purpose | Runtime | Duration |
|----------|---------|---------|----------|
| [01_generate_synthetic_data.ipynb](notebooks/01_generate_synthetic_data.ipynb) | Generate 1M synthetic equations | T4 GPU | 4-8 hrs |
| [02_train_model.ipynb](notebooks/02_train_model.ipynb) | Pretrain LLM-JEPA model | T4 GPU | 6-12 hrs |
| [03_evaluate_model.ipynb](notebooks/03_evaluate_model.ipynb) | Evaluate on AI Feynman | T4 GPU | 1-2 hrs |

**Quick Start:**
1. Open any notebook in Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run cells in order (dependency checks included)
4. All outputs saved to Google Drive (`SymbolicRegression/` folder)

**Full Guide:** [`notebooks/README.md`](notebooks/README.md)

---

## Installation

```bash
git clone https://github.com/chukwueke/GSOC-LM-JEPA_for_Symbolic_Regression.git
cd GSOC-LM-JEPA_for_Symbolic_Regression
pip install -r requirements.txt
```

### Download AI Feynman Dataset

Download from Dropbox (4.1 GB):

```bash
# Using wget
wget -O Feynman_with_units.tar.gz "https://www.dropbox.com/s/7kgfr00qpokgz8w/Feynman_with_units.tar.gz?dl=1"

# Or using curl
curl -L -o Feynman_with_units.tar.gz "https://www.dropbox.com/s/7kgfr00qpokgz8w/Feynman_with_units.tar.gz?dl=1"

# Extract to data/
python -c "import tarfile; tar = tarfile.open('Feynman_with_units.tar.gz'); tar.extractall('data/'); tar.close()"
```

> The `data/Feynman_with_units/` directory should contain ~100 data files (e.g. `I.6.2a`, `I.12.1`, etc.)

---



## Data Generation

Large-scale pretraining requires a synthetic corpus of mathematically valid and physically motivated equations. The `data.synthetic_dataset` generator produces these through a **Physics-Informed** pipeline:

- **Dimensional Homogeneity**: Variables are sampled from physical domains (Mechanics, EM, etc.). Operators (like `sin` or `+`) are only applied if they are dimensionally consistent—preventing invalid operations like adding "meters" to "kilograms".
- **Tree & Pattern Generation**: Uses a 50/50 mix of recursive tree growth (peak depth 4) and expert physics patterns (22+ templates) to ensure structural diversity.
- **V4 Tuning**: Biased toward 3–6 variables and tree depths of 4–5 to perfectly match the complexity distribution of the AI Feynman dataset.
- **Float16 Bit-Featurization**: Inputs are encoded into IEEE-754 bit patterns and stored as direct `float16` tensors, removing CPU-bound `unpackbits` overhead during training.
- **Lazy Sharding**: Equations are stored in sharded `.pt` parts, allowing for 1M+ scales without RAM exhaustion via `LazySyntheticDataset`.

### 1. Generate Synthetic Corpus

You can generate the pretraining data separately or in parallel with training:

```bash
# Generate 1M equations (configured in base_config.yaml)
python -m data.generate_data --config configs/base_config.yaml
```

### 2. Preprocess AIF Dataset
Precompute the evaluation dataset to ensure instant training startup:

```bash
python -m data.preprocess_aif --config configs/base_config.yaml
```

### 3. Start Training
The training script loads existing cache parts and automatically detects new parts as they are generated:

```bash
# In a separate terminal
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
| `training.use_synthetic` | True | Use synthetic pretraining data |
| `hardware.accelerator` | gpu | `gpu` or `cpu` |

**Training outputs:**
- Checkpoints saved to `checkpoints/` (top-3 by validation loss + `last.ckpt`)
- TensorBoard logs saved to `tb_logs/`

**Monitor training:**
```bash
tensorboard --logdir tb_logs
```

---

Run inference on a trained checkpoint to generate symbolic formulas. The model uses **Beam Sampling** (multiple candidates) and **R² Ranking** to select the best formula.

### From the AI Feynman dataset

```bash
python predict.py --ckpt checkpoints/last.ckpt --id I.6.2a --n_candidates 10 --temperature 0.1
```

### From a custom CSV file

The CSV should have variable columns followed by a target column (last column = output):

```bash
python predict.py --ckpt checkpoints/last.ckpt --csv path/to/your/data.csv --n_candidates 10
```

**Example output:**
```
🎯 Predicting equation: I.6.2a...
Fitting constants for 4 unique candidates...

--- Prediction Result ---
ID:           I.6.2a
Ground Truth: exp(-theta**2/2)/sqrt(2*pi)
Prediction:   exp(-0.5*theta**2)/sqrt(2.0*pi)
R²:           1.000000
NED:          0.0000
SA (Equiv):   True
```

---

## Evaluation

Run the **Goldilocks Evaluation Suite** on the AI Feynman dataset to compute precision, complexity, and robustness metrics:

```bash
python run_eval.py --ckpt checkpoints/last.ckpt --n_candidates 50 --temperature 0.1
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/base_config.yaml` | Config file path |
| `--ckpt` | *(required)* | Checkpoint path |
| `--output_dir` | `results` | Directory to save reports |
| `--n_candidates` | 50 | Number of candidates to sample (Pool size N) |
| `--temperature` | 0.1 | Sampling variance for candidate generation |
| `--mode` | `eval` | `eval` for full suite, `predict` for single equation |

**Metrics computed:**

| Category | Metrics |
|---|---|
| **Standard** | Exact recovery rate, valid RPN rate, dimensional validity |
| **Precision** | R² (Pre/Post BFGS), NED (Normalized Edit Distance), SA (Symbolic Agreement) |
| **Complexity** | Mean formula node count |
| **Robustness** | R² vs noise level (ε ∈ {0.001, 0.01, 0.1}), R² vs data size (N ∈ {10, 50, 100, 200}) |
| **Operational** | Generation latency (ms), BFGS fitting latency (ms) |

Detailed reports are saved to `results/goldilocks_report.md`.

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
│   ├── base_config.yaml       # All hyperparameters
│   └── smoke_test.yaml        # Fast local validation (1 epoch, CPU)
├── data/
│   ├── FeynmanEquations.csv   # Equation metadata
│   ├── Feynman_with_units/    # Raw data files (100 equations)
│   ├── aif_dataset.py         # AIF dataset loader
│   ├── preprocess_aif.py      # Standalone AIF preprocessing CLI
│   ├── synthetic_dataset.py   # Synthetic pretraining data (Lazy loading)
│   ├── generate_data.py       # Standalone generation CLI
│   ├── tokenizer.py           # RPN tokenizer (44 tokens)
│   ├── unit_table.py          # Physical unit lookup
│   └── utils.py               # IEEE-754 Encoding: each float → 16-bit float16 features
├── docs/
│   ├── overview.md            # High-level architecture and design decisions
│   └── technical_reference.md # Detailed module-by-module reference
├── models/
│   ├── model.py               # LLMJEPA unified model
│   ├── encoder.py             # MixEncoder (ISAB + column attention)
│   ├── decoder.py             # RPNDecoder (causal transformer)
│   ├── embedders.py           # Data + Unit embedders
│   ├── target_encoder.py      # Formula → z_target (training only)
│   └── predictor.py           # JEPA predictor (bottleneck cross-attention)
├── training/
│   ├── train.py               # Training entry point
│   ├── trainer.py             # Lightning module
│   └── losses.py              # JEPA + SIGReg + LM + Unit losses
├── inference/
│   └── generate.py            # Autoregressive generation with validity masking
├── evaluation/
│   ├── evaluate.py            # Full evaluation suite (noise, OOD, BFGS)
│   └── metrics.py             # Metric functions (R², Acc_τ, node count)
├── tests/
│   ├── test_data.py           # Tokenizer + synthetic dataset tests
│   ├── test_model.py          # Model forward pass test
│   └── test_pipeline.py       # End-to-end training step test
├── predict.py                 # CLI inference
├── run_eval.py                # CLI evaluation
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