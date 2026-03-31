# LLM-JEPA for Symbolic Regression

**Google Summer of Code 2026 Project** | ML4SCI

A Joint Embedding Predictive Architecture (JEPA) for Symbolic Regression. Parses tabular data into IEEE-754 bit-level embeddings and trains a decoder to generate Reverse Polish Notation (RPN) mathematical formulas, validated by unit-dimensional constraints.

**This is the first application of JEPA to symbolic regression.** The model learns representations by predicting one view of data (the equation) from another (the numerical data table) in embedding space, learning the deep structural correspondence between variables and their mathematical relationships.

## Features
- **JEPA Pretraining with SIGReg** — First JEPA-based SR model. Sketched Isotropic Gaussian Regularisation prevents representation collapse without EMA, enabling end-to-end training with both encoders trainable
- **Physics-Informed Architecture** — Unit embeddings encode SI dimensions (mass, length, time, current, temperature) for each variable. Unit prediction head provides dimensional analysis as auxiliary training signal
- **IEEE-754 Bit Encoding** — Direct float16 bit encoding preserves symbolic constants without normalization. No `unpackbits` CPU bottleneck
- **RPN with Validity Masking** — Stack-depth counter enables O(1) grammar-constrained generation. Invalid tokens hard-blocked during inference
- **BFGS Post-processing** — Numerical constants (`c1`...`c5`) fitted with SciPy's L-BFGS-B (100 iterations, 5 restarts). Separates symbolic structure from numerical optimization
- **ODEFormer-Style Inference** — Temperature sampling generates diverse candidate pool (N=50). Skeletons deduplicated, constants fitted, ranked by R²
- **Goldilocks Evaluation** — R² (precision), Symbolic Accuracy (functional equivalence), NED (structural similarity), noise tolerance, data efficiency

---

## 📊 Dataset Documentation

### Synthetic Data Generation (Physics-Informed)

The synthetic data generator creates dimensionally consistent physics equations for pretraining:

- **Dimensional Homogeneity:** Every equation is physically valid. Variables sampled from physics domain pools (mechanics, electromagnetism, thermodynamics). Operators validated via unit propagation rules
- **Pattern-Based Generation (80%):** 55+ physics equation templates covering inverse-square laws (Coulomb, gravity), relativistic equations, energy/wave functions, distance formulas, thermodynamics, electromagnetism
- **AI Feynman Complexity Match:** Mean 3.9-4.1 variables (vs 4.09 in AI Feynman, -5% to 0% gap), mean 12-14 nodes (vs 12.47, -4% to +12% gap), mean depth 2.9-3.0 (vs 2.92, ±0%)
- **Variable Enforcement:** 85% of equations forced to 4+ variables for sufficient complexity
- **Operator Frequency Tuning:** Division-heavy patterns (40%), negation-heavy patterns (30%), addition-rich patterns (23%) to match AI Feynman operator distribution
- **Affine Transformations:** Increases diversity without changing symbolic structure (Kamienny et al. 2022)
- **10% Yield Rate:** Expected given dimensional consistency constraint

**Configuration Guide:**
- **25k equations (~1M params):** Use `configs/small.yaml` (d_model=56, 3 enc/3 dec layers, ~1M params)
- **100k+ equations (~3.4M params):** Use `configs/base_config.yaml` (d_model=256, 4 enc/4 dec layers)

**Generation Command:**
```bash
# Generate 25k equations (small config)
python -m data.generate_data --config configs/small.yaml

# Generate 1M equations (base config)
python -m data.generate_data --config configs/base_config.yaml
```

**Full Documentation:**
- [`docs/SYNTHETIC_DATA_GENERATION.md`](docs/SYNTHETIC_DATA_GENERATION.md) - Physics-informed generation details
- [`docs/DATA_COMPARISON.md`](docs/DATA_COMPARISON.md) - AI Feynman comparison report
- [`docs/SMALL_MODEL_CONFIG.md`](docs/SMALL_MODEL_CONFIG.md) - Model scaling guide

### AI Feynman Comparison

Comprehensive comparison between synthetic pretraining data and AI Feynman evaluation data:

**Latest Results (20k synthetic vs 100 AI Feynman):**

| Metric | Synthetic | AI Feynman | Gap | Quality |
|--------|-----------|------------|-----|---------|
| Mean Variables | 3.36 | 4.09 | -18.0% | ✅ Good |
| Mean Nodes | 11.10 | 12.47 | -11.0% | ✅ Good |
| Mean Depth | 2.87 | 2.92 | -1.7% | ✅ Excellent |
| Division (inv) | 1.13/eq | 1.16/eq | -2.6% | ✅ Excellent |
| Addition | 0.27/eq | 0.66/eq | -59% | ⚠️ Moderate |

**Pattern Coverage:** 55+ physics templates covering inverse-square, relativistic, energy, waves, distance, trigonometric

**Key Insight:** The `/` operator gap is due to SymPy converting `a/b` → `a * b^(-1)` for variable denominators. Combined division (`inv` + `/`) gap is -27.8%.

**Full Report:** [`docs/DATA_COMPARISON.md`](docs/DATA_COMPARISON.md)

**Run your own comparison:**
```bash
python -m data.compare_datasets \
    --config configs/small.yaml \
    --output results/data_comparison/ \
    --n_synthetic 200 \
    --n_aif 200
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

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| [01_generate_synthetic_data.ipynb](notebooks/01_generate_synthetic_data.ipynb) | Generate 1M synthetic equations | T4 GPU |
| [02_train_model.ipynb](notebooks/02_train_model.ipynb) | Pretrain LLM-JEPA model | T4 GPU |
| [03_evaluate_model.ipynb](notebooks/03_evaluate_model.ipynb) | Evaluate on AI Feynman | T4 GPU |

**Quick Start:**
1. Open any notebook in Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run cells in order (dependency checks included)
4. All outputs saved to Google Drive (`SymbolicRegression/` folder)

**Full Guide:** [`notebooks/README.md`](notebooks/README.md)

---

## Installation

```bash
git clone https://github.com/udohchuks/GSOC-LM-JEPA_for_Symbolic_Regression.git
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

## 🎓 GSOC 2026 Proposal

This project was selected for **Google Summer of Code 2026** under ML4SCI.

**Proposal:** [GSoC2026_Proposal_LM-JEPA_SR_v4_updated.txt](GSoC2026_Proposal_LM-JEPA_SR_v4_updated.txt)

**Key Scientific Question:** Does encoding physics knowledge (dimensional analysis, unit constraints) at every level of a JEPA architecture improve symbolic regression over purely data-driven approaches?

**12-Week Plan:**
1. **Weeks 1-2:** Establish baseline on AI Feynman (teacher-forced accuracy, R² on simplest equations)
2. **Weeks 3-4:** Core ablations (no JEPA, no units, EMA vs SIGReg) on 20k dataset
3. **Weeks 5-6:** Scale to 100k-500k equations, 2-4M parameters
4. **Weeks 7-8:** Inference strategy experiments (greedy, beam search, temperature sampling)
5. **Weeks 9-12:** Fine-tuning, final evaluation, documentation

**Expected Contributions:**
- First controlled ablation study isolating JEPA, SIGReg, and dimensional analysis contributions
- Physics-informed synthetic data generator with 55+ templates
- Benchmark results on AI Feynman with reproducible methodology
- Open-source implementation with Colab notebooks for complete workflow



## Data Generation

Large-scale pretraining requires a synthetic corpus of mathematically valid and physically motivated equations. The `data.synthetic_dataset` generator produces these through a **Physics-Informed** pipeline:

- **Dimensional Homogeneity**: Variables are sampled from physical domains (Mechanics, EM, etc.). Operators (like `sin` or `+`) are only applied if they are dimensionally consistent—preventing invalid operations like adding "meters" to "kilograms".
- **Tree & Pattern Generation**: Uses a 80/20 mix of physics pattern templates (55+ templates) and recursive tree growth (peak depth 4) to ensure structural diversity.
- **Complexity Matching**: Biased toward 3–6 variables and tree depths of 4–5 to match the complexity distribution of the AI Feynman dataset.
- **Float16 Bit-Featurization**: Inputs are encoded into IEEE-754 bit patterns and stored as direct `float16` tensors, removing CPU-bound `unpackbits` overhead during training.
- **Lazy Sharding**: Equations are stored in sharded `.pt` parts, allowing for 1M+ scales without RAM exhaustion via `LazySyntheticDataset`.

### 1. Generate Synthetic Corpus

Generate the pretraining data before training:

```bash
# Generate 25k equations (for ~1M param model)
python -m data.generate_data --config configs/small.yaml

# Generate 1M equations (for ~3.4M param model)
python -m data.generate_data --config configs/base_config.yaml
```

### 2. Preprocess AIF Dataset (Optional)
Precompute the evaluation dataset to ensure instant training startup:

```bash
python -m data.preprocess_aif --config configs/small.yaml
```

### 3. Start Training
The training script loads existing cache parts and automatically detects new parts as they are generated:

```bash
# Train with synthetic data
python -m training.train --config configs/small.yaml
```

**Key config options** (`configs/small.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `training.max_epochs` | 15 | Training epochs |
| `training.lr` | 5e-4 | Learning rate |
| `model.d_model` | 56 | Hidden dimension (~1M params) |
| `model.n_enc_layers` | 3 | Encoder transformer layers |
| `model.n_dec_layers` | 3 | Decoder transformer layers |
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

## Inference

Run inference on a trained checkpoint to generate symbolic formulas. The model uses **ODEFormer-Style Sampling & Ranking** with BFGS constant fitting.

### From the AI Feynman dataset

```bash
python predict.py --ckpt checkpoints/last.ckpt --id I.6.2a --n_candidates 10 --temperature 0.1
```

### From a custom CSV file

The CSV should have variable columns followed by a target column (last column = output):

```bash
python predict.py --ckpt checkpoints/last.ckpt --csv path/to/your/data.csv --n_candidates 10
```

**Inference Pipeline:**
1. **Encode:** Input data table → z_context via MixEncoder
2. **Sample:** Generate N candidate formulas (default 50) using temperature-controlled sampling
3. **Skeletonize:** Replace constants (c1, c2, ...) with placeholders for deduplication
4. **Deduplicate:** Keep only structurally unique skeletons
5. **Fit Constants:** L-BFGS-B optimization (100 iterations, 5 restarts) on FULL dataset
6. **Rank:** Sort candidates by R² score, return top-k

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

**Inference Configuration** (`configs/small.yaml`):
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

Run the **Goldilocks Evaluation Suite** on the AI Feynman dataset to compute precision, complexity, and robustness metrics:

```bash
python run_eval.py --ckpt checkpoints/last.ckpt --n_candidates 50 --temperature 0.1
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/small.yaml` | Config file path |
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

**Evaluation Pipeline:**
1. **Load Checkpoint:** Restore model weights from `.ckpt` file
2. **Iterate AIF:** Loop through all 100 Feynman equations
3. **Generate Candidates:** Sample N=50 formulas per equation
4. **Fit & Score:** BFGS optimization on full dataset (100k points)
5. **Compute Metrics:** R², Symbolic Accuracy, NED, Constant Recovery
6. **Stress Tests:** Noise tolerance, data efficiency, OOD extrapolation
7. **Generate Report:** Markdown report with per-equation breakdown

Detailed reports are saved to `results/goldilocks_report.md`.

**Stress Tests:**
- **Noise Tolerance:** Evaluate at Gaussian noise levels ε ∈ {0.001, 0.01, 0.1}
- **Data Efficiency:** Evaluate at N ∈ {10, 50, 100, 200} input points
- **OOD Extrapolation:** Evaluate on data sampled from 10x original variable ranges

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
GSOC-LM-JEPA_for_Symbolic_Regression/
│
├── configs/
│   ├── base_config.yaml       # ~3.4M params, 1M equations
│   ├── small.yaml             # ~1M params, 25k equations (GSoC baseline)
│   └── smoke_test.yaml        # Fast local validation (1 epoch, CPU)
├── data/
│   ├── FeynmanEquations.csv   # Equation metadata (100 equations)
│   ├── Feynman_with_units/    # Raw data files (100 equations, 4.1 GB)
│   ├── aif_dataset.py         # AI Feynman dataset loader (evaluation)
│   ├── preprocess_aif.py      # Standalone AIF preprocessing CLI
│   ├── synthetic_dataset.py   # Physics-informed synthetic data (Lazy loading)
│   ├── generate_data.py       # Standalone generation CLI
│   ├── tokenizer.py           # RPN tokenizer (44 tokens, validity mask)
│   ├── unit_table.py          # SI unit lookup (5 dimensions, class indices)
│   └── utils.py               # IEEE-754 encoding (uint8 optimized), noise
├── docs/
│   ├── overview.md            # High-level architecture and design decisions
│   ├── technical_reference.md # Detailed module-by-module reference
│   ├── SYNTHETIC_DATA_GENERATION.md  # Physics-informed generation details
│   ├── DATA_COMPARISON.md     # AI Feynman comparison analysis report
│   ├── SMALL_MODEL_CONFIG.md  # Model scaling guide for <100k equations
│   └── GPU_REDLINE_MODE.md    # GPU performance optimization guide
├── models/
│   ├── model.py               # LLMJEPA unified model
│   ├── encoder.py             # MixEncoder (ISAB + column attention)
│   ├── decoder.py             # RPNDecoder (causal transformer)
│   ├── embedders.py           # Data + Unit embedders
│   ├── target_encoder.py      # Formula → z_target (training only)
│   └── predictor.py           # JEPA predictor (bottleneck cross-attention)
├── training/
│   ├── train.py               # Training entry point
│   ├── trainer.py             # Lightning module (LLMJEPAModule)
│   └── losses.py              # JEPA + SIGReg + LM + Unit losses
├── inference/
│   └── generate.py            # Autoregressive generation with validity masking
├── evaluation/
│   ├── evaluate.py            # Full evaluation suite (noise, OOD, BFGS)
│   └── metrics.py             # Metric functions (R², Acc_τ, node count)
├── tests/
│   ├── test_data.py           # Tokenizer + synthetic dataset tests
│   ├── test_metrics.py        # Metrics tests (R², NED, SA)
│   ├── test_inference.py      # Inference tests (skeletonize, BFGS)
│   └── test_pipeline.py       # End-to-end training step test
├── notebooks/
│   ├── 01_generate_synthetic_data.ipynb  # Generate 25k-1M equations
│   ├── 02_train_model.ipynb              # Pretrain LLM-JEPA model
│   └── 03_evaluate_model.ipynb           # Evaluate on AI Feynman benchmark
├── predict.py                 # CLI inference (ODEFormer-style)
├── run_eval.py                # CLI evaluation (Goldilocks suite)
├── GSoC2026_Proposal_LM-JEPA_SR_v4_updated.txt  # GSOC 2026 proposal
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