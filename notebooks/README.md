# Google Colab Notebooks

This folder contains 3 Colab notebooks for the complete LLM-JEPA workflow:

| Notebook | Purpose | Recommended For |
|----------|---------|----------------|
| [`01_generate_synthetic_data.ipynb`](01_generate_synthetic_data.ipynb) | Generate synthetic equations | 20k-1M equations |
| [`02_train_model.ipynb`](02_train_model.ipynb) | Pretrain LLM-JEPA model | `configs/small.yaml` (~1M params) |
| [`03_evaluate_model.ipynb`](03_evaluate_model.ipynb) | Evaluate on AI Feynman | Any trained checkpoint |

**Default settings in notebooks:**
- Generate: 20k equations (for ~1M model)
- Train: `configs/small.yaml` (~1M params, 15 epochs)
- Evaluate: Auto-finds latest checkpoint

---

## Quick Start

### ⚠️ Important: Run Notebooks in Order

**The notebooks have dependencies - run them in this exact order:**

```
1️⃣ 01_generate_synthetic_data.ipynb  ← Run FIRST (generates training data)
   ↓
   │ Default: 25k equations
   ↓
2️⃣ 02_train_model.ipynb              ← Run SECOND (needs synthetic data)
   ↓
   │ Default: configs/small.yaml, 15 epochs
   ↓
3️⃣ 03_evaluate_model.ipynb           ← Run THIRD (needs trained checkpoint)
   ↓
   │ Evaluates on all 100 AI Feynman equations
```

**❗ Do NOT skip steps:**
- Training will **fail** without synthetic data
- Evaluation will **fail** without trained checkpoint

---

### How to Use

1. **Open in Colab**

Click any notebook → "Open in Colab" button, or:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click "GitHub" tab
3. Enter: `chukwueke/GSOC-LM-JEPA_for_Symbolic_Regression`
4. Select notebook from `notebooks/` folder

### 2. Connect to Google Drive

Each notebook will:
1. Mount your Google Drive
2. Create `SymbolicRegression` folder
3. Sync code to `SymbolicRegression/code/`
4. Save all outputs to Drive

### 3. Run Cells in Order

Each notebook has cells organized as:
1. 📦 **Setup** - Clone repo, install dependencies
2. ⚙️ **Configuration** - Set parameters
3. 🚀 **Run** - Execute main task
4. 📊 **Results** - View/verify outputs

---

## Folder Structure (in Your Drive)

```
SymbolicRegression/
├── cache/              # Preprocessed data
│   ├── aif_preprocessed.pt
│   └── synthetic_small/   # Generated synthetic equations (configs/small.yaml)
├── checkpoints/        # Training checkpoints
│   ├── last.ckpt
│   └── jepa-step=XXXXX.ckpt
├── code/               # Synced repository code
│   └── (full repo)
├── Feynman_with_units/ # AI Feynman dataset
│   ├── I.6.2a
│   ├── I.12.1
│   └── ...
├── results/            # Evaluation results
│   └── eval_YYYYMMDD_HHMMSS/
└── tb_logs/            # TensorBoard logs
    └── llmjepa_sr_base/
```

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW OVERVIEW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1️⃣  GENERATE                                                    │
│      Notebook: 01_generate_synthetic_data.ipynb                 │
│      Output: SymbolicRegression/cache/synthetic_small/            │
│                                                                  │
│      ⏳ Wait for completion before proceeding!                   │
│                                                                  │
│      ↓                                                          │
│                                                                  │
│  2️⃣  TRAIN                                                      │
│      Notebook: 02_train_model.ipynb                             │
│      Output: SymbolicRegression/checkpoints/                    │
│                                                                  │
│      ⏳ Wait for completion before proceeding!                   │
│                                                                  │
│      ↓                                                          │
│                                                                  │
│  3️⃣  EVALUATE                                                   │
│      Notebook: 03_evaluate_model.ipynb                          │
│      Output: SymbolicRegression/results/                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Each step depends on the previous one completing
- All outputs saved to Google Drive (persist across sessions)
- Can resume interrupted steps (notebooks support resume)

---

## Notebook Details

### 01_generate_synthetic_data.ipynb

**Purpose:** Generate 1M physics-informed synthetic equations for pretraining.

**Key Parameters:**
- `N_SYNTHETIC`: Number of equations (default: 1,000,000)
- `N_DATA_POINTS`: Data rows per equation (default: 2,000)
- `SYNTHETIC_SUBFOLDER`: Output folder name

**Output:** `SymbolicRegression/cache/synthetic_small/`

**Tips:**
- Data saved incrementally (1000 equations per file)
- Can resume if interrupted

---

### 02_train_model.ipynb

**⚠️ Prerequisite:** Must complete `01_generate_synthetic_data.ipynb` first!

**Purpose:** Train LLM-JEPA model on synthetic data.

**Default Configuration:** `configs/small.yaml`
- ~1M parameters (71% smaller than base)
- Tiny predictor (2.5K params, 0.2% of total)
- 15 epochs, LR 5e-4, batch 64

**Key Parameters:**
- `CONFIG_FILE`: Config to use (default: `configs/small.yaml`)
- `MAX_EPOCHS`: Training epochs (default: 15)
- `BATCH_SIZE`: Batch size (default: 64)
- `LEARNING_RATE`: Learning rate (default: 5e-4)
- `USE_SYNTHETIC`: Use synthetic pretraining data (default: True)

**Features:**
- TensorBoard in separate cell (run during or after training)
- Checkpoint saving every 200 steps
- Automatic resume from last checkpoint

**Output:**
- Checkpoints: `SymbolicRegression/checkpoints/`
- Logs: `SymbolicRegression/tb_logs/`

**Tips:**
- Run TensorBoard cell to monitor in real-time
- Can open TensorBoard even after training completes
- Stop early if validation loss plateaus
- Keep runtime alive for long training

---

### 03_evaluate_model.ipynb

**Purpose:** Evaluate trained model on AI Feynman benchmark.

**Key Parameters:**
- `CHECKPOINT_PATH`: Model checkpoint (auto-finds latest)
- `N_CANDIDATES`: Beam sampling candidates (default: 10)
- `TEMPERATURE`: Sampling temperature (default: 0.8)

**Metrics Computed:**
- Exact recovery rate
- Mean R² (pre/post BFGS)
- Valid RPN rate
- Dimensional validity
- Noise tolerance
- Data efficiency
- Extrapolation

**Output:** `SymbolicRegression/results/eval_YYYYMMDD_HHMMSS/`

**Tips:**
- Higher N_CANDIDATES = better recovery but slower
- Test single equations before full evaluation
- Results include JSON + Markdown report

---

## Common Issues

### "Synthetic data NOT found" Error

**Problem:** Training notebook says "Synthetic data NOT found!"

**Solution:**
1. Stop training
2. Open `01_generate_synthetic_data.ipynb`
3. Run all cells to generate synthetic data
4. Wait for generation to complete (check for `.pt` files in `SymbolicRegression/cache/synthetic_small/`)
5. Return to training notebook and re-run

**Why:** Training requires synthetic data to be generated first. The notebook checks for this automatically.

### Runtime Disconnected

**Problem:** Colab disconnects during long runs

**Solutions:**
1. Use "Keep runtime alive" extension (use at own risk)
2. Connect to local runtime instead
3. Run in shorter sessions (generation supports resume)

### Out of Memory

**Problem:** GPU runs out of memory

**Solutions:**
1. Reduce `BATCH_SIZE` in training notebook
2. Use fewer `N_CANDIDATES` in evaluation
3. Restart runtime and try again

### Drive Not Found

**Problem:** Google Drive not mounting

**Solutions:**
1. Re-run the mount cell
2. Manually mount: `drive.mount('/content/drive')`
3. Check Drive permissions

### Download Failed

**Problem:** AI Feynman dataset download fails

**Solutions:**
1. Download manually from repository
2. Upload to `SymbolicRegression/Feynman_with_units/`
3. Re-run the check cell

---

## Cost Estimates (Google Colab)

| Notebook | Colab Free | Colab Pro | Colab Pro+ |
|----------|-----------|-----------|------------|
| Generate | ✅ Works | ✅ Faster | ✅ Fastest |
| Train | ✅ Works | ✅ Better | ✅ Best |
| Evaluate | ✅ Works | ✅ Faster | ✅ Fastest |

**Note:** Colab Pro provides longer runtime sessions, which is beneficial for the complete workflow.

---

## Local Alternative

If you have a local GPU:

1. Clone repository locally
2. Set up environment (see main README)
3. Run commands directly:
   ```bash
   # Generate
   python -m data.generate_data --config configs/base_config.yaml
   
   # Train
   python -m training.train --config configs/base_config.yaml
   
   # Evaluate
   python -m run_eval --ckpt checkpoints/last.ckpt
   ```

---

## Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Documentation:** See `docs/` folder

---

*Last updated: 2026-03-29*
