# Documentation Update Summary

**Date:** March 30, 2026  
**Version:** v4 (GSoC 2026 Baseline)

---

## Files Updated

### 1. README.md
**Changes:**
- Added GSOC 2026 header and project description
- Updated Features section with JEPA/SIGReg emphasis
- Added GSOC 2026 Proposal section with 12-week plan
- Updated synthetic data generation stats (55+ templates, 80% pattern-based)
- Updated AI Feynman comparison table with latest metrics
- Added inference pipeline steps (6-step process)
- Enhanced evaluation section with stress tests description
- Updated project structure to include notebooks/ and GSOC proposal file
- Changed GitHub URL to `udohchuks/GSOC-LM-JEPA_for_Symbolic_Regression`

### 2. docs/overview.md
**Changes:**
- Added GSOC 2026 header
- Added "Open Question" and "Related Work" to The Problem section
- Renamed "The Approach" to "Physics-Informed LM-JEPA" with 3 numbered components
- Added SIGReg explanation with Epps-Pulley statistic details
- Updated Key Design Decisions table:
  - ISAB inducing points: 32 → 20 (matches small.yaml)
  - Added configs/small.yaml reference
  - Updated chunk size range: 2000 → 100-2000 equations/part
- Updated Training section:
  - Added 25k equations command (GSoC baseline)
  - Updated chunk_size: 2000 → 100 (matches small.yaml)
- Updated all config references: base_config.yaml → configs/small.yaml (where appropriate)

### 3. docs/technical_reference.md
**Changes:**
- Added "Important Fix (v4)" note to ValidityWeightedCE section explaining:
  - Why cumsum without shift is correct
  - Example with input_ids=[BOS,x1,x2,+] and targets=[x1,x2,+,EOS]
  - Mathematical justification for stack depth calculation

### 4. Code Fixes Applied

#### data/synthetic_dataset.py
**Bug Fix:** Resume logic in `_generate_corpus`
- **Problem:** When resuming generation, the break condition incorrectly compared new equations against total target
- **Fix:** Added `n_equations_target` to track original target, corrected break condition to `len(equations) >= n_equations`
- **Impact:** Generation now correctly completes when targeting 25,500 equations with resume

#### training/losses.py
**Bug Fix:** RPN stack depth calculation in `_get_batch_depths`
- **Problem:** Shifted cumsum caused valid tokens to be incorrectly penalized
- **Fix:** Removed shift, use `cumsum(deltas)` directly
- **Impact:** EOS and binary operators now validated correctly at inference

#### data/aif_dataset.py
**Improvement:** Enhanced error message for missing Feynman data files
- **Added:** Clear instruction to run "Download AI Feynman Dataset" cell in training notebook

---

## Key Technical Corrections

### 1. RPN Stack Depth Calculation (CRITICAL)
**Before:**
```python
shifted_deltas = torch.cat([zeros, deltas[:, :-1]], dim=1)
depths = torch.cumsum(shifted_deltas, dim=1)
```

**After:**
```python
depths = torch.cumsum(deltas, dim=1)
```

**Why:** At position `t`, the model has seen `input_ids[:t+1]` via causal attention. The validity of `targets[t]` depends on the stack state **after** processing all tokens up to and including `input_ids[t]`. The shift was using stale context (stack before `input_ids[t]`), causing:
- EOS at t=3: depth=2 (invalid, should be 1) ✗
- Binary + at t=2: depth=1 (invalid, should be ≥2) ✗

### 2. Synthetic Data Resume Logic
**Before:**
```python
n_equations = needed  # Remaining to generate
# ... generate loop ...
if (len(equations) + chunk_count * chunk_size) >= n_equations:
    break
```

**After:**
```python
n_equations_target = n_equations  # Original target
n_equations = needed  # Remaining to generate
# ... generate loop ...
if len(equations) >= n_equations:
    break
```

**Why:** The break condition was comparing new equations against the remaining target, but adding existing file count. This caused early termination after generating just 1 equation when resuming.

---

## Configuration Changes

### configs/small.yaml (GSoC 2026 Baseline)
- **d_model:** 56 (~1M parameters)
- **n_enc_layers:** 3
- **n_dec_layers:** 3
- **n_synthetic:** 25,000 (default)
- **chunk_size:** 100 equations/file
- **max_epochs:** 15
- **lr:** 5e-4

### configs/base_config.yaml (Full Scale)
- **d_model:** 256 (~3.4M parameters)
- **n_enc_layers:** 4
- **n_dec_layers:** 4
- **n_synthetic:** 1,000,000 (default)
- **chunk_size:** 2000 equations/file
- **max_epochs:** 30
- **lr:** 3e-4

---

## Documentation Best Practices Applied

1. **Consistent Config References:** Use `configs/small.yaml` for GSoC baseline examples, `configs/base_config.yaml` for full-scale
2. **Explicit Examples:** All commands include both small and base config variants
3. **Mathematical Justification:** Technical fixes include worked examples showing why the fix is correct
4. **Clear Error Messages:** User-facing errors now include actionable fix instructions
5. **GSOC Context:** All documentation now references the GSoC 2026 project framework

---

## Remaining TODOs

1. **Training Results Section:** Add actual training curves and Feynman evaluation numbers once available
2. **Ablation Study Results:** Document findings from Phase 2 ablations (no JEPA, no units, EMA vs SIGReg)
3. **Performance Benchmarks:** Add GPU utilization, throughput numbers for different scales
4. **Inference Comparison:** Document greedy vs beam search vs temperature sampling results

---

## References

- **GSOC Proposal:** `GSoC2026_Proposal_LM-JEPA_SR_v4_updated.txt`
- **Synthetic Data Docs:** `docs/SYNTHETIC_DATA_GENERATION.md`
- **AI Feynman Comparison:** `docs/DATA_COMPARISON.md`
- **Model Scaling Guide:** `docs/SMALL_MODEL_CONFIG.md`
