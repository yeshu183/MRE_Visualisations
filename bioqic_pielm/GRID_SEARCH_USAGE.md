# Grid Search Usage Guide

## Overview

Updated grid search to compare **adaptive sampling strategies** instead of varying BC weights.

## Quick Start

```bash
# Run simplified grid search (recommended)
python bioqic_pielm/grid_search_forward_mu_simple.py
```

This will test 42 configurations (7 sampling strategies × 3 neuron counts × 2 mu types).

**Runtime:** ~5-10 minutes on CPU

## What It Tests

### Fixed Parameters
- **BC weight:** 10
- **Sampling points:** 5000
- **Omega basis:** 170.0

### Variable Parameters
1. **Sampling configurations** (7 total):
   - `uniform`: Baseline random sampling
   - `adaptive_5_25_70_repl`: 5% blob, 25% boundary, 70% background (with replacement)
   - `adaptive_10_20_70_repl`: 10% blob, 20% boundary, 70% background (with replacement)
   - `adaptive_20_10_70_repl`: 20% blob, 10% boundary, 70% background (with replacement)
   - `adaptive_5_25_70_noRepl`: Same as above but caps samples at available points (no duplicates)
   - `adaptive_10_20_70_noRepl`: Same as above but no duplicates
   - `adaptive_20_10_70_noRepl`: Same as above but no duplicates

2. **Neuron counts:** [100, 500, 1000]

3. **Mu types:** Constant (5000 Pa) vs Heterogeneous (3-10 kPa)

## Output Files

### 1. `sampling_comparison_results.csv`
Full results for all 42 configurations.

**Columns:**
- `mu_type`: 'constant_5000' or 'heterogeneous'
- `sampling_config`: Name of sampling strategy
- `use_adaptive`, `blob_ratio`, `boundary_ratio`, `allow_replacement`
- `neurons`: 100, 500, or 1000
- `mse`, `r2`: Overall metrics
- `blob_r2`, `blob_mse`: Blob region metrics
- `background_r2`, `background_mse`: Background region metrics

### 2. `discrimination_summary.csv`
Computed discrimination scores (how well each config distinguishes const vs hetero).

**Columns:**
- `sampling_config`, `neurons`
- `const_r2`, `hetero_r2`, `r2_diff`: Overall R² comparison
- `const_blob_r2`, `hetero_blob_r2`, `blob_r2_diff`: Blob R² comparison

## Key Metrics

### 1. Blob R² (Heterogeneous Case)
**Higher is better**

Measures how well the forward model predicts displacement in stiff blob regions (μ > 8 kPa).

**From test_forward_model.py results:**
- Uniform sampling: **Blob R² = 0.8163**
- Adaptive 20/10/70: **Blob R² = 0.7840** ❌ (worse!)

### 2. Discrimination (R² Difference)
**Higher is better**

Measures how differently the model performs on constant vs heterogeneous stiffness.

```
r2_diff = |R²(heterogeneous) - R²(constant)|
blob_r2_diff = |Blob R²(heterogeneous) - Blob R²(constant)|
```

High discrimination = model is sensitive to stiffness variations.

## Interpreting Results

### What to Look For:

1. **Best Blob R²:**
   ```
   Best Blob R² (heterogeneous):
     Config: uniform
     Neurons: 500
     Blob R²: 0.8163
     Overall R²: 0.9921
   ```
   → This config gives most accurate predictions in blob regions

2. **Best Discrimination:**
   ```
   Best Discrimination (blob R² difference):
     Config: adaptive_5_25_70_noRepl
     Neurons: 1000
     Difference: 0.0523
   ```
   → This config is most sensitive to stiffness heterogeneity

### Expected Outcomes:

#### Scenario 1: Uniform Wins
If `uniform` has highest blob R²:
- **Conclusion:** Adaptive sampling hurts forward problem
- **Reason:** Duplicate points cause ill-conditioning
- **Action:** Use uniform sampling for forward model

#### Scenario 2: No-Replacement Wins
If `adaptive_*_noRepl` > `adaptive_*_repl`:
- **Conclusion:** Sampling with replacement is the problem
- **Reason:** Duplicate points degrade linear solve quality
- **Action:** Always use `allow_replacement=False`

#### Scenario 3: Boundary-Focused (5/25/70) Wins
If `adaptive_5_25_70_*` has best blob R² + discrimination:
- **Conclusion:** Boundaries need more samples than blob interior
- **Reason:** High gradients at interfaces are harder to capture
- **Action:** Use 5/25/70 split for inverse problem

## Console Output Example

```
[21/42] adaptive_5_25_70_noRepl, neurons=500, mu=heterogeneous
  Overall R²: 0.9925, Blob R²: 0.8245

DISCRIMINATION ANALYSIS
==========================================

adaptive_5_25_70_noRepl, neurons=500:
  Overall R²:  Const=0.9921, Hetero=0.9925, Diff=0.0004
  Blob R²:     Const=0.8103, Hetero=0.8245, Diff=0.0142

BEST CONFIGURATIONS
==========================================

Best Blob R² (heterogeneous):
  Config: adaptive_5_25_70_noRepl
  Neurons: 500
  Blob R²: 0.8245
  Overall R²: 0.9925

Best Discrimination (blob R² difference):
  Config: adaptive_5_25_70_noRepl
  Neurons: 500
  Difference: 0.0142
```

## Next Steps After Results

1. **Load and analyze CSV:**
   ```python
   import pandas as pd
   df = pd.read_csv('bioqic_pielm/outputs/sampling_comparison/sampling_comparison_results.csv')

   # Best blob R² in heterogeneous case
   hetero = df[df['mu_type'] == 'heterogeneous']
   best = hetero.loc[hetero['blob_r2'].idxmax()]
   print(f"Best: {best['sampling_config']} with blob R² = {best['blob_r2']:.4f}")
   ```

2. **Compare with test_forward_model.py results:**
   - Uniform achieved Blob R² = 0.8163
   - Did any adaptive config beat this?

3. **Update inverse problem code** if adaptive sampling proves beneficial

## Troubleshooting

### Error: "allow_replacement is not defined"
**Solution:** Make sure you're running `grid_search_forward_mu_simple.py`, not the old `grid_search_forward_mu.py`

### Warning: "Sampling with replacement"
**Expected** for configs ending in `_repl`. Not expected for `_noRepl` configs.

### Very low blob R² (< 0.7)
**Problem:** Adaptive sampling with replacement is hurting performance
**Check:** Compare with uniform baseline

---

**Created:** 2025-01-22
**Purpose:** Test if adaptive sampling helps forward problem distinguish heterogeneous stiffness
