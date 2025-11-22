# Adaptive Sampling Grid Search - Updated Configuration

## Overview

The `grid_search_forward_mu.py` script has been updated to focus on comparing different **adaptive sampling strategies** instead of varying BC weights and neuron counts.

## Key Changes

### Fixed Parameters
- **BC weight:** 10 (fixed)
- **Sampling points:** 5000 (reduced from 10000 for faster testing)
- **Neurons:** [100, 500, 1000] (still varied)

### Sampling Configurations Tested

| Config Name | Blob % | Boundary % | Background % | Replacement | Focus |
|-------------|--------|------------|--------------|-------------|-------|
| `uniform` | 0% | 0% | 100% | N/A | Baseline (random sampling) |
| `adaptive_5_25_70_repl` | 5% | 25% | 70% | Yes | **Boundary-focused** (recommended) |
| `adaptive_10_20_70_repl` | 10% | 20% | 70% | Yes | Balanced |
| `adaptive_20_10_70_repl` | 20% | 10% | 70% | Yes | Blob interior-focused |
| `adaptive_5_25_70_noRepl` | 5% | 25% | 70% | No | Boundary-focused, no duplicates |
| `adaptive_10_20_70_noRepl` | 10% | 20% | 70% | No | Balanced, no duplicates |
| `adaptive_20_10_70_noRepl` | 20% | 10% | 70% | No | Blob-focused, no duplicates |

**Total configurations:** 7 sampling × 3 neurons × 2 mu types = **42 forward solves**

## Hypotheses to Test

### 1. Baseline Performance
**Hypothesis:** Uniform sampling should perform well overall (R² ≈ 0.99) but may struggle in blob regions.

**Expected:**
- Overall R²: 0.99+
- Blob R²: 0.80-0.85 (observed in test_forward_model.py)
- Background R²: 0.99+

### 2. Adaptive with Replacement Hurts Performance
**Hypothesis:** Sampling with replacement creates duplicate points, causing ill-conditioning in the linear solve.

**Expected:**
- `adaptive_*_repl` configs: Blob R² < 0.80
- `adaptive_*_noRepl` configs: Blob R² ≥ 0.80

### 3. Boundary-Focused is Best
**Hypothesis:** Blob **boundaries** (high gradients) are harder to predict than blob **interior** (uniform displacement).

**Expected:**
- `adaptive_5_25_70_*`: Best discrimination between const/hetero mu
- `adaptive_20_10_70_*`: Worst (wastes samples on easy blob interior)

### 4. Loss Function Sensitivity
**Hypothesis:** Some loss functions are more sensitive to stiffness heterogeneity than others.

**Expected (from previous results):**
- Sobolev loss: Highest discrimination (rel_diff > 0.5)
- Correlation loss: Good discrimination
- MSE: Poor discrimination (rel_diff < 0.1)

## Key Metrics

### Discrimination Ability
```python
rel_diff = |Loss(heterogeneous) - Loss(constant)| / Loss(constant)
```

Higher rel_diff = Better at detecting stiffness heterogeneity

### Region-Specific R²
- **Blob R²:** Performance in stiff inclusions (μ > 8 kPa)
- **Background R²:** Performance in soft tissue (μ ≤ 8 kPa)
- **Overall R²:** Combined performance

## Output Files

### CSV: `loss_comparison_results.csv`
Columns:
- `mu_type`: 'constant_5000' or 'heterogeneous'
- `sampling_config`: Name from table above
- `use_adaptive`, `blob_ratio`, `boundary_ratio`, `allow_replacement`
- `neurons`: 100, 500, or 1000
- Loss metrics: `mse`, `relative_l2`, `sobolev`, `correlation_loss`
- Region metrics: `blob_r2`, `blob_mse`, `background_r2`, `background_mse`

### CSV: `loss_discrimination_analysis.csv`
Computed discrimination scores for each config/neuron/loss combination.

## Expected Insights

### If Uniform Wins:
- Adaptive sampling doesn't help the forward problem
- Forward PDE solve benefits from global wave structure sampling
- **Recommendation:** Abandon adaptive sampling for forward model

### If Boundary-Focused (5/25/70) Wins:
- High-gradient regions need more samples
- Blob interior is easy to predict (low variance)
- **Recommendation:** Use 5/25/70 split for inverse problem

### If No-Replacement Wins:
- Duplicate points cause numerical issues
- Need to ensure all samples are unique
- **Recommendation:** Update `allow_replacement=False` in all scripts

## Usage

```bash
# Run grid search (42 configurations, ~5-10 minutes)
python bioqic_pielm/grid_search_forward_mu.py
```

## Interpreting Results

### Best Sampling Strategy:
1. **Highest blob R²** in heterogeneous case
2. **Highest discrimination** (rel_diff) between const/hetero
3. **Lowest blob MSE** in heterogeneous case

### Example Good Result:
```
Config: adaptive_5_25_70_noRepl, Neurons=500
  Const Blob R²:  0.82
  Hetero Blob R²: 0.85
  Sobolev RelDiff: 0.75
```

### Example Bad Result:
```
Config: adaptive_20_10_70_repl, Neurons=500
  Const Blob R²:  0.78 (worse than uniform!)
  Hetero Blob R²: 0.76
  Sobolev RelDiff: 0.12 (poor discrimination)
```

## Next Steps After Results

1. **Identify winner:** Which sampling config has best blob R² + discrimination?
2. **Compare with uniform:** Did any adaptive config beat baseline?
3. **Analyze no-replacement:** Does it consistently outperform with-replacement?
4. **Update inverse problem:** Use best sampling strategy for inverse problem training

---

**Status:** ✓ Grid search script updated and ready to run
**Date:** 2025-01-22
**Expected runtime:** 5-10 minutes on CPU
