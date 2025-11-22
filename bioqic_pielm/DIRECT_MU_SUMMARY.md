# Direct Mu Optimization - Implementation Summary

## Overview

Successfully implemented **direct gradient-based μ optimization** for MRE inverse problems, replacing the neural network approach with classical adjoint method.

## Key Components

### 1. Direct Parameterization ([direct_mu_model.py](direct_mu_model.py))
- **μ is a learnable parameter** (N, 1) instead of neural network output
- Initialization modes: constant, random, uniform
- Forward solver unchanged (PIELM)
- Gradients flow directly to μ values

### 2. Training with PDE Loss ([direct_mu_trainer.py](direct_mu_trainer.py))
- **PDE Residual Loss**: ∇·(μ∇u) + ρω²u = 0
- **Normalization**: Divides by ρω² to prevent scale mismatch
- **Multi-term loss**: Sobolev + PDE + TV + Prior
- Comprehensive visualizations (12 plots)

### 3. Visualizations
**Progress plots** (saved every 500 iterations):
- Row 1: True μ map, Predicted μ map, μ histogram
- Row 2: μ scatter, Displacement fit, Gradient norm
- Row 3: Data/PDE/TV losses, Mu MSE, μ range evolution

**Final results** (saved at end):
- Row 1: True μ map, Predicted μ map, Error map
- Row 2: μ histogram, μ scatter, Error histogram
- Row 3: Displacement scatter, Displacement error, Displacement map
- Row 4: All losses, Mu MSE, Gradient norm

## Results Summary

### Test Run (5000 points, 1000 iterations)
```
Initial: mu=[5000, 5000] Pa, MuMSE=5.19M
Final:   mu=[3000, 5846] Pa, MuMSE=4.49M

✅ Mu MSE DECREASED (13.5% improvement)
✅ PDE loss normalized (~1e-4, comparable to Sobolev)
✅ Gradients stable (~4e-8)
⚠️  Range limited (only reached 5846 of 10000 Pa max)
```

### Loss Evolution
- **Sobolev**: 4.38e-3 → 2.60e-3 (40% reduction)
- **PDE**: 1.07e-3 → 4.49e-4 (58% reduction)
- **Mu MSE**: 5.19M → 4.49M (13% reduction)

## Key Insights

### 1. PDE Loss Normalization (CRITICAL!)
**Problem**: Raw PDE residual `μ∇²u + ρω²u` has magnitude ~10^13 due to large ρω² = 142M

**Solution**: Normalize by dividing by ρω² before squaring:
```python
residual_normalized = (mu / rho_omega2) * laplacian_u + u
loss_pde = torch.mean(residual_normalized ** 2)
```

This reduces PDE loss from ~10^13 to ~10^-4, preventing it from overwhelming other losses.

### 2. Gradient Magnitude
- **Without PDE**: ~1e-9 (too weak!)
- **With normalized PDE**: ~4e-8 (better, but still small)
- Suggests need for higher learning rate or stronger PDE weight

### 3. Optimization Behavior
- μ correctly moves toward bounds [3000, 10000]
- Lower bound (3000) reached quickly
- Upper bound (10000) not yet explored
- Need more iterations or higher LR

## Comparison: NN vs Direct Mu

| Aspect | Neural Network | Direct Mu |
|--------|---------------|-----------|
| **Parameters** | ~10K (NN weights) | 5000 (one per point) |
| **Update Target** | θ (weights) | μ (stiffness) |
| **Gradients** | ~1e-9 (weak) | ~4e-8 (stronger with PDE) |
| **Learning Rate** | 0.001-0.005 | 10-1000 |
| **Smoothness** | Implicit (architecture) | Explicit (TV/prior) |
| **Mu MSE Change** | Increasing | **Decreasing** ✅ |

## Recommended Parameters

### For Full Training (5000 points):
```bash
python bioqic_pielm/train_direct_mu.py \
  --subsample 5000 \
  --iterations 5000 \
  --loss_type sobolev \
  --lr 50.0 \
  --bc_weight 10.0 \
  --tv_weight 0.001 \
  --pde_weight 5.0
```

### For Quick Testing (2000 points):
```bash
python bioqic_pielm/train_direct_mu.py \
  --subsample 2000 \
  --iterations 2000 \
  --loss_type sobolev \
  --lr 50.0 \
  --bc_weight 10.0 \
  --pde_weight 5.0
```

### Parameter Rationale:
- **lr=50.0**: 5x higher to compensate for small gradients (~4e-8)
- **pde_weight=5.0**: Strengthen physics constraint
- **tv_weight=0.001**: Light smoothing (optional)
- **bc_weight=10.0**: Lower than NN approach (allows μ exploration)

## Next Steps

1. **Increase PDE weight** to 5.0-10.0 for stronger physics enforcement
2. **Increase learning rate** to 50-100 to explore full μ range
3. **Longer training** (5000 iterations) to reach true bounds
4. **Compare with NN results** from `outputs/physical_sobolev/`

## Files Created

- [`direct_mu_model.py`](direct_mu_model.py) - Direct μ parameterization
- [`direct_mu_trainer.py`](direct_mu_trainer.py) - Training loop with PDE loss
- [`train_direct_mu.py`](train_direct_mu.py) - Training script
- [`DIRECT_MU_README.md`](DIRECT_MU_README.md) - Usage guide
- [`DIRECT_MU_SUMMARY.md`](DIRECT_MU_SUMMARY.md) - This file

## Mathematical Details

### PDE Residual Gradient (Normalized)
```
Original PDE: ∇·(μ∇u) + ρω²u = 0

Normalized residual: r = (μ/ρω²)∇²u + u

Loss: L_PDE = ||r||² = ||(μ/ρω²)∇²u + u||²

Gradient: ∂L_PDE/∂μ = 2r · (∇²u/ρω²)
                    = 2[(μ/ρω²)∇²u + u] · (∇²u/ρω²)
```

This provides **direct physics-based gradients** that guide μ toward values satisfying the PDE.

## Conclusion

The direct μ optimization approach is **working correctly**:
- ✅ Mu MSE decreasing (not increasing!)
- ✅ PDE loss properly normalized
- ✅ Stable gradients
- ✅ Moving toward correct bounds

With increased LR and PDE weight, this should achieve better reconstruction than the NN approach.
