# Direct Mu Parameterization for MRE Inverse Problem

## Overview

This implementation provides a **classical gradient-based inverse solver** that directly optimizes stiffness values μ(x) without using a neural network.

## Key Difference from Neural Network Approach

### Neural Network Approach (Original)
```
μ(x) = f_θ(x)  (neural network with weights θ)
Optimization: θ^(k+1) = θ^k - α ∇_θ J
```

### Direct Mu Approach (New)
```
μ(x) = μ  (direct parameterization, one value per point)
Optimization: μ^(k+1) = μ^k - α ∇_μ J
```

## Algorithm (Classical Adjoint Method)

```
Initialize: μ⁰ (constant, random, or uniform)
For k = 0, 1, 2, ...
    1. Forward solve: Solve PDE to get u^k from μ^k
       ∇·(μ∇u) + ρω²u = 0

    2. Compute objective: J^k = ||u^k - u_meas||²

    3. Adjoint gradient: Compute ∇_μ J via PyTorch autograd
       (gradients flow through PIELM solver to μ directly)

    4. Update: μ^(k+1) = μ^k - α^k ∇_μ J^k

    5. Check convergence
```

## Files

- **`direct_mu_model.py`**: Direct μ parameterization model (replaces StiffnessNetwork)
- **`direct_mu_trainer.py`**: Trainer for direct optimization
- **`train_direct_mu.py`**: Training script

## Usage

### Basic Training
```bash
python train_direct_mu.py
```

### With Options
```bash
# MSE loss
python train_direct_mu.py --loss_type mse --lr 10.0

# Sobolev loss (gradient-enhanced)
python train_direct_mu.py --loss_type sobolev --lr 15.0

# With TV regularization
python train_direct_mu.py --tv_weight 0.01

# Different initialization
python train_direct_mu.py --init_mode random --lr 20.0

# Quick test (fewer points)
python train_direct_mu.py --subsample 2000 --iterations 2000
```

### Arguments
- `--loss_type`: Loss function (`mse`, `correlation`, `relative_l2`, `sobolev`)
- `--lr`: Learning rate (typically 10-20, higher than NN-based)
- `--iterations`: Number of iterations (default: 5000)
- `--bc_weight`: BC enforcement weight (default: 1000.0)
- `--tv_weight`: TV regularization (default: 0.0)
- `--init_mode`: Initialization (`constant`, `random`, `uniform`)
- `--mu_init`: Initial μ value for constant init (default: 5000.0 Pa)
- `--subsample`: Number of points to use (default: 5000)

## Visualizations

The trainer saves the following plots:

### Final Results (`final_results.png`)
1. **Predicted vs True Stiffness**: Scatter plot showing reconstruction accuracy
2. **Stiffness Distribution**: Histogram comparing predicted and true μ
3. **Displacement Fit**: Scatter plot of u_pred vs u_meas
4. **Data Loss & TV vs Iteration**: Evolution of data loss and TV regularization
5. **Mu MSE vs Iteration**: Stiffness reconstruction error over time (SEPARATE)
6. **Gradient Norm**: ∇_μ J norm over iterations

### Progress Plots (`progress_iter_*.png`)
- Saved every 500 iterations
- Same layout as final results

## Advantages

1. **Direct Control**: Optimize μ values directly, not through NN weights
2. **Classical Theory**: Follows standard inverse problem methodology
3. **Interpretability**: No "black box" neural network
4. **Simplicity**: Fewer hyperparameters (no hidden dims, Fourier features for μ)
5. **Gradient Transparency**: Direct ∇_μ J, not ∇_θ J → ∇_θ μ → ∇_μ J

## Disadvantages

1. **Memory**: More parameters (N values vs NN weights)
2. **Smoothness**: Requires explicit regularization (TV, Sobolev)
3. **Generalization**: No spatial continuity guarantee without regularization

## Comparison with NN-Based Approach

| Aspect | NN-Based | Direct Mu |
|--------|----------|-----------|
| Parameters | ~few thousand (NN weights) | N (one per point) |
| Update | θ (weights) | μ (stiffness) |
| Smoothness | Implicit (NN architecture) | Explicit (TV, Sobolev) |
| Learning Rate | 0.001-0.005 | 10-20 |
| Convergence | Slower (indirect) | Faster (direct) |
| Memory | Low | Higher (N params) |

## Expected Results

With proper tuning:
- **Mu MSE**: Should decrease steadily
- **Data Loss**: Should converge to similar values as NN-based
- **Displacement Fit**: Should match measured data
- **Gradient Norm**: Should decrease over time

## Tips

1. **Learning Rate**: Start with 10.0, increase if too slow (up to 50.0)
2. **BC Weight**: Keep at 1000.0 for unique solution (high BC enforcement)
3. **TV Weight**:
   - 0.0 for smooth fields
   - 0.001-0.01 for piecewise-constant features
4. **Initialization**:
   - `constant` (5000 Pa): Safe, converges slowly
   - `random`: Faster, may be unstable
   - `uniform`: Good for sanity checks

## Forward Solver

The forward solver remains **UNCHANGED**:
- PIELM solver (Physics-Informed Extreme Learning Machine)
- Basis functions: sin(w·x + b)
- Differentiable through PyTorch autograd
- Gradients flow: J → u → C → H → μ

## Loss Functions

Same as NN-based approach:
- **MSE**: Standard L2 loss
- **Correlation**: Cosine similarity (phase/shape)
- **Relative L2**: Normalized MSE
- **Sobolev**: Gradient-enhanced (α=0.1 L2 + β=0.9 gradient)

## Example Output

```
Direct Mu Optimization: MSE Loss
======================================================================
  Points: 5000
  BC points: 1234 (24.7%)
  Iterations: 5000
  LR: 10.0, decay every 1000 by 0.9
  Weights: bc=1000.0, data=0.0, tv=0.0
  Loss type: mse
  Optimizer: Adam on μ_field directly
======================================================================
Iter     0: MSE=1.2345e-08, MuMSE=1.2345e+06, Grad=5.6789e+02
           mu=[4995, 5005] Pa, true=[3000, 10000] Pa
...
Iter  5000: MSE=1.2345e-10, MuMSE=2.3456e+04, Grad=1.2345e+00
           mu=[3100, 9800] Pa, true=[3000, 10000] Pa

FINAL RESULTS
======================================================================
  Data Loss: 1.2345e-10
  Mu MSE:    2.3456e+04
  Results saved to: outputs/direct_mu_mse
======================================================================
```

## Next Steps

1. Run baseline: `python train_direct_mu.py --loss_type mse`
2. Compare with Sobolev: `python train_direct_mu.py --loss_type sobolev`
3. Add TV regularization: `python train_direct_mu.py --tv_weight 0.01`
4. Tune learning rate based on gradient norm evolution
5. Compare results with NN-based approach (see `outputs/physical_sobolev/`)
