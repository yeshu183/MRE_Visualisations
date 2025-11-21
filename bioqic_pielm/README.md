# BIOQIC-PIELM: Physics-Informed MRE Inversion

Solves the inverse problem in Magnetic Resonance Elastography (MRE) to reconstruct tissue stiffness from wave displacement measurements.

## Overview

This implementation uses **Physics-Informed Extreme Learning Machine (PIELM)** with:
- Custom differentiable solver with analytical gradient backpropagation
- Random Fourier basis functions for wave field approximation
- Neural network parameterization of stiffness field

### The Inverse Problem

Given measured displacement field `u(x)` from MRE scan, recover stiffness `μ(x)`.

**Physics**: Helmholtz wave equation
```
∇·(μ∇u) + ρω²u = 0
```

**Approach**: Analysis-by-synthesis
1. Neural network predicts μ(x)
2. PIELM solver computes u(x) given μ
3. Compare predicted vs measured displacement
4. Backpropagate through solver to update μ-network

## BIOQIC Dataset

BIOQIC FEM Phantom (Phase 1 Box):
- **Grid**: 80 × 100 × 10 mm (1mm voxels)
- **Frequency**: 60 Hz (ω = 377 rad/s)
- **Material**: Voigt viscoelastic (G* = μ + iωη)
- **Background**: μ = 3 kPa, η = 1 Pa·s
- **Inclusions**: μ = 10 kPa (4 cylindrical targets)

## Quick Start

```bash
# Default baseline experiment
python train.py

# Physics-informed BCs (actuator on top Y-face)
python train.py --experiment actuator

# Data-driven approach (no PDE constraints)
python train.py --experiment data_only

# Quick test with fewer points
python train.py --subsample 2000
```

## Experiments

| Experiment | Description | Key Settings |
|------------|-------------|--------------|
| `baseline` | Minimal BCs, effective physics | bc_weight=200, data_weight=0 |
| `actuator` | Physics-informed top Y-face BCs | bc_weight=100 |
| `data_only` | Pure data fitting (no PDE) | data_weight=100, bc_weight=0 |
| `strong_tv` | Strong regularization | tv_weight=0.01 |
| `physical` | True physical rho_omega2 | Uses 142M instead of 400 |

## Key Files

| File | Purpose |
|------|---------|
| `pielm_solver.py` | Differentiable least-squares with custom backward |
| `data_loader.py` | BIOQIC data loading and preprocessing |
| `stiffness_network.py` | Neural network for μ(x) |
| `forward_model.py` | PIELM forward solver |
| `trainer.py` | Training loop with visualization |
| `train.py` | Main training script |

## Critical Hyperparameters

From validated approach folder:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `bc_weight` | 100-200 | BC row emphasis (critical!) |
| `data_weight` | 0 | Set to 0 for inverse problem |
| `tv_weight` | 0.001-0.01 | Total variation regularization |
| `n_wave_neurons` | 60 | Number of basis functions |
| `omega_basis` | 15.0 | Wave frequency scale |
| `lr` | 0.005 | Learning rate |

### Why bc_weight=200?

Counter-intuitive but essential! With 2 BC rows and 100 PDE rows:
- BC rows contribute `bc_weight² × 2` to energy
- PDE rows contribute ~100 each
- Need `bc_weight² × 2 ≈ 100` → bc_weight ≈ 7 for balance
- But BCs must **dominate** for unique solution → use 100-200

### Why data_weight=0?

Data constraints suppress gradients:
- `data_weight=0`: grad_norm ≈ 2e-4 ✓
- `data_weight=10`: grad_norm ≈ 5e-5 (77% reduction!)

Data rows don't depend on μ, diluting gradient signal.

## PIELM Solver

Custom `torch.autograd.Function` with analytical backward:

```
Forward: C = (H^T H + λI)^{-1} H^T b
Backward: dC = -(H^T H)^{-1} [dH^T r + H^T dH C]
```

where `r = HC - b` is residual.

Features:
- Cholesky decomposition with adaptive regularization
- Fallback to QR/SVD if ill-conditioned
- Enables gradient flow: Loss → u → C → H → μ

## Known Limitations

1. **Viscoelastic Mismatch**: BIOQIC uses complex modulus G* = μ + iωη, our model uses real μ only
2. **Wave Basis**: Sine functions may not capture sharp inclusion boundaries
3. **3D Data**: Full 80K points may require GPU and batching

## Results Directory

Training outputs saved to `outputs/{experiment}/`:
- `progress_iter_*.png`: Training progress snapshots
- `final_results.png`: Final reconstruction comparison
- `training_history.npy`: Loss curves and metrics

## References

- BIOQIC FEM Phantom documentation
- approach/ folder: Validated PIELM framework
- COMPREHENSIVE_LEARNINGS.md: Detailed analysis of approach
