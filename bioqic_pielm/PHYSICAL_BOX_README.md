# Physical Box Experiment - Fixed Forward Model for Inverse Problem

## Summary of Changes

The forward model has been successfully validated with optimal parameters. The inverse problem training has been updated to use the same physics-consistent configuration.

## Key Issues Fixed

### 1. Unit Mismatch ✅
- **Problem**: Old experiments used normalized μ ∈ [1, 2] with physical ρω² ≈ 1.42e8
- **Result**: The stiffness term vanished (1/1.4e8 ≈ 10⁻⁹), causing trivial solutions
- **Fix**: New `physical_box` experiment uses physical units μ ∈ [3000, 10000] Pa

### 2. Boundary Starvation ✅
- **Problem**: `actuator` or `minimal` BC strategies provided insufficient constraints
- **Result**: Waves decayed to zero away from boundaries
- **Fix**: Full box BC strategy enforces boundaries on all 6 faces

### 3. Neuron Starvation ✅
- **Problem**: Old experiments used 60-80 neurons
- **Result**: Insufficient capacity to resolve inclusions (acts like low-pass filter)
- **Fix**: Increased to 1000 neurons based on grid sweep results

### 4. Basis Function ✅
- **Problem**: Some experiments used tanh basis
- **Fix**: Standardized on sin (Fourier) basis functions

### 5. Data Loading ✅
- **Problem**: Data loader was taking absolute values, destroying phase information
- **Fix**: Now uses signed displacement values (real part of complex data)

## Optimal Parameters (from Grid Sweep)

Based on comprehensive testing, the optimal configuration is:

```python
'physical_box': {
    'sampling_points': 10000,      # Balance between accuracy and speed
    'n_wave_neurons': 1000,        # Sufficient capacity for inclusions
    'bc_weight': 10,               # Optimal BC enforcement
    'omega_basis': 170.0,          # Physical wavenumber (2π/λ)
    'rho_omega2': 1.42e8,         # Physical value (60 Hz, ρ=1000 kg/m³)
    'mu_range': (3000, 10000),    # Physical Pascals
    'basis_type': 'sin',           # Fourier basis
    'bc_strategy': 'box',          # All 6 faces
}
```

## File Structure

### Main Files
- **`test_forward_model.py`**: Test forward solver with optimized fixed parameters
- **`grid_sweep_forward.py`**: Comprehensive grid sweep for parameter exploration
- **`train.py`**: Training script with `physical_box` experiment
- **`forward_model.py`**: PIELM forward solver (supports sin/tanh basis)
- **`data_loader.py`**: Fixed to use signed displacement values
- **`trainer.py`**: Training loop (unchanged, works with any units)
- **`pielm_solver.py`**: Differentiable least-squares solver
- **`stiffness_network.py`**: Neural network for μ(x)

### How Each File Uses the Forward Model

#### 1. `test_forward_model.py` (Testing)
```python
# Tests forward solve with KNOWN ground truth stiffness
model = ForwardMREModel(
    n_wave_neurons=1000,
    omega_basis=170.0,
    mu_min=3000.0,  # Physical range
    mu_max=10000.0,
    basis_type='sin'
)

# Solve: given μ_true, predict u
u_pred, _ = model.solve_given_mu(
    x, mu_true, bc_indices, u_bc_vals, rho_omega2,
    bc_weight=10
)
```

#### 2. `grid_sweep_forward.py` (Parameter Search)
```python
# Tests all combinations:
# - bc_weights: [0, 1, 10, 100]
# - neurons: [2000, 5000, 10000]
# - sampling: [1000, 10000, 50000]

# Finds optimal parameters for forward solve
```

#### 3. `train.py` (Inverse Problem)
```python
# Uses forward model for INVERSE solve
# Network predicts μ, forward solver predicts u

model = ForwardMREModel(
    n_wave_neurons=1000,
    omega_basis=170.0,
    mu_min=3000.0,  # Physical range
    mu_max=10000.0,
    basis_type='sin'
)

# Training loop:
# 1. Neural net predicts: μ_pred = StiffnessNet(x)
# 2. Forward solve: u_pred = PIELM(μ_pred)
# 3. Loss: ||u_pred - u_meas||²
# 4. Backprop through solver to update StiffnessNet
```

## Usage

### Forward Model Testing
```bash
# Test with optimized parameters
python bioqic_pielm/test_forward_model.py

# Run comprehensive grid sweep (takes time!)
python bioqic_pielm/grid_sweep_forward.py
```

### Inverse Problem Training
```bash
# Run the optimized physical_box experiment
python bioqic_pielm/train.py --experiment physical_box --subsample 10000

# Quick test with fewer points
python bioqic_pielm/train.py --experiment physical_box --subsample 5000

# Other experiments (for comparison)
python bioqic_pielm/train.py --experiment baseline
python bioqic_pielm/train.py --experiment actuator
```

## Expected Results

### Forward Model Test (test_forward_model.py)
- **MSE**: ~1e-12 to 1e-10
- **R²**: > 0.95
- **u_pred range**: Similar to u_meas (±0.02 m)
- **Visualization**: Diagonal scatter plot showing good correlation

### Inverse Problem (train.py --experiment physical_box)
- **Early iterations**: Random μ prediction, poor u fit
- **Mid training**: μ structure emerges, u fit improves
- **Convergence**: μ_pred should show 4 cylindrical inclusions
- **Final R²**: Target > 0.8 for displacement fit

## Key Differences from Old Experiments

| Aspect | Old Experiments | physical_box |
|--------|----------------|--------------|
| μ range | [1, 2] (normalized) | [3000, 10000] Pa (physical) |
| ρω² | 400 (effective) | 1.42e8 (physical) |
| Neurons | 60-80 | 1000 |
| BCs | Actuator only | Full box (6 faces) |
| Basis | Mixed | Sin (Fourier) |
| Displacement | abs(u) | Signed (real part) |
| Sampling | 5000 | 10000 |
| BC weight | 100-200 | 10 |

## Troubleshooting

### If forward solve gives flat-line (R² ≈ 0):
- Check BC strategy is 'box'
- Verify bc_weight is not too high (try 10)
- Confirm using signed displacement (not absolute values)

### If inverse training doesn't converge:
- Reduce learning rate (try 0.001)
- Increase TV weight for smoother μ (try 0.01)
- Check BC values have variation (std > 1e-4)
- Verify using physical μ range [3000, 10000]

### If memory issues:
- Reduce subsample (try 5000)
- Reduce n_wave_neurons (try 500, but expect blurring)
- Use CPU if GPU memory insufficient

## Next Steps

1. Run forward test to verify setup:
   ```bash
   python bioqic_pielm/test_forward_model.py
   ```

2. Run inverse training:
   ```bash
   python bioqic_pielm/train.py --experiment physical_box --subsample 10000
   ```

3. Monitor training:
   - Check `outputs/physical_box/` for progress plots
   - Look for μ structure emergence around iteration 2000-3000
   - Final results should show clear inclusion boundaries

4. Compare with baseline:
   ```bash
   python bioqic_pielm/train.py --experiment baseline --subsample 10000
   ```

## References

- Forward model validation: `outputs/forward_model_tests/u_comparison.png`
- Grid sweep results: `outputs/forward_model_tests/comprehensive_grid_sweep.png`
- Training progress: `outputs/physical_box/progress_iter_*.png`
