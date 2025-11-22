## Gradient Term Effect Test

### Overview

This test evaluates whether including the gradient term (∇μ·∇u) in the weak form PDE helps discriminate between constant and heterogeneous stiffness fields.

### Motivation

**Current implementation:**
```
∇·(μ∇u) ≈ μ·∇²u + ρω²·u = 0
```
This assumes μ is constant (∇μ = 0), which is valid for homogeneous materials but not for heterogeneous stiffness fields.

**Full weak form:**
```
∇·(μ∇u) = μ·∇²u + ∇μ·∇u + ρω²·u = 0
```
This includes the gradient term (∇μ·∇u), which should be significant when μ varies spatially.

### Hypothesis

1. **Constant μ:** Both formulations should perform identically since ∇μ = 0
2. **Heterogeneous μ:** Full form should be more accurate since ∇μ ≠ 0
3. **Discrimination:** Full form should better distinguish const vs hetero fields

### Test Configuration

Based on grid search findings:

- **Sampling:** Uniform (not adaptive)
- **Neurons:** [1,000, 5,000, 10,000] (grid search across neuron counts)
- **Points:** 5,000
- **BC weight:** 10
- **Omega basis:** 170.0

**Note:** Analytical gradients are used instead of autograd to avoid memory issues. For sin basis `φ(x) = sin(w·x + b)`, we use `∇φ = cos(w·x + b) * w` and `∇²φ = -||w||² * φ`.

### Running the Test

```bash
python bioqic_pielm/test_gradient_term_effect.py
```

**Expected runtime:** ~15-30 minutes (tests 12 configurations: 3 neuron counts × 4 test cases)

### Test Cases

| Test Name | μ Type | Gradient Term | Expected Outcome |
|-----------|--------|---------------|------------------|
| `const_no_grad` | Constant (5000 Pa) | No | Baseline |
| `const_with_grad` | Constant (5000 Pa) | Yes | Same as baseline (∇μ = 0) |
| `hetero_no_grad` | Heterogeneous (3-10 kPa) | No | Current implementation |
| `hetero_with_grad` | Heterogeneous (3-10 kPa) | Yes | Should improve if gradient term matters |

### Metrics

#### Primary Metric: Blob R²
- **Definition:** R² computed only on stiff blob regions (μ > 8 kPa)
- **Why it matters:** Blob boundaries have highest μ gradients (∇μ ≠ 0)
- **Expected:** `hetero_with_grad` should have highest blob R²

#### Discrimination Score
```python
blob_r2_diff = |Blob R²(hetero) - Blob R²(const)|
```
- **Higher = Better** at detecting stiffness heterogeneity
- **Expected:** `with_grad` should have higher discrimination than `no_grad`

### Interpretation

#### Scenario 1: Gradient Term Helps
```
hetero_with_grad:    Blob R² = 0.87
hetero_no_grad:      Blob R² = 0.86
Improvement: +0.01 (1%)
```
**Conclusion:** Include gradient term in forward model

#### Scenario 2: No Effect
```
hetero_with_grad:    Blob R² = 0.86
hetero_no_grad:      Blob R² = 0.86
Improvement: ±0.001
```
**Conclusion:** Gradient term doesn't matter (current implementation is fine)

**Possible reasons:**
1. PIELM basis functions already capture μ·∇²u + ∇μ·∇u implicitly
2. Gradient term magnitude is small compared to μ·∇²u
3. BC enforcement dominates (bc_weight=10 is strong)

#### Scenario 3: Gradient Term Hurts
```
hetero_with_grad:    Blob R² = 0.85
hetero_no_grad:      Blob R² = 0.86
Degradation: -0.01
```
**Conclusion:** Gradient term adds numerical noise

**Possible reasons:**
1. Finite difference ∇μ estimation is inaccurate
2. Autograd gradients are noisy
3. Ill-conditioning in least-squares system

### Implementation Details

#### Gradient Computation

**Basis function gradients:**
```python
for j in range(n_neurons):
    # Compute ∇φ_j using autograd
    grad_phi = torch.autograd.grad(phi[:, j], x, create_graph=True)[0]

    # Compute ∇²φ_j (Laplacian)
    lap = sum(∂²φ_j/∂x_d² for d in [0, 1, 2])
```

**Stiffness gradient (finite differences):**
```python
for d in [0, 1, 2]:
    ∇μ[:, d] = (μ(x + ε·e_d) - μ(x - ε·e_d)) / (2ε)
```

**Gradient term:**
```python
grad_term[j] = ∇μ · ∇φ_j  # dot product
```

#### PDE Rows in System

**Without gradient term (current):**
```python
H_pde = (μ / ρω²) * ∇²φ + φ
b_pde = 0
```

**With gradient term (full form):**
```python
H_pde = (μ / ρω²) * ∇²φ + (∇μ·∇φ / ρω²) + φ
b_pde = 0
```

### Output Files

#### CSV Files

**1. `gradient_term_comparison.csv`** - Full results for all configurations

| neurons | test_name | mu_type | include_grad_term | r2 | blob_r2 | mse |
|---------|-----------|---------|-------------------|----|---------| ----|
| 1000 | const_no_grad | constant_5000 | False | 0.9921 | 0.8163 | ... |
| 1000 | const_with_grad | constant_5000 | True | 0.9921 | 0.8163 | ... |
| 1000 | hetero_no_grad | heterogeneous | False | 0.9921 | 0.8163 | ... |
| 1000 | hetero_with_grad | heterogeneous | True | 0.9925 | 0.8245 | ... |
| 5000 | ... | ... | ... | ... | ... | ... |

**2. `gradient_term_summary.csv`** - Summary statistics by neuron count

| neurons | hetero_improvement | discrimination_improvement | const_no_blob_r2 | hetero_with_blob_r2 |
|---------|-------------------|---------------------------|------------------|---------------------|
| 1000 | +0.0041 | +0.0021 | 0.8163 | 0.8245 |
| 5000 | ... | ... | ... | ... |

#### Visualizations

All plots saved to `outputs/gradient_term_test/`:

1. **`blob_r2_comparison.png`** - Bar chart comparing Blob R² across all configurations
   - Shows 4 bars per neuron count: const (no/with grad), hetero (no/with grad)
   - Identifies best configuration visually

2. **`gradient_term_improvement.png`** - Two-panel plot showing gradient term impact
   - Left: Heterogeneous case improvement/degradation
   - Right: Discrimination ability change
   - Green bars = improvement, Red bars = degradation

3. **`discrimination_comparison.png`** - Line plot showing discrimination ability
   - Blue line: Without gradient term
   - Red line: With gradient term
   - Shows if gradient term helps distinguish const vs hetero fields

4. **`overall_vs_blob_r2.png`** - Scatter plot of Overall R² vs Blob R²
   - Circle = constant μ, Triangle = heterogeneous μ
   - Blue = no gradient term, Red = with gradient term
   - Point size = neuron count

5. **`blob_r2_heatmap.png`** - Heatmap showing all Blob R² values
   - Rows = neuron counts (1000, 5000, 10000)
   - Cols = test configurations
   - Color intensity = Blob R² value (green = high, red = low)

6. **`mu_gradient_visualization.png`** - Diagnostic plots for ∇μ quality
   - Shows μ field, ∂μ/∂x, ∂μ/∂y, |∇μ|, gradient directions, and histogram
   - Reveals quality issues with finite difference method

7. **`mu_gradient_3d.png`** - 3D visualization of gradient magnitude
   - Shows spatial distribution of |∇μ| throughout domain

### Gradient Quality Diagnostic

Run this to visualize the finite difference gradient estimation:

```bash
python bioqic_pielm/visualize_mu_gradients.py
```

**Key findings:**
- 83% of gradients are zero (no nearby neighbors)
- 17% have extremely large magnitudes (1-8 million Pa/m)
- Gradients are 11.6× larger than physically expected
- Finite differences on random sampling produce artifacts, not smooth gradients

### Limitations

1. **Finite difference gradient:** Simple central differences, may be noisy
2. **Computational cost:** Autograd is slow (~10× slower than current implementation)
3. **Nearest neighbor interpolation:** μ gradient computed at same points (no interpolation)

### Next Steps

**If gradient term helps:**
1. Implement analytical gradients for speed
2. Use smooth μ interpolation (splines, RBF)
3. Include gradient term in inverse problem

**If gradient term doesn't help:**
1. Abandon this approach
2. Focus on other improvements (adaptive sampling, loss functions, etc.)
3. Document why current approximation is sufficient

---

**Created:** 2025-01-22
**Purpose:** Determine if full weak form ∇·(μ∇u) = μ·∇²u + ∇μ·∇u improves accuracy vs simplified form μ·∇²u
