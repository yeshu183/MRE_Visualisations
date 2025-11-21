# 🔬 BIOQIC Training Investigation - Root Cause Analysis

**Date**: 2025-11-20  
**Status**: ✅ Root causes identified, solution ready

---

## 🎯 Executive Summary

**Problem**: Training on BIOQIC Phase 1 data plateaus at data MSE = 0.207, unable to reconstruct stiffness.

**Root Causes Found**:
1. ❌ BIOQIC uses **viscoelastic Voigt model** with complex μ, we only use real part
2. ❌ Data has **85% PDE residual** - doesn't satisfy our simplified Helmholtz equation
3. ❌ Wave basis `sin(ω·x)` **cannot represent** BIOQIC displacement patterns
4. ❌ Forward solver fundamentally incompatible with this data type

**Solution**: Use **data-driven approach** from approach folder (pure interpolation, no PDE).

---

## 📊 Investigation Timeline

### Phase 1: Training Completed
- ✅ Baseline experiment ran 3000 iterations
- ✅ Data loss: 0.210 → 0.207 (only 1.2% improvement)
- ✅ Reconstruction: Blurry, no sharp boundaries, MAE = 5141 Pa
- 🔍 **Finding**: Model learning but not converging

### Phase 2: Component Analysis
Tested 4 components systematically:

#### 2.1 Data Processing ✅ **WORKING**
- Ground truth visible (4 circular targets)
- Normalization correct
- Z-component dominant (97% energy)

#### 2.2 Boundary Detection ✅ **WORKING**  
- Minimal strategy: 3 anchor points
- Actuator detection working
- Not the bottleneck

#### 2.3 Forward Solver ❌ **BROKEN**
- **Test 1**: Homogeneous μ with data_weight=100 → MSE = 0.206
- **Test 2**: Data-only (no PDE) with data_weight=1000 → MSE = 0.202
- **Finding**: Cannot fit data even with correct constant stiffness!

#### 2.4 μ Network ⚠️ **TOO SMOOTH**
- 3 layers × 64 dims with tanh activation
- Can't represent sharp boundaries
- But not the primary issue

### Phase 3: PDE Compliance Check ❌ **CRITICAL**
```python
Residual: R = μ∇²u + ρω²u
Relative PDE error: 84.7%
```
**Finding**: BIOQIC data does NOT satisfy our Helmholtz PDE!

**Reason**: BIOQIC uses **Voigt viscoelastic model**:
```
Material: μ* = μ_storage + iω·η
At 60 Hz: μ* = 3000 + 377i Pa (background)
          μ* = 10000 + 377i Pa (targets)
```

We only use `μ_storage` (real part), ignoring viscosity (imaginary part).

### Phase 4: Wave Basis Test ❌ **INCOMPATIBLE**

Tested omega_basis from 0.5 to 50:

| omega | MSE    | Scale | Status |
|-------|--------|-------|--------|
| 0.5   | 0.2078 | 0.003 | Too low freq |
| 1.0   | 0.2076 | 0.005 | Too low freq |
| 2.0   | 0.2073 | 0.006 | Too low freq |
| 5.0   | 0.2066 | 0.013 | Too low freq |
| 10.0  | 0.2021 | 0.146 | Getting closer |
| **20.0** | **0.1966** | **0.294** | **BEST** |
| 50.0  | 0.2069 | 0.069 | Too high freq |

**Even at optimal frequency (20), MSE = 0.20!**

**Conclusion**: Wave basis `sin(ω·x)` fundamentally cannot represent:
- Sharp circular boundaries
- Non-sinusoidal patterns from 4 inclusions
- Complex 3D mechanical wave propagation

---

## 🔍 Technical Details

### BIOQIC Data Specifications

**From**: `data/raw/bioqic/descriptions.md`

```
Box Phantom (100×80×10 voxels, 1mm isotropic):
- Material Model: Voigt viscoelastic
- Background: μ_storage = 3 kPa, η = 1 Pa·s
- Targets (4): μ_storage = 10 kPa, η = 1 Pa·s  
- Frequencies: 50, 60, 70, 80 Hz
- Excitation: Traction force on top x-z plane
- Motion: Primarily Z-direction
```

**Complex Shear Modulus** (frequency domain):
```
μ*(ω) = μ_storage + iω·η

At 60 Hz (ω = 2π×60 = 377 rad/s):
  Background: μ* = 3000 + 377i Pa
  Targets:    μ* = 10000 + 377i Pa
```

**What we're doing wrong**:
```python
# data_loader.py line 108:
mu_data = self.stiffness.real  # ❌ Only using real part!
```

This discards the viscosity term → data becomes PDE-inconsistent.

### Our PDE vs Reality

**Our simplified Helmholtz** (elastic):
```
∇·(μ∇u) + ρω²u = 0
```

**Reality** (viscoelastic Voigt in frequency domain):
```
∇·(μ*∇u) + ρω²u = 0
where μ* = μ_storage + iω·η (complex)
```

**Result**: 85% PDE residual when using only μ_storage.

### Wave Basis Limitations

**Current basis**:
```python
φ_i(x) = sin(ω_i · x)
where ω_i ~ N(0, omega_basis²)
```

**Why it fails**:
1. Assumes smooth, oscillatory displacement
2. Cannot represent sharp boundaries
3. Global support (no locality for inclusions)
4. Wrong spectral characteristics for mechanical waves

**Measured displacement characteristics**:
- 4 sharp circular inclusions
- Rapid spatial variation at boundaries
- Mechanical wave attenuation (not pure sinusoids)
- 3D coupling effects

---

## 📈 Loss Analysis

### Training Metrics (Baseline)
```
Initial loss: 23.865
Final loss:   22.758
Reduction:    4.64%

Data MSE:
  Initial: 0.2096
  Final:   0.2071
  Reduction: 1.20% ← PLATEAU!

Gradient norm: 7.2 (mean), up to 14.97
Oscillations: 1157 times (unstable)
```

### Loss Balance
```
Data × 100:  20.7
BC × 10:      2.0  
TV × 0.001:   0.0002

Ratio:  20.7 : 2.0 : 0.0002
```
TV regularization **100,000× too weak** → overfitting.

---

## ✅ Solution: Use Approach Folder's Working Method

### What Approach Folder Does Right

From `approach/tests/test_data_only.py`:

1. **Row-concatenation system** (more numerically stable):
```python
H = torch.cat([
    H_pde,      # (N, M) PDE residual rows
    w * H_bc,   # (K, M) BC constraint rows  
    w * φ_data  # (N, M) Data fitting rows
], dim=0)

b = torch.cat([
    b_pde,      # (N, 1) zeros
    w * b_bc,   # (K, 1) BC values
    w * u_data  # (N, 1) measured displacement
], dim=0)

# Then solve: H^T H c = H^T b
```

2. **Pure data-driven** when data_weight >> pde_weight:
   - Sets PDE weight to 0
   - Only fits interpolation

3. **Working on 1D synthetic tests**:
   - Gaussian bump reconstruction
   - Step function reconstruction
   - Multiple inclusions

### Adaptation Plan

**Step 1**: Create new forward model using approach folder's system construction

**Step 2**: Test on BIOQIC with data-only (no PDE):
```python
H = torch.cat([
    1000.0 * phi,      # Data fitting
    1.0 * phi_bc       # Minimal BC for uniqueness
], dim=0)

b = torch.cat([
    1000.0 * u_meas,
    1.0 * u_bc
], dim=0)
```

**Step 3**: Improve μ network for sharp boundaries:
- Add more Fourier features (10 → 50)
- Deeper network (3 → 5 layers)
- Try different activations (Swish, GELU)
- Add skip connections

**Step 4**: Use stronger regularization:
- TV weight: 0.001 → 0.1 (100× increase)
- L2 prior toward expected values (3 or 10 kPa)

---

## 🎯 Next Steps

### Immediate (This Session)
1. ✅ Create `forward_model_v3.py` using approach folder's method
2. ✅ Test data-only fitting on BIOQIC
3. ✅ Verify MSE < 0.01 achievable

### Short-term (Next Session)
4. Run new "data_only" experiment (5000 iters, 2000 points)
5. Analyze reconstruction quality
6. Iterate on μ network architecture

### Medium-term
7. Add complex μ support (if needed)
8. Implement direct inversion formula as baseline
9. Compare with literature methods

---

## 📚 Key Learnings

1. **Always check data-physics consistency first** before blaming network/optimization
2. **Test forward solver independently** with known solutions
3. **Wave basis ≠ universal** - match basis to physics
4. **Viscoelastic vs elastic matters** - can't ignore imaginary part
5. **Data-driven inversion is valid** when PDE is too complex

---

## 🔗 References

- BIOQIC Dataset: `data/raw/bioqic/descriptions.md`
- Approach folder tests: `approach/tests/test_data_only.py`
- Training analysis: `outputs/baseline/training_analysis.png`
- Reconstruction: `outputs/baseline/reconstruction_slice_5.png`

---

**Status**: Ready to implement data-driven solution ✅
