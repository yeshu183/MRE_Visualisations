# Troubleshooting: Mu MSE Increasing

## Problem Observed

When running with `physical_sobolev` experiment:
```
Iter     0: MuMSE=2.5681e+06
Iter  2300: MuMSE=1.4541e+07  ❌ INCREASING!
```

Even though μ range is correct [3000, 10000], the **Mu MSE is increasing**, meaning the neural network is learning the **wrong spatial distribution**.

## Root Cause Analysis

### Issue 1: No Data Guidance in Forward Solver
```python
'physical_sobolev': {
    'bc_weight': 1000.0,   # Very high - dominates optimization
    'data_weight': 0.0,    # ❌ NO DATA CONSTRAINT in forward solver!
    'pde_weight': 5.0,     # High PDE enforcement
}
```

**Problem**: With `data_weight=0.0`, the forward solver has NO constraint to match measured displacement **during the solve**. The network can output any μ(x) that satisfies BC + PDE, but produces displacement that happens to minimize Sobolev loss.

This leads to **pathological solutions**:
- Network outputs μ values that minimize `||u_pred - u_meas||²`
- But the spatial distribution of μ is WRONG
- Network exploits the fact that many different μ(x) can produce similar u(x)

### Issue 2: Very High BC Weight
`bc_weight=1000.0` makes boundary conditions dominate the forward solve, creating a very stiff optimization landscape where small changes in μ have large effects on the loss.

### Issue 3: No Regularization
`tv_weight=0.0` means no smoothness constraint, allowing the network to create wild spatial variations.

### Issue 4: Low Learning Rate
`lr=0.001` is too low to escape local minima once the network gets stuck in a bad solution.

## Solution

### Option 1: Use `physical_box` Experiment (RECOMMENDED)

This experiment has balanced weights:
```python
'physical_box': {
    'bc_weight': 10.0,      # ✅ Balanced (not too high)
    'data_weight': 10.0,    # ✅ Data guidance in forward solver
    'tv_weight': 0.01,      # ✅ Smoothness regularization
    'lr': 0.001,
}
```

**Command**:
```powershell
python bioqic_pielm/train.py --experiment physical_box --subsample 2000 --pde_weight 1.0
```

**Why this works**:
1. `data_weight=10.0` - Forward solver includes data constraint, preventing pathological solutions
2. `bc_weight=10.0` - Balanced, not dominating
3. `tv_weight=0.01` - Encourages smooth μ(x)
4. `pde_weight=1.0` - Physics guidance without overwhelming other terms

### Option 2: Fix `physical_sobolev` Manually

Override the bad defaults:
```powershell
# NOT POSSIBLE - train.py doesn't expose data_weight override
# Would need to modify EXPERIMENTS dict in train.py
```

### Option 3: Lower PDE Weight

The issue might also be **PDE weight too high**. Try reducing it:
```powershell
python bioqic_pielm/train.py --experiment physical_box --subsample 2000 --pde_weight 0.5
```

## Understanding the Mu MSE Trend

### Good Training (Mu MSE Decreasing)
```
Iter     0: MuMSE=5.2e+06, mu=[3531, 3531] Pa
Iter   500: MuMSE=3.8e+06, mu=[3200, 7600] Pa  ✅ Exploring range
Iter  1000: MuMSE=2.1e+06, mu=[3050, 9200] Pa  ✅ Getting closer
Iter  3000: MuMSE=1.2e+06, mu=[3010, 9850] Pa  ✅ Converging
```

The network is learning the correct spatial structure.

### Bad Training (Mu MSE Increasing) - YOUR CASE
```
Iter     0: MuMSE=2.5e+06, mu=[3531, 3531] Pa
Iter   100: MuMSE=12.9e+06, mu=[3003, 9608] Pa  ❌ Range OK but MSE increasing
Iter  2300: MuMSE=14.5e+06, mu=[3000, 10000] Pa  ❌ Full range but WRONG distribution
```

The network reached the correct range but learned the **wrong spatial patterns**. It's likely putting high μ where there should be low μ and vice versa.

### Why Does This Happen?

**Inverse Problem Ill-Posedness**: Multiple μ(x) distributions can produce similar displacement fields u(x) when:
1. No data constraint in forward solver (`data_weight=0`)
2. Very high BC weight creates artificial constraints
3. PDE loss can be minimized by wrong μ distributions that happen to satisfy the equation

## Mathematical Explanation

### Forward Solver (PIELM)
Solves: `∇·(μ∇u) + ρω²u = 0` subject to BCs

System matrix construction:
```python
H_pde = (μ/ρω²) * φ_lap + φ           # PDE rows
H_bc = φ[bc_indices] * bc_weight      # BC rows (WEIGHTED!)
H_data = φ * data_weight               # Data rows (if data_weight > 0)

H = [H_pde; H_bc; H_data]
b = [0; u_bc * bc_weight; u_meas * data_weight]
```

**When `data_weight=0`**:
- No `H_data` rows → forward solver ignores measured displacement
- Only satisfies PDE + BC
- Many μ(x) can satisfy this (non-unique!)

**When `bc_weight=1000`**:
- BC rows dominate: `||H_bc * C - b_bc||²` >>> `||H_pde * C||²`
- Forward solve essentially just interpolates BCs
- μ has little effect on the solution quality

### Loss Landscape
```
Total Loss = Sobolev(u_pred, u_meas) + pde_weight * PDE(μ, u_pred)

where:
  u_pred = ForwardSolve(μ, bc_weight=1000, data_weight=0)
```

The network finds:
1. **Step 1**: Pick any μ(x) → Forward solve gives some u_pred
2. **Step 2**: Adjust μ(x) to minimize `||u_pred - u_meas||²`
3. **Result**: u_pred matches u_meas, but μ(x) is WRONG

This is because the forward solver doesn't use u_meas, so the network can "cheat" by learning μ that produces the right displacement **after the fact**, not during solving.

## Recommended Fix

**STOP current training** and use:

```powershell
python bioqic_pielm/train.py --experiment physical_box --subsample 2000 --pde_weight 1.0
```

Expected behavior:
- Mu MSE should **decrease** steadily
- μ range should expand: [3531, 3531] → [3000, 9500]
- Final Mu MSE: ~1-2 million (not 14 million!)

## Verification

After training completes, check:

1. **Mu MSE trend**: Should be decreasing
2. **Spatial maps**: `outputs/physical_box/final_results.png`
   - Compare true μ map vs predicted μ map
   - Should show similar spatial patterns (not random)
3. **Histogram**: Should show bimodal distribution matching true μ

If Mu MSE is still increasing with `physical_box`, the problem is deeper and might require:
- Even lower PDE weight (try 0.1 or 0.0)
- Higher learning rate (try 0.005)
- More TV regularization (try tv_weight=0.05)
