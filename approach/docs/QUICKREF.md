# Quick Reference Card

## 📁 What's Where?

| Want to... | Go to... |
|-----------|----------|
| **Run examples** | `examples/run_all_examples.py` |
| **Test core math** | `tests/test_core_components.py` |
| **Tune parameters** | `config_forward.json` |
| **Debug issues** | `debug/` folder |
| **Read docs** | `docs/VALIDATION_REPORT.md` |
| **Add new test case** | Copy `examples/example_gaussian_bump.py` |

---

## ⚙️ Critical Parameters

```json
{
  "bc_weight": 200,      // 🔴 CRITICAL: Must be 100-200 for unique solution
  "data_weight": 0,      // ⚠️  Keep at 0 for inverse problems
  "tv_weight": 0.001,    // Use 0.001 for smooth, 0.002 for discontinuous
  "lr": 0.005,          // Learning rate
  "iterations": 5000     // Training iterations
}
```

---

## 🚨 Common Issues & Fixes

### Issue: Network predicts constant mu (flat line)

**Causes:**
1. `bc_weight` too low (< 50)
2. `data_weight` too high (> 5)
3. Zero boundary conditions producing zero wave field

**Fixes:**
- Set `bc_weight = 200`
- Set `data_weight = 0`
- Check BC values are non-zero (currently set to [0.01, 0.0])

### Issue: Gradients vanish (< 1e-6)

**Causes:**
1. Data constraints dominating
2. BC too weak

**Fixes:**
- Increase `bc_weight` to 200
- Reduce or remove `data_weight`
- Check gradient norms in output

### Issue: Poor reconstruction despite low loss

**Causes:**
1. Multiple solutions possible (underconstrained)
2. Loss landscape has many local minima

**Fixes:**
- Ensure `bc_weight >= 100`
- Add TV regularization (`tv_weight = 0.001`)
- Check if BCs are enforced (BC error should be < 1e-9)

### Issue: Network hits clamps [0.7, 6.0]

**Causes:**
1. Learning rate too high
2. Bad initialization
3. Gradient exploding

**Fixes:**
- Reduce `lr` from 0.005 to 0.001
- Check gradient clipping is disabled or set high (1.0)
- Network now uses Fourier features (should prevent this)

---

## 📊 Expected Results

### Gaussian Bump
- Data loss: ~1e-4
- Mu MSE: ~0.5-1.0
- Gradient norm: ~2e-4
- Status: ✅ Works

### Multiple Inclusions
- Data loss: ~1e-4
- Mu MSE: ~1.0-2.0
- Gradient norm: ~2e-4
- Status: ✅ Works

### Step Function
- Data loss: ~1e-4
- Mu MSE: ~2.0-3.0 (higher due to smoothing)
- Gradient norm: ~2e-4
- Status: ⚠️ Acceptable (inherent smoothing)

---

## 🔬 Validation Checklist

Before claiming the method works, verify:

- [ ] `test_core_components.py` passes all 7 tests
- [ ] Gradient norm > 1e-5 throughout training
- [ ] BC error < 1e-9 (boundaries enforced)
- [ ] PDE residual < 1e-6 (physics satisfied)
- [ ] Mu doesn't hit clamps [0.7, 6.0]
- [ ] Loss decreases over iterations
- [ ] Example scripts run without errors

---

## 🎯 For Real MRE

1. **Estimate BCs** from boundary measurements:
   ```python
   bc_indices = [0, -1]  # Image edges
   u_bc_vals = u_meas[[0, -1]]  # Use measured values
   ```

2. **Adjust weights**:
   ```json
   {
     "bc_weight": 50-100,   // Lower confidence than synthetic
     "data_weight": 1-2,    // Moderate interior weighting
     "tv_weight": 0.002     // More regularization
   }
   ```

3. **Increase iterations**: Weaker gradients need more steps
   ```json
   {
     "iterations": 10000,
     "early_stopping_patience": 2000
   }
   ```

---

## 📞 Quick Diagnostics

### Check if BCs are working:
```bash
python approach/debug/diagnose_bc_scaling.py
```
Look for: BC contribution > 0.1% of total

### Check if data weights hurt gradients:
```bash
python approach/debug/diagnose_data_scaling.py
```
Look for: Gradient norm stays > 1e-4

### Check if network can learn:
```bash
python approach/tests/test_mu_network.py
```
Look for: Final MSE < 0.001

---

## 🔑 Key Takeaways

1. **BC weighting is NOT intuitive**: Need ~100-200 to balance 2 BC rows vs 100 PDE rows
2. **Data constraints hurt inverse problems**: They suppress gradients even though they improve fit
3. **Fourier features are essential**: Prevent network from collapsing to constant output
4. **For real MRE**: Use estimated BCs from edges + moderate interior weighting

---

## 📚 Learn More

- Full validation results: `docs/VALIDATION_REPORT.md`
- Usage guide: `docs/MODULAR_README.md`
- Main overview: `docs/README.md`
