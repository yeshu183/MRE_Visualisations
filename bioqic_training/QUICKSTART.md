# Quick Start Guide - BIOQIC Training

## 🚀 Ready to Train!

All components are implemented and tested. Follow these steps to run experiments.

---

## ✅ Prerequisites

```bash
# Activate environment
conda activate MRE-PINN

# Navigate to training folder
cd bioqic_training
```

---

## 📋 List Available Experiments

```bash
python train_bioqic.py --list-experiments
```

**Available experiments:**
1. **baseline** - Simplest (minimal BCs, z-component, effective physics)
2. **actuator** - Physics-informed BCs (top Y-face)
3. **vector** - Full 3-component vector displacement
4. **physical** - Physical ρω² with coordinate rescaling
5. **strong_tv** - Strong TV regularization
6. **more_data** - 5000 data points for better coverage

---

## 🎯 Run Your First Experiment

### **Recommended: Start with baseline**

```bash
python train_bioqic.py --experiment baseline
```

This will:
- Load 1000 subsampled points
- Use Z-component displacement (dominant direction)
- Apply minimal boundary conditions (3 anchor points)
- Train for 3000 iterations (~5-10 minutes)
- Generate comprehensive visualizations every 500 iterations
- Save results to `outputs/baseline/`

**Expected output structure:**
```
outputs/baseline/
├── config.json                 # Experiment configuration
├── training_history.json       # All metrics per iteration
├── best_model.pt              # Best model checkpoint
├── progress_iter_00500.png    # Visualization at iter 500
├── progress_iter_01000.png    # Visualization at iter 1000
├── progress_iter_01500.png    # Visualization at iter 1500
└── ...
```

---

## 📊 Understanding the Visualizations

Each progress plot shows **12 subplots**:

### **Row 1: Loss Evolution (4 plots)**
1. Total loss (log scale)
2. Loss components (data vs BC)
3. Gradient norm
4. Learning rate schedule

### **Row 2: Stiffness Evolution (4 plots)**
5. μ range evolution (min/max/mean vs target 3000-10000 Pa)
6. μ variability (standard deviation)
7. μ distribution histogram (predicted vs ground truth)
8. μ reconstruction scatter (predicted vs true)

### **Row 3: Displacement Fit (4 plots)**
9. Measured displacement (3D)
10. Predicted displacement (3D)
11. Displacement error (3D)
12. Displacement fit scatter (predicted vs measured)

---

## 🔬 Run Different Experiments

### **Physics-Informed Boundaries**
```bash
python train_bioqic.py --experiment actuator
```
Uses top Y-face as actuator (traction force location from BIOQIC FEM).

### **Full Vector Displacement**
```bash
python train_bioqic.py --experiment vector
```
Uses all 3 components [u_x, u_y, u_z] instead of just Z.

### **Physical Scaling**
```bash
python train_bioqic.py --experiment physical
```
Uses true physical ρω²=142M with coordinate rescaling.

### **Strong Regularization**
```bash
python train_bioqic.py --experiment strong_tv
```
10× stronger TV for sharper boundaries (piecewise constant).

### **More Data**
```bash
python train_bioqic.py --experiment more_data
```
Uses 5000 points and 5000 iterations for potentially better reconstruction.

---

## 📈 Monitor Training Progress

Training prints detailed logs every 50 iterations:

```
────────────────────────────────────────────────────────────────────────────────
Iter   500 | Time: 0.234s | LR: 5.12e-03
────────────────────────────────────────────────────────────────────────────────
  Losses:
    Total:  2.345678e-04
    Data:   2.123456e-04 (MSE)
    BC:     1.234567e-05
    TV:     9.876543e-07
    L2:     0.000000e+00
  Optimization:
    Grad norm: 1.234e-02
    u MSE:     2.123e-04
  Stiffness (normalized):
    Range: [0.3456, 0.8765]
    Mean:  0.6123 ± 0.1234
  Stiffness (Pa):
    Range: [3456, 8765] Pa
    Mean:  6123 ± 1234 Pa
    (Target: [3000, 10000] Pa)
```

**What to look for:**
- ✅ **Data loss decreasing**: Should drop to ~1e-4 to 1e-6
- ✅ **Stiffness range**: Should approach [3000, 10000] Pa
- ✅ **Stable gradients**: Grad norm should stabilize (not explode)
- ⚠️ **No collapse**: μ range shouldn't shrink to narrow band

---

## 🔧 Customize Experiments

You can modify `EXPERIMENTS` dict in `train_bioqic.py`:

```python
'my_experiment': {
    'description': 'Custom configuration',
    'displacement_mode': 'z_component',  # or 'magnitude', '3_components'
    'boundary_strategy': 'actuator',     # or 'minimal', 'weighted'
    'stiffness_strategy': 'direct',      # or 'log', 'softplus'
    'physics_mode': 'effective',         # or 'physical'
    'mu_min': 0.2,
    'mu_max': 1.2,
    'data_weight': 50.0,
    'bc_weight': 50.0,
    'tv_weight': 0.001,
    'l2_weight': 0.0,
    'n_wave_neurons': 100,
    'lr': 0.01,
    'iterations': 3000,
    'subsample': 1000
}
```

Then run:
```bash
python train_bioqic.py --experiment my_experiment
```

---

## 🐛 Troubleshooting

### **Issue: Training very slow**
- **Solution**: Reduce `subsample` (e.g., 500 instead of 1000)
- **Solution**: Reduce `n_wave_neurons` (e.g., 60 instead of 100)

### **Issue: Stiffness collapses to narrow range**
- **Check**: `mu_min` and `mu_max` match data (should be [0.2, 1.2])
- **Try**: Increase `tv_weight` for stronger regularization
- **Try**: Increase `data_weight` relative to `bc_weight`

### **Issue: Data loss plateau**
- **Check**: Not over-constraining with too many BC points
- **Try**: Switch from 'actuator' to 'minimal' boundary strategy
- **Try**: Increase `data_weight` to 100+

### **Issue: Gradient explosion**
- **Check**: Learning rate not too high
- **Try**: Lower `lr` to 0.005 or 0.001
- **Try**: Use 'physical' mode with proper coordinate scaling

---

## 📊 Compare Results

After running multiple experiments:

```python
import json
import matplotlib.pyplot as plt

# Load histories
experiments = ['baseline', 'actuator', 'vector']
histories = {}

for exp in experiments:
    with open(f'outputs/{exp}/training_history.json') as f:
        histories[exp] = json.load(f)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for exp in experiments:
    h = histories[exp]
    axes[0].semilogy(h['iteration'], h['loss_total'], label=exp)
    axes[1].plot(h['iteration'], h['mu_mean'], label=exp)
    axes[2].plot(h['iteration'], h['u_mse'], label=exp)

axes[0].set_title('Total Loss')
axes[1].set_title('Mean Stiffness')
axes[2].set_title('Displacement MSE')

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_comparison.png')
```

---

## 🎓 Theory Confirmation

**Q: Is this the same theory as the approach folder?**

**A: YES! Exact same gradient-based optimization:**

```
Helmholtz PDE: ∇·(μ∇u) + ρω²u = 0

PIELM approach:
1. u(x) = Σ C_i φ_i(x)  [wave basis expansion]
2. Solve: min_C ‖-μ∇²u - ρω²u‖² + BC penalty
3. C* = solve(H, b)  [differentiable linear solver]
4. Backprop: ∂Loss/∂μ via autograd through solver
5. Update: μ_net ← μ_net - α·∇Loss
```

**What changed:**
- ✅ Better boundary detection (physics-informed)
- ✅ Flexible stiffness bounds (match data)
- ✅ Multiple displacement modes (richer constraints)
- ✅ Physics scaling options (physical vs effective)
- ✅ More visualization and logging

**Core theory: IDENTICAL**

---

## 📚 Next Steps

1. **Run baseline first** to establish reference
2. **Compare with actuator** to see impact of physics-informed BCs
3. **Try vector mode** to use full displacement information
4. **Analyze results** using provided visualizations
5. **Iterate**: Adjust hyperparameters based on observations

---

## 💡 Tips for Success

1. **Start simple**: Always run `baseline` first
2. **Check visualizations**: Look at μ distribution and u fit quality
3. **Monitor stiffness range**: Should cover [3000, 10000] Pa
4. **Be patient**: Training takes 5-10 minutes per experiment
5. **Compare experiments**: Run multiple configs to find best

---

## 🚨 Important Notes

- **Data location**: Assumes `../data/processed/phase1_box/` exists
- **Output location**: Creates `outputs/` in current directory
- **GPU**: Automatically uses CUDA if available (much faster!)
- **Reproducibility**: Fixed seed (42) for consistent results

---

## ✅ Ready to Go!

```bash
# Run your first experiment NOW:
python train_bioqic.py --experiment baseline

# Watch the progress and visualizations
# Results will be in outputs/baseline/
```

**Good luck! 🎉**
