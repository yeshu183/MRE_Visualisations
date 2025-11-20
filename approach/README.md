# PIELM-MRE: Physics-Informed Extreme Learning Machine for MRE Inverse Problems# Differentiable PIELM for MRE Inversion



**Clean, validated implementation with comprehensive testing and documentation.**This framework solves the **Inverse Problem** for Magnetic Resonance Elastography (MRE) using **Analysis-by-Synthesis** (Forward Optimization). It utilizes a custom Differentiable Linear Solver to backpropagate gradients through the PIELM matrix inversion.



------



## 📁 Folder Structure## 1. Mathematical Formulation



```### The Forward Model (PIELM)

approach/We approximate the wave field $u(x)$ using a neural network with **fixed** basis functions $\phi(x)$ and **trainable** output weights $C$:

├── core/                      # Core reusable modules$$u(x) = \sum_{j=1}^{N} C_j \phi_j(x) = \Phi(x) C$$

│   ├── __init__.py           # Public API exports

│   ├── data_generators.py    # Synthetic data generationTo solve for the weights $C$, we minimize the PDE residual. [cite_start]As derived in the reference document[cite: 26, 27], this linear least-squares problem is formulated as:

│   ├── solver.py             # Training and evaluation

│   └── visualization.py      # Plotting utilities$$J(C) = \frac{1}{2} || H C - b ||^2$$

│

├── examples/                  # Example scripts (ready to run!)Where:

│   ├── example_gaussian_bump.py* [cite_start]**H (The Design Matrix):** Stacks the PDE operator applied to basis functions and boundary conditions[cite: 86, 97].

│   ├── example_multiple_inclusions.py    $$H = \begin{bmatrix} L[\phi(x_1)] \\ \vdots \\ L[\phi(x_N)] \\ \phi(x_{boundary}) \end{bmatrix}$$

│   ├── example_step_function.py* [cite_start]**b (The Target Vector):** Contains PDE targets (zeros) and boundary values[cite: 91].

│   └── run_all_examples.py   # Batch runner* [cite_start]**Normal Equation:** Minimizing $J(C)$ leads to the solution[cite: 133, 158]:

│    $$C = (H^T H)^{-1} H^T b$$

├── tests/                     # Validation and unit tests

│   ├── test_core_components.py      # ✅ Core validation suite---

│   ├── test_gradient_flow.py        # Gradient diagnostics

│   ├── test_mu_network.py           # Network capability tests## 2. The Custom Gradient Derivation ($\frac{\partial C}{\partial H}$)

│   ├── test_data_constraints.py     # BC vs Data comparison

│   ├── test_data_only.py            # Data-only approachTo optimize the stiffness $\mu$, we must backpropagate through the linear solve. Standard autodiff is slow ($O(N^3)$) and unstable. We use an analytical matrix derivative.

│   ├── test_interior_weighting.py   # Interior weighting study

│   └── ...### Derivation Steps

│Starting from the Normal Equation:

├── debug/                     # Debugging and diagnostic tools$$(H^T H) C = H^T b$$

│   ├── debug_forward_solve.py       # Forward solve analysis

│   ├── debug_forward_data_fit.py    # Data fitting tests**Step 1: Differentiate both sides** (Product Rule $d(XY) = dX Y + X dY$):

│   ├── diagnose_bc_scaling.py       # BC weight analysis$$d(H^T H) C + (H^T H) dC = d(H^T) b$$

│   └── diagnose_data_scaling.py     # Data weight analysis

│**Step 2: Expand the differential $d(H^T H)$:**

├── docs/                      # Documentation$$(dH^T H + H^T dH) C + (H^T H) dC = dH^T b$$

│   ├── MODULAR_README.md            # Usage guide

│   ├── VALIDATION_REPORT.md         # Comprehensive validation results**Step 3: Isolate the term with $dC$:**

│   └── README.md                    # This file$$(H^T H) dC = dH^T b - (dH^T H + H^T dH) C$$

│$$(H^T H) dC = dH^T b - dH^T H C - H^T dH C$$

├── results/                   # Generated plots and outputs

│**Step 4: Factor out $dH^T$:**

├── models.py                  # Neural network architectures$$(H^T H) dC = dH^T (b - H C) - H^T dH C$$

├── pielm_solver.py           # Differentiable PIELM with custom backward

├── config_forward.json       # Configuration parameters**Step 5: Introduce the Residual $r$:**

└── main_mre.py               # (Optional) Main entry pointDefine the residual vector $r = H C - b$. Therefore, $(b - H C) = -r$.

```$$(H^T H) dC = - dH^T r - H^T dH C$$



---**Step 6: Final Expression:**

Let $A = H^T H$. Multiply by $A^{-1}$:

## 🚀 Quick Start

$$dC = - (H^T H)^{-1} \left[ dH^T r + H^T dH C \right]$$

### Run All Examples

```bashThis formula allows us to compute the gradients efficiently without storing the computational graph of the inversion.

python approach/examples/run_all_examples.py

```---



### Run Individual Examples## 3. The Optimization Chain Rule ($\frac{d Loss}{d \mu}$)

```bash

python approach/examples/example_gaussian_bump.pyWe update the stiffness estimate $\mu(x)$ by minimizing the data mismatch loss $L_{data} = || u_{pred} - u_{meas} ||^2$. The gradient flows as follows:

python approach/examples/example_multiple_inclusions.py

python approach/examples/example_step_function.py$$\frac{\partial L}{\partial \mu} = \underbrace{\frac{\partial L}{\partial u_{pred}}}_{\text{Data Error}} \cdot \underbrace{\frac{\partial u_{pred}}{\partial C}}_{\text{Basis}} \cdot \underbrace{\frac{\partial C}{\partial H}}_{\text{Solver Grad}} \cdot \underbrace{\frac{\partial H}{\partial \mu}}_{\text{Physics}}$$

```

### Component Breakdown

### Validate Core Components

```bash1.  **Data Mismatch:**

python approach/tests/test_core_components.py    $$\frac{\partial L}{\partial u} = 2(u_{pred} - u_{meas})$$

```

2.  **Wave Reconstruction:**

---    Since $u = \Phi C$, the gradient is the basis functions themselves:

    $$\frac{\partial u}{\partial C} = \Phi(x)$$

## ⚙️ Configuration

3.  **Solver Gradient:**

Edit `config_forward.json` to tune hyperparameters:    Calculated via the custom derivation in Section 2.



```json4.  **Physics Gradient ($\frac{\partial H}{\partial \mu}$):**

{    The matrix $H$ is constructed from the PDE: $\mu \nabla^2 \phi + \rho \omega^2 \phi$.

  "n_points": 100,           // Spatial discretization    Since this is linear in $\mu$, the derivative is simply the Laplacian of the basis functions:

  "n_wave_neurons": 60,      // Wave basis functions    $$\frac{\partial H}{\partial \mu} = \nabla^2 \Phi$$

  "iterations": 5000,        // Training iterations    *(Note: This assumes local homogeneity. If $\nabla \mu$ terms are included, additional terms for $\nabla \Phi$ appear).*

  "lr": 0.005,              // Learning rate

  "bc_weight": 200.0,       // Boundary condition weight (CRITICAL!)---

  "data_weight": 0.0,       // Data constraint weight (0 for inverse problem)

  "tv_weight": 0.001,       // Total variation regularization## Usage

  "seed": 0                 // Random seed

}```python

```# Forward Pass (Analysis)

mu_guess = generator(coords)

### **Key Parameters:**H = construct_physics_matrix(mu_guess, basis_derivs)

C = DifferentiablePIELM.apply(H, b) # Solves (H^T H)C = H^T b

- **`bc_weight`**: Must be 100-200 for unique solution (see `VALIDATION_REPORT.md`)u_pred = basis @ C

- **`data_weight`**: Keep at 0 for inverse problems (data constraints weaken gradients)

- **`tv_weight`**: Use 0.001-0.002 for smooth cases, higher for discontinuities# Backward Pass (Synthesis/Optimization)

loss = MSE(u_pred, u_measured)

---loss.backward() # Triggers the custom chain rule above

optimizer.step() # Updates mu_guess
## ✅ Validation Status

### Core Components (All Validated ✓)
- ✅ Forward solver: Deterministic, consistent
- ✅ Custom backward pass: Mathematically correct (1e-10 error)
- ✅ Gradient flow: Working through entire chain
- ✅ Loss sensitivity: Strong optimization signal

### Test Results
- ✅ **Gaussian Bump**: Works well with `bc_weight=200`
- ✅ **Multiple Inclusions**: Reasonable reconstruction
- ⚠️ **Step Function**: Acceptable (inherent smoothing limitation)

See `docs/VALIDATION_REPORT.md` for detailed analysis.

---

## 🔬 Key Findings from Testing

### 1. Boundary Condition Weighting is Critical

**Problem:** With only 2 BC rows vs 100 PDE rows, need `bc_weight ≈ 100-200` to balance contributions.

```
bc_weight = 2:    BC contributes 0.0004% → Underconstrained, fails
bc_weight = 200:  BC contributes 0.4%    → Unique solution, works ✅
```

See: `tests/diagnose_bc_scaling.py`

### 2. Data Constraints Suppress Gradients

**Problem:** Data rows don't depend on mu → weaken gradient signal for inverse problem.

```
bc_weight=200, data_weight=0:   Gradient = 2.15e-04  ✅
bc_weight=200, data_weight=10:  Gradient = 4.84e-05  ❌ (4× weaker)
```

**Conclusion:** For inverse problems, use `bc_weight=200, data_weight=0`.

See: `tests/test_data_constraints.py`, `debug/diagnose_data_scaling.py`

### 3. Interior Weighting Can Provide Constraints

**But:** Gradients are still weakened (10-30×) compared to BC approach.

**For Real MRE:** Use hybrid approach with estimated BCs from boundary measurements + moderate interior weighting.

See: `tests/test_interior_weighting.py`

### 4. Fourier Features Prevent Mode Collapse

**Problem:** Simple networks collapse to constant mu (mode collapse).

**Solution:** Use random Fourier features as input:
```python
[sin(2πx), cos(2πx), sin(4πx), cos(4πx), ...]
```

Network can now learn spatial variation.

See: `tests/test_mu_network.py`

---

## 📊 Recommended Settings

### For Synthetic Tests (Known BCs)
```json
{
  "bc_weight": 200,
  "data_weight": 0,
  "tv_weight": 0.001,
  "lr": 0.005,
  "iterations": 5000
}
```

### For Real MRE (Estimated BCs)
```json
{
  "bc_weight": 50-100,     // From boundary measurements
  "data_weight": 1-2,      // Moderate interior weighting
  "tv_weight": 0.002,      // Stronger regularization
  "lr": 0.005,
  "iterations": 10000      // More iterations (weaker gradients)
}
```

---

## 🏗️ Architecture

### Neural Network (Stiffness Generator)
- **Input:** 1D spatial coordinates x ∈ [0, 1]
- **Embedding:** Random Fourier features (20D)
- **Architecture:** 2 layers × 64 units, Tanh activation
- **Output:** mu(x) ∈ [0.7, 6.0] via softplus + clamping
- **Initialization:** Xavier uniform (gain=1.0)

### Forward Model (PIELM)
- **Basis:** Random Fourier features (sine waves)
- **PDE:** μ(x)∇²u(x) + ρω²u(x) = 0
- **Solver:** Differentiable PIELM with custom backward pass
- **Regularization:** Adaptive Cholesky (8 attempts, 5× escalation)

---

## 📖 Documentation

- **`docs/MODULAR_README.md`**: Detailed usage guide
- **`docs/VALIDATION_REPORT.md`**: Comprehensive test results and known issues
- **`docs/README.md`**: This overview

---

## 🧪 Testing

### Validation Suite
```bash
python approach/tests/test_core_components.py
```

Runs 7 validation tests:
1. Forward solver consistency
2. Gradient flow verification
3. Analytical gradient correctness
4. Loss sensitivity analysis
5. Network learning capability
6. Gradient magnitude under different conditions
7. Ill-posedness check

### Debugging Tools
```bash
python approach/debug/debug_forward_solve.py       # Analyze forward solve
python approach/debug/diagnose_bc_scaling.py       # BC weight analysis
python approach/debug/diagnose_data_scaling.py     # Data weight analysis
```

---

## 🎯 Next Steps

### For Synthetic Data (Current Status: ✅ Working)
1. ✅ Core components validated
2. ✅ Examples run successfully
3. ⚠️ Step function has inherent smoothing (acceptable)

### For Real MRE Data
1. Estimate BCs from boundary measurements or anatomical knowledge
2. Start with `bc_weight=50-100`, `data_weight=1-2`
3. Tune `tv_weight` based on expected smoothness
4. Monitor gradient norms - if too weak (<1e-5), increase `bc_weight`

---

## 📝 Citation

If you use this code, please cite the original PIELM paper and acknowledge the MRE formulation.

---

## 🐛 Known Issues

1. **Step function reconstruction**: Smooth networks can't capture sharp transitions (use TV regularization)
2. **Gradient magnitude**: With weak BCs or heavy data weighting, gradients become very small
3. **Ill-posedness**: Problem is moderately ill-posed (Δmu/Δu ≈ 2e-4)

See `docs/VALIDATION_REPORT.md` for detailed analysis and workarounds.

---

## 📧 Contact

For questions or issues, see the main repository README.
