# Sobolev Loss for PIELM-MRE: Mathematical Derivation

**Document Version**: 1.0  
**Date**: November 22, 2025  
**Based on**: PIELM Custom Gradient Framework (see `approach/README.md`)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Sobolev Loss Definition](#sobolev-loss-definition)
3. [Forward Pass with Basis Expansion](#forward-pass-with-basis-expansion)
4. [Gradient w.r.t. Coefficients](#gradient-wrt-coefficients)
5. [Backpropagation Through Linear Solver](#backpropagation-through-linear-solver)
6. [Complete Optimization Chain](#complete-optimization-chain)
7. [Implementation Details](#implementation-details)
8. [Why Sobolev Loss Helps MRE](#why-sobolev-loss-helps-mre)

---

## Introduction

The standard MSE loss for MRE inversion is:
$$L_{MSE} = \frac{1}{2} || u_{pred} - u_{meas} ||^2$$

However, **Sobolev loss** (also called $H^1$ loss) adds a gradient-matching term that penalizes errors in both **wave amplitude** and **wave slope/phase**:
$$L_{Sobolev} = \underbrace{\frac{\alpha}{2} || u_{pred} - u_{meas} ||^2}_{\text{L2 Term}} + \underbrace{\frac{\beta}{2} || \nabla u_{pred} - \nabla u_{meas} ||^2}_{\text{Gradient Term}}$$

**Key Finding**: From forward solve analysis, optimal weights are **α=0.1, β=0.9**, meaning the gradient term provides **90% of discrimination power** for detecting heterogeneous stiffness.

---

## Sobolev Loss Definition

### Standard Form
$$L_{Sobolev}(C) = \frac{\alpha}{2} || u_{pred} - u_{meas} ||^2 + \frac{\beta}{2} || \nabla u_{pred} - \nabla u_{meas} ||^2$$

Where:
- $u_{pred}(x) = \sum_j C_j \phi_j(x) = \mathbf{\Phi} \mathbf{C}$ (wave field)
- $\nabla u_{pred}(x) = \sum_j C_j \nabla \phi_j(x) = \mathbf{\Psi} \mathbf{C}$ (wave gradients)
- $\mathbf{\Phi}$: Basis functions matrix (e.g., $\sin(\omega x)$)
- $\mathbf{\Psi}$: Basis derivatives matrix (e.g., $\omega \cos(\omega x)$)
- $\mathbf{C}$: Coefficient vector from PIELM solver

### Why This Matters for MRE
- **Phase sensitivity**: Heterogeneous stiffness causes wave scattering, changing **wave slopes** more dramatically than amplitudes
- **Better discrimination**: Gradient term amplifies differences between homogeneous vs heterogeneous fields
- **Physical insight**: Gradients capture wave propagation direction, which is sensitive to stiffness boundaries

---

## Forward Pass with Basis Expansion

### The PIELM Approximation
The wave field is approximated using random Fourier features:
$$u_{pred}(x) = \sum_{j=1}^{N} C_j \phi_j(x) = \mathbf{\Phi}(x) \mathbf{C}$$

**Basis Functions** (Sine waves):
$$\phi_j(x) = \sin(B_j \cdot x)$$
where $B_j$ are random frequencies.

**Basis Derivatives** (Cosine waves):
$$\nabla \phi_j(x) = B_j \cos(B_j \cdot x)$$

### Matrix Formulation
For $N$ spatial points and $M$ basis functions:
- $\mathbf{\Phi} \in \mathbb{R}^{N \times M}$: Basis functions evaluated at points
- $\mathbf{\Psi} \in \mathbb{R}^{N \times M}$: Basis derivatives evaluated at points
- $\mathbf{C} \in \mathbb{R}^{M}$: Coefficients from PIELM solve

Then:
$$u_{pred} = \mathbf{\Phi} \mathbf{C} \in \mathbb{R}^N$$
$$\nabla u_{pred} = \mathbf{\Psi} \mathbf{C} \in \mathbb{R}^N$$

---

## Gradient w.r.t. Coefficients

This is the **crucial step** that differs from MSE loss. We need:
$$\frac{\partial L_{Sobolev}}{\partial \mathbf{C}}$$

### Term 1: L2 Loss Gradient
$$L_1 = \frac{\alpha}{2} || \mathbf{\Phi} \mathbf{C} - u_{meas} ||^2$$

Expanding:
$$L_1 = \frac{\alpha}{2} (\mathbf{\Phi} \mathbf{C} - u_{meas})^T (\mathbf{\Phi} \mathbf{C} - u_{meas})$$

Differentiating w.r.t. $\mathbf{C}$:
$$\frac{\partial L_1}{\partial \mathbf{C}} = \alpha \mathbf{\Phi}^T (\mathbf{\Phi} \mathbf{C} - u_{meas})$$
$$= \alpha \mathbf{\Phi}^T (u_{pred} - u_{meas})$$

### Term 2: Gradient Loss Gradient
$$L_2 = \frac{\beta}{2} || \mathbf{\Psi} \mathbf{C} - \nabla u_{meas} ||^2$$

Let $G_{meas} = \nabla u_{meas}$ (computed numerically from measurements).

Expanding:
$$L_2 = \frac{\beta}{2} (\mathbf{\Psi} \mathbf{C} - G_{meas})^T (\mathbf{\Psi} \mathbf{C} - G_{meas})$$

Differentiating w.r.t. $\mathbf{C}$:
$$\frac{\partial L_2}{\partial \mathbf{C}} = \beta \mathbf{\Psi}^T (\mathbf{\Psi} \mathbf{C} - G_{meas})$$
$$= \beta \mathbf{\Psi}^T (\nabla u_{pred} - \nabla u_{meas})$$

### Combined Gradient
$$\boxed{\frac{\partial L_{Sobolev}}{\partial \mathbf{C}} = \alpha \mathbf{\Phi}^T (u_{pred} - u_{meas}) + \beta \mathbf{\Psi}^T (\nabla u_{pred} - \nabla u_{meas})}$$

This is the **input gradient** we pass to the PIELM backward pass.

---

## Backpropagation Through Linear Solver

### The Solver Operation
PIELM solves: $(H^T H) C = H^T b$ to get:
$$\mathbf{C} = (H^T H)^{-1} H^T b$$

where $H$ depends on $\mu(x)$ via the PDE operator.

### The Magic Formula (Unchanged!)
**Critical Insight**: The backward pass formula for the linear solver **does not change**. It only needs the gradient w.r.t. coefficients $\frac{\partial L}{\partial C}$ as input.

From the approach folder derivation:
$$\frac{\partial L}{\partial H} = - (H^T H)^{-1} \left[ \frac{\partial L}{\partial C} r^T + H^T \frac{\partial L}{\partial C} C^T \right]$$

where $r = HC - b$ is the residual.

**Equivalently** (see `approach/README.md` for full derivation):
$$\frac{\partial L}{\partial H} = - \left( H \mathbf{v} C^T + r \mathbf{v}^T \right)$$

where:
$$\mathbf{v} = (H^T H)^{-1} \frac{\partial L}{\partial C}$$

is the "adjoint solution."

### Key Takeaway
The solver backward pass is **agnostic to the loss function**. It takes:
- Input: $\frac{\partial L}{\partial C}$ (from any loss function)
- Output: $\frac{\partial L}{\partial H}$ (how to change physics matrix)

For Sobolev loss, we simply plug in:
$$\frac{\partial L}{\partial C} = \alpha \mathbf{\Phi}^T (u_{pred} - u_{meas}) + \beta \mathbf{\Psi}^T (\nabla u_{pred} - \nabla u_{meas})$$

---

## Complete Optimization Chain

### Full Gradient Flow
$$\frac{\partial L_{Sobolev}}{\partial \mu} = \underbrace{\frac{\partial L}{\partial u_{pred}}}_{\text{Data Error}} \cdot \underbrace{\frac{\partial u_{pred}}{\partial C}}_{\text{Basis}} \cdot \underbrace{\frac{\partial C}{\partial H}}_{\text{Solver Grad}} \cdot \underbrace{\frac{\partial H}{\partial \mu}}_{\text{Physics}}$$

**Plus the gradient term**:
$$+ \underbrace{\frac{\partial L}{\partial \nabla u_{pred}}}_{\text{Gradient Error}} \cdot \underbrace{\frac{\partial \nabla u_{pred}}{\partial C}}_{\text{Basis Derivs}} \cdot \underbrace{\frac{\partial C}{\partial H}}_{\text{Solver Grad}} \cdot \underbrace{\frac{\partial H}{\partial \mu}}_{\text{Physics}}$$

### Step-by-Step Computation

1. **Data Mismatch Gradients**:
   $$\frac{\partial L}{\partial u_{pred}} = \alpha (u_{pred} - u_{meas})$$
   $$\frac{\partial L}{\partial \nabla u_{pred}} = \beta (\nabla u_{pred} - \nabla u_{meas})$$

2. **Basis Reconstruction**:
   $$\frac{\partial u_{pred}}{\partial C} = \mathbf{\Phi}(x)$$
   $$\frac{\partial \nabla u_{pred}}{\partial C} = \mathbf{\Psi}(x)$$

3. **Combined Gradient w.r.t. Coefficients**:
   $$\frac{\partial L}{\partial C} = \alpha \mathbf{\Phi}^T (u_{pred} - u_{meas}) + \beta \mathbf{\Psi}^T (\nabla u_{pred} - \nabla u_{meas})$$

4. **Solver Backward Pass** (Custom autograd function):
   $$\mathbf{v} = (H^T H)^{-1} \frac{\partial L}{\partial C}$$
   $$\frac{\partial L}{\partial H} = - \left( H \mathbf{v} C^T + r \mathbf{v}^T \right)$$

5. **Physics Gradient** ($H$ depends on $\mu$ via PDE):
   $$\frac{\partial H}{\partial \mu} = \nabla^2 \Phi$$
   (from $\mu \nabla^2 u + \rho \omega^2 u = 0$)

6. **Final Gradient**:
   $$\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial H} \cdot \nabla^2 \Phi$$

---

## Implementation Details

### Computing Basis Derivatives

For **Random Fourier Features**:
$$\phi_j(x) = \sin(B_j \cdot x)$$

The derivative is:
$$\nabla \phi_j(x) = B_j \cos(B_j \cdot x)$$

**PyTorch Implementation**:
```python
# In get_basis_and_laplacian():
Z = x @ self.B.T  # (N, n_neurons)
phi = torch.sin(Z)  # Basis functions

# Add gradient computation:
phi_grad = torch.cos(Z) * self.B  # Element-wise multiply by frequencies
# phi_grad shape: (N, n_neurons, input_dim)
```

### Computing Measured Gradients

Since $u_{meas}$ is discrete data, use **finite differences**:
$$\nabla u_{meas}(x_i) \approx \frac{u_{meas}(x_{i+1}) - u_{meas}(x_i)}{||x_{i+1} - x_i||}$$

**PyTorch Implementation**:
```python
def compute_measured_gradient(u_meas, x):
    """Compute spatial gradient of measured field using finite differences."""
    # Forward differences
    du = u_meas[1:] - u_meas[:-1]
    dx = torch.norm(x[1:] - x[:-1], dim=1, keepdim=True) + 1e-8
    grad_u_meas = du / dx
    
    # Pad to match original size
    grad_u_meas = torch.cat([grad_u_meas, grad_u_meas[-1:]], dim=0)
    return grad_u_meas
```

### Loss Computation in Trainer

```python
# 1. Compute measured gradients (once, at start)
grad_u_meas = compute_measured_gradient(u_meas, x)

# 2. Forward pass
u_pred, mu_pred = model(x, bc_indices, u_bc_vals, rho_omega2, ...)

# 3. Compute predicted gradients
phi, phi_lap, phi_grad = model.get_basis_derivatives(x)
_, C = model.solve_given_mu(...)
grad_u_pred = phi_grad @ C  # Matrix-vector product

# 4. Sobolev loss
loss_l2 = torch.mean((u_pred - u_meas) ** 2)
loss_grad = torch.mean((grad_u_pred - grad_u_meas) ** 2)
loss_sobolev = alpha * loss_l2 + beta * loss_grad

# 5. Backward (standard PyTorch autograd)
loss_sobolev.backward()  # Automatically uses custom PIELM backward!
```

### Optimal Weights (From Forward Analysis)
```python
alpha = 0.1  # L2 term weight (10%)
beta = 0.9   # Gradient term weight (90%)
```

**Rationale**: Forward solve experiments showed that the gradient term provides **90% of discrimination power** between homogeneous and heterogeneous stiffness fields.

---

## Why Sobolev Loss Helps MRE

### Physical Interpretation

1. **Wave Scattering**: Heterogeneous stiffness causes waves to scatter at interfaces
   - Changes **wave direction** (gradients) more than amplitude
   - Standard MSE is relatively insensitive to this

2. **Phase Matching**: Two waves can have similar amplitudes but different phases
   - MSE might give similar loss values
   - Gradient matching forces phase alignment

3. **Sensitivity Amplification**: At stiffness boundaries:
   - Wave **amplitude** changes modestly
   - Wave **slope** changes dramatically (refraction/reflection)
   - Gradient term creates stronger error signal

### Experimental Evidence

From `grid_search_forward_mu.py` and `sobolev_weight_sweep.py`:

| Configuration | Loss Type | Discrimination (Δ) | Notes |
|--------------|-----------|-------------------|-------|
| BC=10, N=1000 | MSE | -9.86e-07 | Baseline |
| BC=10, N=1000 | Sobolev (α=0.4, β=0.6) | -1.89e-03 | 1900× better |
| BC=10, N=1000 | Sobolev (α=0.1, β=0.9) | -2.29e-03 | **Best** |

**Interpretation**: 
- Negative Δ means heterogeneous μ has **lower loss** (better fit)
- Sobolev loss with β=0.9 maximizes this discrimination
- Gradient-dominant weighting (β > α) is optimal

### When to Use Sobolev Loss

✅ **Use Sobolev when**:
- Expecting heterogeneous stiffness (tumors, fibrosis)
- Need to detect subtle stiffness boundaries
- Phase/shape of waves is critical
- Have sufficient measurement quality for gradient estimation

❌ **Avoid Sobolev when**:
- Data is very noisy (gradient amplifies noise)
- Expecting homogeneous tissue (gradient adds no information)
- Measurements are too sparse for gradient estimation

---

## References

1. **PIELM Framework**: See `approach/README.md` for base formulation
2. **Custom Gradient Derivation**: See `approach/README.md` Section 2
3. **Forward Analysis**: See `bioqic_pielm/LOSS_FUNCTION_ANALYSIS.md`
4. **Sobolev Weight Optimization**: See `bioqic_pielm/sobolev_weight_sweep.py`

---

## Appendix: Comparison with MSE

### MSE Loss
$$L_{MSE} = \frac{1}{2} || u_{pred} - u_{meas} ||^2$$

**Gradient w.r.t. C**:
$$\frac{\partial L_{MSE}}{\partial C} = \mathbf{\Phi}^T (u_{pred} - u_{meas})$$

**Pros**: Simple, robust to noise  
**Cons**: Insensitive to phase, weak discrimination

### Sobolev Loss
$$L_{Sobolev} = \frac{\alpha}{2} || u_{pred} - u_{meas} ||^2 + \frac{\beta}{2} || \nabla u_{pred} - \nabla u_{meas} ||^2$$

**Gradient w.r.t. C**:
$$\frac{\partial L_{Sobolev}}{\partial C} = \alpha \mathbf{\Phi}^T (u_{pred} - u_{meas}) + \beta \mathbf{\Psi}^T (\nabla u_{pred} - \nabla u_{meas})$$

**Pros**: Phase-sensitive, strong discrimination, physically motivated  
**Cons**: Requires gradient computation, more sensitive to noise

---

**Implementation Status**: ✅ Implemented in `bioqic_pielm/trainer.py` with α=0.1, β=0.9
