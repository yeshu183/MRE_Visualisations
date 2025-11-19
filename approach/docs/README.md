# Differentiable PIELM for MRE Inversion

This framework solves the **Inverse Problem** for Magnetic Resonance Elastography (MRE) using **Analysis-by-Synthesis** (Forward Optimization). It utilizes a custom Differentiable Linear Solver to backpropagate gradients through the PIELM matrix inversion.

---

## 1. Mathematical Formulation

### The Forward Model (PIELM)
We approximate the wave field $u(x)$ using a neural network with **fixed** basis functions $\phi(x)$ and **trainable** output weights $C$:
$$u(x) = \sum_{j=1}^{N} C_j \phi_j(x) = \Phi(x) C$$

To solve for the weights $C$, we minimize the PDE residual. [cite_start]As derived in the reference document[cite: 26, 27], this linear least-squares problem is formulated as:

$$J(C) = \frac{1}{2} || H C - b ||^2$$

Where:
* [cite_start]**H (The Design Matrix):** Stacks the PDE operator applied to basis functions and boundary conditions[cite: 86, 97].
    $$H = \begin{bmatrix} L[\phi(x_1)] \\ \vdots \\ L[\phi(x_N)] \\ \phi(x_{boundary}) \end{bmatrix}$$
* [cite_start]**b (The Target Vector):** Contains PDE targets (zeros) and boundary values[cite: 91].
* [cite_start]**Normal Equation:** Minimizing $J(C)$ leads to the solution[cite: 133, 158]:
    $$C = (H^T H)^{-1} H^T b$$

---

## 2. The Custom Gradient Derivation ($\frac{\partial C}{\partial H}$)

To optimize the stiffness $\mu$, we must backpropagate through the linear solve. Standard autodiff is slow ($O(N^3)$) and unstable. We use an analytical matrix derivative.

### Derivation Steps
Starting from the Normal Equation:
$$(H^T H) C = H^T b$$

**Step 1: Differentiate both sides** (Product Rule $d(XY) = dX Y + X dY$):
$$d(H^T H) C + (H^T H) dC = d(H^T) b$$

**Step 2: Expand the differential $d(H^T H)$:**
$$(dH^T H + H^T dH) C + (H^T H) dC = dH^T b$$

**Step 3: Isolate the term with $dC$:**
$$(H^T H) dC = dH^T b - (dH^T H + H^T dH) C$$
$$(H^T H) dC = dH^T b - dH^T H C - H^T dH C$$

**Step 4: Factor out $dH^T$:**
$$(H^T H) dC = dH^T (b - H C) - H^T dH C$$

**Step 5: Introduce the Residual $r$:**
Define the residual vector $r = H C - b$. Therefore, $(b - H C) = -r$.
$$(H^T H) dC = - dH^T r - H^T dH C$$

**Step 6: Final Expression:**
Let $A = H^T H$. Multiply by $A^{-1}$:

$$dC = - (H^T H)^{-1} \left[ dH^T r + H^T dH C \right]$$

This formula allows us to compute the gradients efficiently without storing the computational graph of the inversion.

---

## 3. The Optimization Chain Rule ($\frac{d Loss}{d \mu}$)

We update the stiffness estimate $\mu(x)$ by minimizing the data mismatch loss $L_{data} = || u_{pred} - u_{meas} ||^2$. The gradient flows as follows:

$$\frac{\partial L}{\partial \mu} = \underbrace{\frac{\partial L}{\partial u_{pred}}}_{\text{Data Error}} \cdot \underbrace{\frac{\partial u_{pred}}{\partial C}}_{\text{Basis}} \cdot \underbrace{\frac{\partial C}{\partial H}}_{\text{Solver Grad}} \cdot \underbrace{\frac{\partial H}{\partial \mu}}_{\text{Physics}}$$

### Component Breakdown

1.  **Data Mismatch:**
    $$\frac{\partial L}{\partial u} = 2(u_{pred} - u_{meas})$$

2.  **Wave Reconstruction:**
    Since $u = \Phi C$, the gradient is the basis functions themselves:
    $$\frac{\partial u}{\partial C} = \Phi(x)$$

3.  **Solver Gradient:**
    Calculated via the custom derivation in Section 2.

4.  **Physics Gradient ($\frac{\partial H}{\partial \mu}$):**
    The matrix $H$ is constructed from the PDE: $\mu \nabla^2 \phi + \rho \omega^2 \phi$.
    Since this is linear in $\mu$, the derivative is simply the Laplacian of the basis functions:
    $$\frac{\partial H}{\partial \mu} = \nabla^2 \Phi$$
    *(Note: This assumes local homogeneity. If $\nabla \mu$ terms are included, additional terms for $\nabla \Phi$ appear).*

---

## Usage

```python
# Forward Pass (Analysis)
mu_guess = generator(coords)
H = construct_physics_matrix(mu_guess, basis_derivs)
C = DifferentiablePIELM.apply(H, b) # Solves (H^T H)C = H^T b
u_pred = basis @ C

# Backward Pass (Synthesis/Optimization)
loss = MSE(u_pred, u_measured)
loss.backward() # Triggers the custom chain rule above
optimizer.step() # Updates mu_guess