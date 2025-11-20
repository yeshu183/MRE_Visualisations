# PIELM Formulation: Comprehensive Derivation

**Based on Document:** `PIELM_formulation_Rishi.pdf`

## 1. Problem Definition
We aim to solve a linear Boundary Value Problem (BVP) of the form:
$$Lu(x) = 0, \quad x \in [x_l, x_r]$$
Subject to Dirichlet boundary conditions:
$$u(x_l) = u_l, \quad u(x_r) = u_r$$

## 2. Network Architecture (ELM)
The solution $u(x)$ is approximated by a single-hidden-layer feedforward network.
* **Input Layer:** $x$ (1 neuron)
* **Hidden Layer:** $\phi(x)$ ($N_\phi$ neurons). The weights $W$ and biases are **randomly initialized and fixed**.
* **Output Layer:** Linear weights $C$ ($N_\phi$ trainable parameters).

$$u(x) = \sum_{j=1}^{N_\phi} C_j \phi_j(x) = C^T \phi(x)$$

* Dimension of $C$: $(N_\phi \times 1)$
* Dimension of $\phi(x)$: $(N_\phi \times 1)$

---

## 3. Loss Function Derivation
The objective is to minimize the squared error residuals for both the differential equation and the boundaries.

$$J(C) = \underbrace{\frac{1}{2}\sum_{i=1}^{N_x} (Lu(x_i))^2}_{\text{PDE Residual}} + \underbrace{\frac{1}{2}(u(x_l) - u_l)^2 + \frac{1}{2}(u(x_r) - u_r)^2}_{\text{Boundary Errors}}$$

Where $N_x$ is the number of collocation points in the domain.

---

## 4. Gradient Computation (Normal Equations)
To find the optimal $C$, we set $\frac{\partial J}{\partial C} = 0$.

### A. Gradient of the PDE Term
The PDF (Source 196-199) derives this using the chain rule:
$$\frac{\partial}{\partial C} \left[ \frac{1}{2} \sum (C^T L\phi(x_i))^2 \right] = \sum_{i=1}^{N_x} (C^T L\phi(x_i)) \cdot L\phi(x_i)$$
Rearranging the scalar product:
$$= \sum_{i=1}^{N_x} L\phi(x_i) (L\phi(x_i)^T C)$$
$$= \left( \sum_{i=1}^{N_x} L\phi(x_i) L\phi(x_i)^T \right) C$$

### B. Gradient of Boundary Terms
(Source 200-207)
At boundary $x_l$:
$$\frac{\partial}{\partial C} \frac{1}{2}(C^T \phi(x_l) - u_l)^2 = \phi(x_l)(C^T \phi(x_l) - u_l) = \phi(x_l)\phi(x_l)^T C - \phi(x_l)u_l$$

Similarly for $x_r$.

### C. Combined System
Combining all gradients and setting to zero:
$$\left( \sum_{i=1}^{N_x} L\phi(x_i) L\phi(x_i)^T + \phi(x_l)\phi(x_l)^T + \phi(x_r)\phi(x_r)^T \right) C = \phi(x_l)u_l + \phi(x_r)u_r$$

---

## 5. Matrix Construction & Dimensions

To implement this numerically, we define the matrices explicitly (Source 211-230).

### The Operator Matrix $\Psi$
Let $\Psi$ be the matrix of the operator applied to all neurons at all collocation points.
$$\Psi = \begin{bmatrix} 
(L\phi(x_1))^T \\ 
(L\phi(x_2))^T \\ 
\vdots \\ 
(L\phi(x_{N_x}))^T 
\end{bmatrix}$$
* **Dimension:** $(N_x \times N_\phi)$
* Note: The PDF denotes this as stacking row vectors $\psi_i^T$.

This allows us to write the summation as a matrix product:
$$\sum_{i=1}^{N_x} L\phi(x_i) L\phi(x_i)^T = \Psi^T \Psi$$
* **Dimension:** $(N_\phi \times N_\phi)$

### The Unified Linear System (H matrix)
We stack the PDE constraints and boundary constraints into a single overdetermined system (Source 252-264).

**The Data Matrix H:**
$$H = \begin{bmatrix} 
\Psi \\ 
\phi(x_l)^T \\ 
\phi(x_r)^T 
\end{bmatrix}$$
* $\Psi$: $(N_x \times N_\phi)$
* Boundaries: $(2 \times N_\phi)$
* **Total Dimension:** $((N_x + 2) \times N_\phi)$

**The Target Vector b:**
$$b = \begin{bmatrix} 
0 \\ 
\vdots \\ 
0 \\ 
u_l \\ 
u_r 
\end{bmatrix}$$
* Zeros: $(N_x \times 1)$ (Because $Lu=0$)
* Boundaries: $(2 \times 1)$
* **Total Dimension:** $((N_x + 2) \times 1)$

---

## 6. Solution
The minimization of the loss function $J(C)$ is mathematically equivalent to solving the linear system $HC = b$ using the **Least Squares** method.

$$C = (H^T H)^{-1} H^T b$$

or using the Moore-Penrose Pseudo-Inverse:

$$C = H^{\dagger} b$$

Once $C$ is found, the solution is:
$$u(x) = C^T \phi(x)$$

**PIELM Architecture Diagram**



### **Explanation of the Diagram Components:**

* **Input Layer ($x$):**
    * The network takes a spatial coordinate $x$ as input.
    * In the diagram, this is the yellow circle on the left labeled "$x$".
    * There is also a bias term (often denoted as "1" or "b") connected to the hidden layer, shown in the purple circle.

* **Hidden Layer ($\phi_1, \phi_2, ..., \phi_{N_\phi}$):**
    * This layer consists of $N_\phi$ neurons (yellow circles in the middle column).
    * **Input Weights ($W$):** The connections from the input $x$ to these neurons (white arrows) have weights $W_1, W_2, ..., W_{N_\phi}$.
    * **Biases ($b$):** The connections from the bias node to these neurons (dashed purple arrows) have weights $b_1, b_2, ..., b_{N_\phi}$.
    * **Key Feature:** In PIELM, these input weights ($W$) and biases ($b$) are **randomly assigned and fixed**. They are *not* updated during training.
    * **Activation:** Each neuron applies a non-linear activation function (e.g., sigmoid, tanh) to produce an output $\phi_j(x)$.

* **Output Layer ($u(x)$):**
    * The final output is a weighted sum of the hidden layer activations.
    * **Output Weights ($C$):** The connections from the hidden neurons to the output (orange/red arrows) have weights $C_1, C_2, ..., C_{N_\phi}$.
    * **Trainable Parameters:** These $C$ weights are the **only** parameters that are calculated. As the derivation showed, they are solved analytically using the linear system $(H^T H)C = H^T b$.
    * **Result:** The final calculation is $u(x) = \sum_{j=1}^{N_\phi} C_j \phi_j(x)$.