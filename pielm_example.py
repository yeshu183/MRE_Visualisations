import numpy as np
import matplotlib.pyplot as plt

# This code solves the differential equation $u''(x) - 4u(x) = 0$ 
# with boundary conditions $u(0)=1, u(1)=e^2$ using the matrix construction 
# method described in your PDF.
# Result:The method solves the differential equation almost instantly 
# without iterative training (no epochs).

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d2_sigmoid(x):
    # Second derivative of sigmoid for u''
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2*s)

# --- 1. Problem Setup: u'' - 4u = 0 ---
# Exact solution: u(x) = e^(2x)
N_x = 50            # Collocation points
N_neurons = 30      # Hidden neurons
x_l, x_r = 0.0, 1.0 # Domain
u_l, u_r = 1.0, np.exp(2.0) # Boundary Values

# --- 2. ELM Initialization (Fixed Weights) ---
np.random.seed(42)
# Random input weights and biases (fixed)
W = np.random.normal(0, 10, (N_neurons, 1)) 
b_bias = np.random.normal(0, 10, (N_neurons, 1))

def phi(x):
    # Forward pass: Activation(Wx + b)
    return sigmoid(np.dot(x, W.T) + b_bias.T)

def L_phi(x):
    # Operator L[u] = u'' - 4u applied to hidden neurons
    z = np.dot(x, W.T) + b_bias.T
    val = sigmoid(z)
    d2_val = d2_sigmoid(z) * (W.T ** 2) # Chain rule: * w^2
    return d2_val - 4 * val

# --- 3. Construct Matrix H (As per PDF Page 4) ---
x_col = np.linspace(x_l, x_r, N_x).reshape(-1, 1)

# Matrix Psi: Operator on collocation points
Psi = L_phi(x_col) 

# Boundary rows
row_left = phi(np.array([[x_l]]))
row_right = phi(np.array([[x_r]]))

# Stack to form H
H = np.vstack([Psi, row_left, row_right])

# --- 4. Construct Target Vector b ---
# Zeros for PDE residual, BC values for boundaries
b = np.zeros((N_x + 2, 1))
b[-2] = u_l
b[-1] = u_r

# --- 5. Solve Linear System ---
# C = pinv(H) * b
H_pinv = np.linalg.pinv(H)
C = np.dot(H_pinv, b)

# --- 6. Predict & Plot ---
x_test = np.linspace(x_l, x_r, 100).reshape(-1, 1)
u_pred = np.dot(phi(x_test), C)
u_exact = np.exp(2 * x_test)

plt.figure(figsize=(10, 6))
plt.plot(x_test, u_exact, 'k--', label='Exact')
plt.plot(x_test, u_pred, 'r', label='PIELM Prediction', alpha=0.7)
plt.legend()
plt.title("PIELM Solution for u'' - 4u = 0")
plt.show()