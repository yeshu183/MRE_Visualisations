"""Comprehensive test to validate core mathematical components.

Tests:
1. Forward solver: Given known mu, can we solve for u correctly?
2. Gradient flow: Do gradients propagate through the custom PIELM backward?
3. Gradient correctness: Is the analytical backward pass mathematically correct?
4. Sensitivity: Can the loss actually change when mu changes?
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ForwardMREModel
from pielm_solver import DifferentiablePIELM
import numpy as np

device = torch.device('cpu')

print("="*70)
print("CORE COMPONENT VALIDATION")
print("="*70)

# Load config
with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

# Test setup
n_points = 100
x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
bc_indices = torch.tensor([0, n_points - 1], dtype=torch.long, device=device)
u_bc_vals = torch.tensor([[0.01], [0.0]], device=device)

print("\n" + "="*70)
print("TEST 1: Forward Solver Consistency")
print("="*70)
print("Can we solve the forward problem consistently?")

# Create two models with same seed
model1 = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)
model2 = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)

# Test with constant mu
mu_const = torch.ones(n_points, 1, device=device) * 2.0

with torch.no_grad():
    u1, _ = model1.solve_given_mu(x, mu_const, bc_indices, u_bc_vals, 
                                   config['rho_omega2'], config['bc_weight'])
    u2, _ = model2.solve_given_mu(x, mu_const, bc_indices, u_bc_vals,
                                   config['rho_omega2'], config['bc_weight'])

diff = torch.abs(u1 - u2).max().item()
print(f"\nSolving with same mu twice:")
print(f"  u1 range: [{u1.min():.6f}, {u1.max():.6f}]")
print(f"  u2 range: [{u2.min():.6f}, {u2.max():.6f}]")
print(f"  Max difference: {diff:.2e}")

if diff < 1e-10:
    print("✅ Forward solver is DETERMINISTIC and CONSISTENT")
else:
    print("❌ Forward solver has INCONSISTENCY")

print("\n" + "="*70)
print("TEST 2: Gradient Flow Through Custom Backward")
print("="*70)
print("Do gradients flow from loss -> u -> C -> H -> mu?")

# Create model and enable gradients
model = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)

# Create a simple varying mu
mu_var = 1.5 + 0.5 * torch.sin(2 * 3.14159 * x)
mu_var.requires_grad_(True)

# Forward solve
u_pred, _ = model.solve_given_mu(x, mu_var, bc_indices, u_bc_vals,
                                  config['rho_omega2'], config['bc_weight'],
                                  verbose=False)

# Simple loss
loss = torch.mean(u_pred ** 2)

print(f"\nForward pass:")
print(f"  mu range: [{mu_var.min():.3f}, {mu_var.max():.3f}]")
print(f"  u range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
print(f"  Loss: {loss.item():.6e}")
print(f"  u_pred.requires_grad: {u_pred.requires_grad}")
print(f"  u_pred.grad_fn: {u_pred.grad_fn}")

# Backward pass
loss.backward()

print(f"\nBackward pass:")
print(f"  mu_var.grad exists: {mu_var.grad is not None}")
if mu_var.grad is not None:
    grad_norm = mu_var.grad.norm().item()
    grad_mean = mu_var.grad.mean().item()
    print(f"  mu gradient norm: {grad_norm:.6e}")
    print(f"  mu gradient mean: {grad_mean:.6e}")
    print(f"  mu gradient range: [{mu_var.grad.min():.6e}, {mu_var.grad.max():.6e}]")
    
    if grad_norm > 1e-12:
        print("✅ Gradients FLOW through custom backward pass")
    else:
        print("❌ Gradients are ZERO (broken backward)")
else:
    print("❌ No gradients computed (broken backward)")

print("\n" + "="*70)
print("TEST 3: Analytical Gradient Correctness")
print("="*70)
print("Is the custom backward mathematically correct? (Finite difference test)")

# Simple test case for gradient verification
torch.manual_seed(42)
H = torch.randn(50, 20, device=device, dtype=torch.float64, requires_grad=True)
b = torch.randn(50, 1, device=device, dtype=torch.float64, requires_grad=True)

# Forward with custom
C_custom = DifferentiablePIELM.apply(H, b, 1e-2)
loss_custom = (C_custom ** 2).sum()
loss_custom.backward()

grad_H_custom = H.grad.clone()
grad_b_custom = b.grad.clone()

# Numerical gradient via finite differences
H.grad = None
b.grad = None
eps = 1e-5

# Test one element of H
i, j = 10, 5
H_plus = H.clone().detach()
H_plus[i, j] += eps
H_minus = H.clone().detach()
H_minus[i, j] -= eps

with torch.no_grad():
    C_plus = DifferentiablePIELM.apply(H_plus, b.detach(), 1e-2)
    C_minus = DifferentiablePIELM.apply(H_minus, b.detach(), 1e-2)
    loss_plus = (C_plus ** 2).sum()
    loss_minus = (C_minus ** 2).sum()
    
numerical_grad_H = (loss_plus - loss_minus) / (2 * eps)
analytical_grad_H = grad_H_custom[i, j].item()

error = abs(numerical_grad_H.item() - analytical_grad_H)
rel_error = error / (abs(numerical_grad_H.item()) + 1e-10)

print(f"\nGradient check for H[{i},{j}]:")
print(f"  Numerical gradient:  {numerical_grad_H.item():.8e}")
print(f"  Analytical gradient: {analytical_grad_H:.8e}")
print(f"  Absolute error:      {error:.8e}")
print(f"  Relative error:      {rel_error:.8e}")

if rel_error < 1e-4:
    print("✅ Custom backward is MATHEMATICALLY CORRECT")
else:
    print("❌ Custom backward has MATHEMATICAL ERROR")

print("\n" + "="*70)
print("TEST 4: Loss Sensitivity to Mu Changes")
print("="*70)
print("Does changing mu actually affect the loss?")

model = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)

# Generate reference data with mu=2.0
mu_ref = torch.ones(n_points, 1, device=device) * 2.0
with torch.no_grad():
    u_target, _ = model.solve_given_mu(x, mu_ref, bc_indices, u_bc_vals,
                                        config['rho_omega2'], config['bc_weight'])

print(f"\nReference solution with mu=2.0:")
print(f"  u_target range: [{u_target.min():.6f}, {u_target.max():.6f}]")

# Test different mu values
mu_values = [1.5, 2.0, 2.5, 3.0]
losses = []

for mu_val in mu_values:
    mu_test = torch.ones(n_points, 1, device=device) * mu_val
    with torch.no_grad():
        u_pred, _ = model.solve_given_mu(x, mu_test, bc_indices, u_bc_vals,
                                          config['rho_omega2'], config['bc_weight'])
    loss = torch.mean((u_pred - u_target) ** 2).item()
    losses.append(loss)
    print(f"  mu={mu_val:.1f}: loss={loss:.6e}")

# Check if loss has a clear minimum at mu=2.0
min_loss_idx = np.argmin(losses)
expected_idx = mu_values.index(2.0)

loss_variation = max(losses) / (min(losses) + 1e-10)
print(f"\nLoss variation factor: {loss_variation:.2f}x")
print(f"Minimum loss at mu={mu_values[min_loss_idx]}")

if min_loss_idx == expected_idx and loss_variation > 5:
    print("✅ Loss is SENSITIVE to mu changes (clear minimum exists)")
elif loss_variation > 2:
    print("⚠️  Loss has SOME sensitivity but might be weak")
else:
    print("❌ Loss is INSENSITIVE to mu changes (optimization impossible)")

print("\n" + "="*70)
print("TEST 5: Network Can Learn Gaussian Bump")
print("="*70)
print("Can the network learn to recover a Gaussian inclusion?")

# Generate data with Gaussian bump (more challenging)
mu_true = 1.0 + 2.0 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))
gen_model = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)
with torch.no_grad():
    u_meas, _ = gen_model.solve_given_mu(x, mu_true, bc_indices, u_bc_vals,
                                          config['rho_omega2'], config['bc_weight'])
    # Add significant noise
    u_meas = u_meas + 0.002 * torch.randn_like(u_meas)

# Train to recover Gaussian bump
train_model = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)
optimizer = torch.optim.Adam(train_model.mu_net.parameters(), lr=0.005)

print(f"\nTarget: Gaussian bump, mu range [{mu_true.min():.3f}, {mu_true.max():.3f}]")
print(f"Training for 500 iterations with noisy data (std=0.002)...")

best_loss = float('inf')
best_mu_mse = float('inf')

for i in range(500):
    optimizer.zero_grad()
    u_pred, mu_pred = train_model(x, bc_indices, u_bc_vals, 
                                   config['rho_omega2'], config['bc_weight'],
                                   verbose=False)
    loss = torch.mean((u_pred - u_meas) ** 2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(train_model.mu_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    with torch.no_grad():
        mu_mse = torch.mean((mu_pred - mu_true) ** 2).item()
        if mu_mse < best_mu_mse:
            best_mu_mse = mu_mse
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    if i % 100 == 0:
        mu_min = mu_pred.min().item()
        mu_max = mu_pred.max().item()
        mu_mean = mu_pred.mean().item()
        print(f"  Iter {i:3d}: loss={loss.item():.6e}, mu=[{mu_min:.3f}, {mu_max:.3f}], mean={mu_mean:.3f}, MSE={mu_mse:.6e}")

# Final evaluation
with torch.no_grad():
    final_mu_mse = torch.mean((mu_pred - mu_true) ** 2).item()
    final_mu_mae = torch.mean(torch.abs(mu_pred - mu_true)).item()
    peak_error = abs(mu_pred[50].item() - mu_true[50].item())  # Error at center
    
print(f"\nFinal result:")
print(f"  Best data loss:  {best_loss:.6e}")
print(f"  Best mu MSE:     {best_mu_mse:.6e}")
print(f"  Final mu MSE:    {final_mu_mse:.6e}")
print(f"  Final mu MAE:    {final_mu_mae:.3f}")
print(f"  Peak error:      {peak_error:.3f} (at x=0.5)")
print(f"  True peak:       {mu_true[50].item():.3f}")
print(f"  Pred peak:       {mu_pred[50].item():.3f}")

if final_mu_mse < 0.5 and peak_error < 1.0:
    print("✅ Network CAN recover Gaussian bump accurately")
elif final_mu_mse < 2.0 and best_mu_mse < final_mu_mse:
    print("⚠️  Network learns but overfits or diverges")
elif best_mu_mse < 1.0:
    print("⚠️  Network CAN learn but needs better stopping/regularization")
else:
    print("❌ Network STRUGGLES with spatial variations")

print("\n" + "="*70)
print("TEST 6: Gradient Magnitude Under Different Conditions")
print("="*70)
print("Are gradients strong enough for learning under various scenarios?")

model = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)

test_cases = [
    ("Small BC (u_bc=0.001)", torch.tensor([[0.001], [0.0]], device=device)),
    ("Medium BC (u_bc=0.01)", torch.tensor([[0.01], [0.0]], device=device)),
    ("Large BC (u_bc=0.05)", torch.tensor([[0.05], [0.0]], device=device)),
]

print("\nTesting gradient strength with different boundary conditions:")
for name, u_bc in test_cases:
    # Generate reference data
    mu_ref = 1.5 + 0.5 * torch.sin(2 * 3.14159 * x)
    with torch.no_grad():
        u_target, _ = model.solve_given_mu(x, mu_ref, bc_indices, u_bc,
                                            config['rho_omega2'], config['bc_weight'])
    
    # Forward with network
    model.mu_net.zero_grad()
    u_pred, mu_pred = model(x, bc_indices, u_bc, config['rho_omega2'], 
                           config['bc_weight'], verbose=False)
    loss = torch.mean((u_pred - u_target) ** 2)
    loss.backward()
    
    # Measure gradient strength
    grad_norm = sum(p.grad.norm().item()**2 for p in model.mu_net.parameters() 
                   if p.grad is not None)**0.5
    
    print(f"  {name:25s}: loss={loss.item():.6e}, grad_norm={grad_norm:.6e}")

print("\n✅ Gradient analysis complete (larger BC → stronger gradients expected)")

print("\n" + "="*70)
print("TEST 7: Inverse Problem Ill-Posedness Check")
print("="*70)
print("Can different mu values produce very similar wave fields? (Ill-posedness)")

model = ForwardMREModel(n_neurons_wave=60, seed=42).to(device)

# Test if small changes in mu lead to small changes in u
mu_base = torch.ones(n_points, 1, device=device) * 2.0
perturbations = [0.0, 0.01, 0.05, 0.1, 0.5]

print("\nTesting sensitivity of wave field to mu perturbations:")
print("  (Smaller u_diff for same mu_diff = more ill-posed)")

with torch.no_grad():
    u_base, _ = model.solve_given_mu(x, mu_base, bc_indices, u_bc_vals,
                                      config['rho_omega2'], config['bc_weight'])
    
    for delta in perturbations[1:]:
        mu_pert = mu_base + delta
        u_pert, _ = model.solve_given_mu(x, mu_pert, bc_indices, u_bc_vals,
                                          config['rho_omega2'], config['bc_weight'])
        u_diff = torch.mean((u_pert - u_base) ** 2).item()
        mu_diff = delta
        ratio = u_diff / (mu_diff**2 + 1e-10)
        print(f"  Δmu={mu_diff:.3f}: Δu={u_diff:.6e}, ratio={ratio:.6e}")

print("\n⚠️  If ratio is very small (< 1e-4), problem is severely ill-posed")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
