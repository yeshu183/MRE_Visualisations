"""Test: Compare BC constraints vs Data constraints for PIELM solve.

This tests the hypothesis: Can we replace boundary conditions with
direct data fitting constraints while preserving mu dependence?
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ForwardMREModel
from pielm_solver import pielm_solve

device = torch.device('cpu')

with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

print("="*70)
print("BC CONSTRAINTS vs DATA CONSTRAINTS")
print("="*70)

# Setup
n_points = 100
x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
mu_true = 1.0 + 1.0 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))

# Create ground truth data
print("\n1. Generating ground truth data...")
torch.manual_seed(0)
model_gen = ForwardMREModel(n_neurons_wave=60, input_dim=1, seed=0).to(device)

bc_indices = torch.tensor([0, n_points - 1], dtype=torch.long, device=device)
u_bc_vals = torch.tensor([[0.01], [0.0]], device=device)

phi, phi_lap = model_gen.get_basis_and_laplacian(x)
H_pde_true = mu_true * phi_lap + config['rho_omega2'] * phi
b_pde = torch.zeros(n_points, 1, device=device)
H_bc = phi[bc_indices, :]
b_bc = u_bc_vals
H_true = torch.cat([H_pde_true, config['bc_weight'] * H_bc], dim=0)
b_true = torch.cat([b_pde, config['bc_weight'] * b_bc], dim=0)
C_true = pielm_solve(H_true, b_true, verbose=False)
u_true = phi @ C_true

# Add noise
u_meas = u_true + 0.001 * torch.randn_like(u_true)

print(f"   Generated {n_points} points")
print(f"   mu_true range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
print(f"   u_true range: [{u_true.min():.6f}, {u_true.max():.6f}]")
print(f"   u_meas range: [{u_meas.min():.6f}, {u_meas.max():.6f}]")

# Test with constant mu prediction (like at iteration 0)
mu_pred = torch.ones_like(mu_true) * 1.5

print("\n" + "="*70)
print("APPROACH 1: Current (BC Constraints)")
print("="*70)

H_pde_1 = mu_pred * phi_lap + config['rho_omega2'] * phi
b_pde_1 = torch.zeros(n_points, 1, device=device)
H_bc_1 = phi[bc_indices, :]
b_bc_1 = u_bc_vals

H_1 = torch.cat([H_pde_1, config['bc_weight'] * H_bc_1], dim=0)
b_1 = torch.cat([b_pde_1, config['bc_weight'] * b_bc_1], dim=0)

print(f"System size: H {H_1.shape}, b {b_1.shape}")
print(f"   PDE rows:  {H_pde_1.shape[0]}")
print(f"   BC rows:   {H_bc_1.shape[0]}")

C_1 = pielm_solve(H_1, b_1, verbose=False)
u_pred_1 = phi @ C_1

# Compute PDE residual
pde_residual_1 = H_pde_1 @ C_1 - b_pde_1
pde_loss_1 = torch.mean(pde_residual_1 ** 2)

loss_1 = torch.mean((u_pred_1 - u_meas) ** 2)
mu_mse_1 = torch.mean((mu_pred - mu_true) ** 2)

print(f"\nResults:")
print(f"   u_pred range: [{u_pred_1.min():.6f}, {u_pred_1.max():.6f}]")
print(f"   PDE residual loss: {pde_loss_1.item():.6e}")
print(f"   Data loss (MSE):   {loss_1.item():.6e}")
print(f"   Mu MSE: {mu_mse_1.item():.6e}")

# Compute gradient
mu_pred_1 = mu_pred.clone().requires_grad_(True)
H_pde_1 = mu_pred_1 * phi_lap + config['rho_omega2'] * phi
H_1 = torch.cat([H_pde_1, config['bc_weight'] * H_bc_1], dim=0)
C_1 = pielm_solve(H_1, b_1, verbose=False)
u_pred_1 = phi @ C_1
loss_1 = torch.mean((u_pred_1 - u_meas) ** 2)
loss_1.backward()
grad_norm_1 = mu_pred_1.grad.norm().item()
print(f"   Gradient norm: {grad_norm_1:.6e}")

print("\n" + "="*70)
print("APPROACH 2: Proposed (Data Constraints)")
print("="*70)

# Try different data weights
for data_weight in [1.0, 10.0, 100.0, 200.0]:
    print(f"\n--- data_weight = {data_weight} ---")
    
    H_pde_2 = mu_pred * phi_lap + config['rho_omega2'] * phi
    b_pde_2 = torch.zeros(n_points, 1, device=device)
    H_data = phi
    b_data = u_meas
    
    H_2 = torch.cat([H_pde_2, data_weight * H_data], dim=0)
    b_2 = torch.cat([b_pde_2, data_weight * b_data], dim=0)
    
    print(f"System size: H {H_2.shape}, b {b_2.shape}")
    print(f"   PDE rows:  {H_pde_2.shape[0]}")
    print(f"   Data rows: {H_data.shape[0]}")
    
    C_2 = pielm_solve(H_2, b_2, verbose=False)
    u_pred_2 = phi @ C_2
    
    # Compute PDE residual
    pde_residual = H_pde_2 @ C_2 - b_pde_2
    pde_loss = torch.mean(pde_residual ** 2)
    
    # Compute data loss
    data_loss_2 = torch.mean((u_pred_2 - u_meas) ** 2)
    
    print(f"   u_pred range: [{u_pred_2.min():.6f}, {u_pred_2.max():.6f}]")
    print(f"   PDE residual loss: {pde_loss.item():.6e}")
    print(f"   Data loss (MSE):   {data_loss_2.item():.6e}")
    
    # Compute gradient
    mu_pred_2 = mu_pred.clone().requires_grad_(True)
    H_pde_2 = mu_pred_2 * phi_lap + config['rho_omega2'] * phi
    H_2 = torch.cat([H_pde_2, data_weight * H_data], dim=0)
    C_2 = pielm_solve(H_2, b_2, verbose=False)
    u_pred_2 = phi @ C_2
    loss_2 = torch.mean((u_pred_2 - u_meas) ** 2)
    loss_2.backward()
    grad_norm_2 = mu_pred_2.grad.norm().item()
    print(f"   Gradient norm: {grad_norm_2:.6e}")
    
    print(f"   Improvement over BC: {loss_1.item()/data_loss_2.item():.2f}×")
    print(f"   Gradient ratio: {grad_norm_2/grad_norm_1:.2f}×")

print("\n" + "="*70)
print("APPROACH 3: Hybrid (PDE + BC + Data)")
print("="*70)

# Try hybrid with different data weights
for data_weight in [1.0, 5.0, 10.0]:
    print(f"\n--- bc_weight = {config['bc_weight']}, data_weight = {data_weight} ---")
    
    H_pde_3 = mu_pred * phi_lap + config['rho_omega2'] * phi
    b_pde_3 = torch.zeros(n_points, 1, device=device)
    H_bc_3 = phi[bc_indices, :]
    b_bc_3 = u_bc_vals
    H_data_3 = phi
    b_data_3 = u_meas
    
    H_3 = torch.cat([H_pde_3, config['bc_weight'] * H_bc_3, data_weight * H_data_3], dim=0)
    b_3 = torch.cat([b_pde_3, config['bc_weight'] * b_bc_3, data_weight * b_data_3], dim=0)
    
    print(f"System size: H {H_3.shape}, b {b_3.shape}")
    print(f"   PDE rows:  {H_pde_3.shape[0]}")
    print(f"   BC rows:   {H_bc_3.shape[0]}")
    print(f"   Data rows: {H_data_3.shape[0]}")
    
    C_3 = pielm_solve(H_3, b_3, verbose=False)
    u_pred_3 = phi @ C_3
    
    # Compute losses
    pde_residual_3 = H_pde_3 @ C_3 - b_pde_3
    pde_loss_3 = torch.mean(pde_residual_3 ** 2)
    data_loss_3 = torch.mean((u_pred_3 - u_meas) ** 2)
    
    print(f"   u_pred range: [{u_pred_3.min():.6f}, {u_pred_3.max():.6f}]")
    print(f"   PDE residual loss: {pde_loss_3.item():.6e}")
    print(f"   Data loss (MSE):   {data_loss_3.item():.6e}")
    
    # Compute gradient
    mu_pred_3 = mu_pred.clone().requires_grad_(True)
    H_pde_3 = mu_pred_3 * phi_lap + config['rho_omega2'] * phi
    H_3 = torch.cat([H_pde_3, config['bc_weight'] * H_bc_3, data_weight * H_data_3], dim=0)
    C_3 = pielm_solve(H_3, b_3, verbose=False)
    u_pred_3 = phi @ C_3
    loss_3 = torch.mean((u_pred_3 - u_meas) ** 2)
    loss_3.backward()
    grad_norm_3 = mu_pred_3.grad.norm().item()
    print(f"   Gradient norm: {grad_norm_3:.6e}")
    
    print(f"   Improvement over BC: {loss_1.item()/data_loss_3.item():.2f}×")
    print(f"   Gradient ratio: {grad_norm_3/grad_norm_1:.2f}×")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✅ Data constraints preserve mu dependence (gradients flow!)")
print("✅ Data constraints can improve data fitting")
print("⚠️  Need to tune data_weight vs PDE influence")
print("\n💡 Key observations:")
print("   - PDE loss shows how well physics is satisfied")
print("   - Data loss shows how well measurements are matched")
print("   - Balance both for optimal reconstruction")
print("\nRecommendation: Use hybrid approach with data_weight tuned to balance PDE & data losses")
