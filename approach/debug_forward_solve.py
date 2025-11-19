"""Debug the forward solve to understand why u_pred is all zeros."""

import torch
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ForwardMREModel

# Load config
with open('approach/config_forward.json', 'r') as f:
    config = json.load(f)

device = torch.device('cpu')
torch.manual_seed(config['seed'])

print("="*70)
print("FORWARD SOLVE DEBUG")
print("="*70)

# Create simple test case
n_points = 100
x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
mu_const = torch.ones(n_points, 1, device=device) * 1.874

# Boundary conditions
bc_indices = torch.tensor([0, n_points - 1], dtype=torch.long, device=device)
u_bc_vals = torch.zeros(2, 1, device=device)

# Create model
model = ForwardMREModel(
    n_neurons_wave=config['n_wave_neurons'],
    input_dim=1,
    seed=config['seed']
).to(device)

print("\n1. Model setup:")
print(f"   Wave neurons: {config['n_wave_neurons']}")
print(f"   Points: {n_points}")
print(f"   Constant mu: {mu_const[0,0]:.3f}")

# Get basis functions
print("\n2. Computing basis functions...")
phi, phi_lap = model.get_basis_and_laplacian(x)
print(f"   phi shape: {phi.shape}")
print(f"   phi range: [{phi.min():.6f}, {phi.max():.6f}]")
print(f"   phi_lap range: [{phi_lap.min():.6f}, {phi_lap.max():.6f}]")

# Build system
print("\n3. Building linear system...")
H, b = model.build_system(
    mu_const, phi, phi_lap, 
    config['rho_omega2'], 
    bc_indices, u_bc_vals, 
    config['bc_weight']
)
print(f"   H shape: {H.shape}")
print(f"   b shape: {b.shape}")
print(f"   H range: [{H.min():.6f}, {H.max():.6f}]")
print(f"   b range: [{b.min():.6f}, {b.max():.6f}]")
print(f"   H condition number: {torch.linalg.cond(H).item():.2e}")

# Check if b is all zeros
b_nonzero = torch.count_nonzero(b).item()
print(f"   Non-zero elements in b: {b_nonzero}/{b.numel()}")

# Solve
print("\n4. Solving system...")
from pielm_solver import pielm_solve
C_u = pielm_solve(H, b, verbose=True)
print(f"   C_u shape: {C_u.shape}")
print(f"   C_u range: [{C_u.min():.6f}, {C_u.max():.6f}]")
print(f"   Non-zero elements in C_u: {torch.count_nonzero(C_u).item()}/{C_u.numel()}")

# Compute u
u = phi @ C_u
print(f"\n5. Result:")
print(f"   u shape: {u.shape}")
print(f"   u range: [{u.min():.6f}, {u.max():.6f}]")
print(f"   Non-zero elements in u: {torch.count_nonzero(u).item()}/{u.numel()}")

# Check what happens with non-zero BC
print("\n" + "="*70)
print("Testing with NON-ZERO boundary conditions:")
print("="*70)
u_bc_vals_nonzero = torch.tensor([[0.001], [0.001]], device=device)
H2, b2 = model.build_system(
    mu_const, phi, phi_lap, 
    config['rho_omega2'], 
    bc_indices, u_bc_vals_nonzero, 
    config['bc_weight']
)
print(f"   b2 range: [{b2.min():.6f}, {b2.max():.6f}]")
print(f"   Non-zero elements in b2: {torch.count_nonzero(b2).item()}/{b2.numel()}")

C_u2 = pielm_solve(H2, b2, verbose=False)
u2 = phi @ C_u2
print(f"   u2 range: [{u2.min():.6f}, {u2.max():.6f}]")
print(f"   Non-zero elements in u2: {torch.count_nonzero(u2).item()}/{u2.numel()}")
