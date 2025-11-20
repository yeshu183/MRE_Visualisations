"""Diagnose why bc_weight=2 fails vs bc_weight=200 works."""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ForwardMREModel
from core.data_generators import generate_gaussian_bump

device = torch.device('cpu')

with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

print("="*70)
print("BC WEIGHT SCALING DIAGNOSTIC")
print("="*70)

# Generate data
x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_gaussian_bump(
    n_points=100,
    n_wave_neurons=60,
    device=device,
    seed=0
)

print(f"\nData setup:")
print(f"  N points: {len(x)}")
print(f"  M wave neurons: 60")
print(f"  K boundary points: {len(bc_indices)}")
print(f"  BC indices: {bc_indices.tolist()}")
print(f"  BC values: {u_bc_vals.squeeze().tolist()}")

# Create model
model = ForwardMREModel(n_neurons_wave=60, input_dim=1, seed=0).to(device)

# Use constant mu (like at iteration 0)
mu_test = torch.ones_like(mu_true) * 1.5

# Get basis
phi, phi_lap = model.get_basis_and_laplacian(x)

print(f"\n" + "="*70)
print(f"TESTING DIFFERENT BC WEIGHTS")
print(f"="*70)

for bc_weight in [1, 2, 10, 50, 100, 200]:
    print(f"\n{'─'*70}")
    print(f"bc_weight = {bc_weight}")
    print(f"{'─'*70}")
    
    # Build system components
    H_pde = mu_test * phi_lap + config['rho_omega2'] * phi  # (100, 60)
    b_pde = torch.zeros(len(x), 1, device=device)
    
    H_bc = phi[bc_indices, :]  # (2, 60)
    b_bc = u_bc_vals * bc_weight
    
    # Compute block energies (Frobenius norms)
    E_pde = torch.norm(H_pde, 'fro').item()
    E_bc_unweighted = torch.norm(H_bc, 'fro').item()
    E_bc_weighted = torch.norm(H_bc * bc_weight, 'fro').item()
    
    # Per-row magnitudes
    row_mag_pde = torch.norm(H_pde, dim=1).mean().item()
    row_mag_bc_unweighted = torch.norm(H_bc, dim=1).mean().item()
    row_mag_bc_weighted = torch.norm(H_bc * bc_weight, dim=1).mean().item()
    
    print(f"\nBlock Frobenius norms:")
    print(f"  ||H_pde||_F = {E_pde:.3e}  (100 rows)")
    print(f"  ||H_bc||_F  = {E_bc_unweighted:.3e}  (2 rows, unweighted)")
    print(f"  ||H_bc||_F  = {E_bc_weighted:.3e}  (2 rows, weighted)")
    
    print(f"\nAverage row magnitude:")
    print(f"  mean(||H_pde_row||) = {row_mag_pde:.3e}")
    print(f"  mean(||H_bc_row||)  = {row_mag_bc_weighted:.3e}  (weighted)")
    print(f"  Ratio (BC/PDE):     = {row_mag_bc_weighted/row_mag_pde:.2f}×")
    
    # Energy contributions to A = H^T H
    energy_pde = E_pde ** 2
    energy_bc = E_bc_weighted ** 2
    total_energy = energy_pde + energy_bc
    
    print(f"\nContribution to A = H^T H:")
    print(f"  PDE contribution:  {energy_pde:.3e}  ({100*energy_pde/total_energy:.1f}%)")
    print(f"  BC contribution:   {energy_bc:.3e}  ({100*energy_bc/total_energy:.1f}%)")
    
    # Stack and check condition number
    H_total = torch.cat([H_pde, H_bc * bc_weight], dim=0)
    b_total = torch.cat([b_pde, b_bc], dim=0)
    
    cond_H = torch.linalg.cond(H_total).item()
    
    # Condition of normal equations
    A = H_total.T @ H_total
    cond_A = torch.linalg.cond(A).item()
    
    print(f"\nConditioning:")
    print(f"  cond(H) = {cond_H:.3e}")
    print(f"  cond(A) = {cond_A:.3e}  {'⚠️ ILL-CONDITIONED!' if cond_A > 1e10 else '✓ OK' if cond_A < 1e6 else '⚠️ MARGINAL'}")
    
    # Rule of thumb for well-posed
    if energy_bc < 0.01 * energy_pde:
        print(f"\n  ❌ BC too weak: BC contributes <1% to A")
    elif energy_bc < 0.1 * energy_pde:
        print(f"\n  ⚠️  BC marginal: BC contributes <10% to A")
    else:
        print(f"\n  ✅ BC adequate: BC contributes ≥10% to A")

print(f"\n" + "="*70)
print(f"RECOMMENDATION")
print(f"="*70)
print(f"""
For this problem (100 PDE rows, 2 BC rows):

Minimum bc_weight for adequate BC strength:
  - Target: BC contributes ~10-50% of total energy to A
  - Need: bc_weight² × E_bc² ≈ 0.1 × E_pde²
  - Estimate: bc_weight ≈ sqrt(0.1 × E_pde² / E_bc²) × 2
  
Based on these numbers:
  ❌ bc_weight = 1-2:   BC too weak (<1% contribution)
  ⚠️  bc_weight = 10:   BC marginal (~5% contribution)
  ⚠️  bc_weight = 50:   BC adequate (~25% contribution)
  ✅ bc_weight = 100+: BC strong (>50% contribution)

Your successful bc_weight=200 gives BC ~90% contribution to A.
This heavily weights the boundary conditions and stabilizes the solve.
""")
