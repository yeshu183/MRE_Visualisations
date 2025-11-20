"""Diagnostic: Test different data_weight values to understand scaling.

Similar to diagnose_bc_scaling.py but for data constraints.
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ForwardMREModel

device = torch.device('cpu')

with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

print("="*70)
print("DATA WEIGHT SCALING DIAGNOSTIC")
print("="*70)

# Setup
n_points = 100
x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
mu_test = 1.0 + 1.0 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))

print("\nData setup:")
print(f"  N points: {n_points}")
print(f"  M wave neurons: {config['n_wave_neurons']}")
print(f"  mu range: [{mu_test.min():.3f}, {mu_test.max():.3f}]")

# Create model and get basis
torch.manual_seed(config['seed'])
model = ForwardMREModel(
    n_neurons_wave=config['n_wave_neurons'],
    input_dim=1,
    seed=config['seed']
).to(device)

phi, phi_lap = model.get_basis_and_laplacian(x)

# Generate reference data
bc_indices = torch.tensor([0, n_points - 1], dtype=torch.long, device=device)
u_bc_vals = torch.tensor([[0.01], [0.0]], device=device)

H_pde_ref = mu_test * phi_lap + config['rho_omega2'] * phi
b_pde_ref = torch.zeros(n_points, 1, device=device)
H_bc_ref = phi[bc_indices, :]
b_bc_ref = u_bc_vals * 200.0  # Use strong BC for reference

from pielm_solver import pielm_solve
H_ref = torch.cat([H_pde_ref, H_bc_ref * 200.0], dim=0)
b_ref = torch.cat([b_pde_ref, b_bc_ref], dim=0)
C_ref = pielm_solve(H_ref, b_ref, verbose=False)
u_ref = phi @ C_ref

print(f"  Reference u range: [{u_ref.min():.6f}, {u_ref.max():.6f}]")

# Use constant mu for testing
mu_const = torch.ones_like(mu_test) * 1.5
H_pde = mu_const * phi_lap + config['rho_omega2'] * phi
b_pde = torch.zeros(n_points, 1, device=device)
H_bc = phi[bc_indices, :]
b_bc = u_bc_vals

print("\n" + "="*70)
print("TESTING DIFFERENT DATA WEIGHTS (with bc_weight=200)")
print("="*70)

bc_weight = 200.0
data_weights = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

for data_weight in data_weights:
    print("\n" + "─"*70)
    print(f"bc_weight = {bc_weight}, data_weight = {data_weight}")
    print("─"*70)
    
    # Build system
    if data_weight > 0:
        H_data = phi
        b_data = u_ref
        H = torch.cat([H_pde, bc_weight * H_bc, data_weight * H_data], dim=0)
        b = torch.cat([b_pde, bc_weight * b_bc, data_weight * b_data], dim=0)
        n_data_rows = n_points
    else:
        H = torch.cat([H_pde, bc_weight * H_bc], dim=0)
        b = torch.cat([b_pde, bc_weight * b_bc], dim=0)
        n_data_rows = 0
    
    # Compute norms
    E_pde = torch.norm(H_pde).item()
    E_bc_weighted = torch.norm(bc_weight * H_bc).item()
    
    print(f"\nBlock Frobenius norms:")
    print(f"  ||H_pde||_F = {E_pde:.3e}  ({n_points} rows)")
    print(f"  ||H_bc||_F  = {E_bc_weighted:.3e}  (2 rows, weighted)")
    
    if data_weight > 0:
        E_data_weighted = torch.norm(data_weight * H_data).item()
        print(f"  ||H_data||_F = {E_data_weighted:.3e}  ({n_data_rows} rows, weighted)")
    
    # Compute contributions to A = H^T H
    contrib_pde = E_pde ** 2
    contrib_bc = E_bc_weighted ** 2
    
    if data_weight > 0:
        contrib_data = E_data_weighted ** 2
        total = contrib_pde + contrib_bc + contrib_data
        pct_pde = 100 * contrib_pde / total
        pct_bc = 100 * contrib_bc / total
        pct_data = 100 * contrib_data / total
    else:
        contrib_data = 0
        total = contrib_pde + contrib_bc
        pct_pde = 100 * contrib_pde / total
        pct_bc = 100 * contrib_bc / total
        pct_data = 0
    
    print(f"\nContribution to A = H^T H:")
    print(f"  PDE contribution:  {contrib_pde:.3e}  ({pct_pde:.1f}%)")
    print(f"  BC contribution:   {contrib_bc:.3e}  ({pct_bc:.1f}%)")
    if data_weight > 0:
        print(f"  Data contribution: {contrib_data:.3e}  ({pct_data:.1f}%)")
    
    # Solve and check result
    C = pielm_solve(H, b, verbose=False)
    u_pred = phi @ C
    
    data_loss = torch.mean((u_pred - u_ref) ** 2).item()
    pde_residual = torch.mean((H_pde @ C) ** 2).item()
    
    print(f"\nForward solve results:")
    print(f"  u_pred range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
    print(f"  Data loss (MSE): {data_loss:.6e}")
    print(f"  PDE residual:    {pde_residual:.6e}")
    
    # Check conditioning
    condH = torch.linalg.cond(H).item()
    print(f"  cond(H) = {condH:.3e}", end="")
    if condH > 1e8:
        print("  ⚠️ ILL-CONDITIONED!")
    else:
        print()
    
    # Gradient test
    mu_grad_test = mu_const.clone().requires_grad_(True)
    H_pde_test = mu_grad_test * phi_lap + config['rho_omega2'] * phi
    
    if data_weight > 0:
        H_test = torch.cat([H_pde_test, bc_weight * H_bc, data_weight * H_data], dim=0)
    else:
        H_test = torch.cat([H_pde_test, bc_weight * H_bc], dim=0)
    
    C_test = pielm_solve(H_test, b, verbose=False)
    u_test = phi @ C_test
    loss_test = torch.mean((u_test - u_ref) ** 2)
    loss_test.backward()
    grad_norm = mu_grad_test.grad.norm().item()
    
    print(f"  Gradient norm:   {grad_norm:.6e}")
    
    # Assessment
    print(f"\nAssessment:")
    if pct_data > 50:
        print(f"  ❌ Data dominates ({pct_data:.1f}%) - gradient signal suppressed")
    elif pct_data > 20:
        print(f"  ⚠️  Data significant ({pct_data:.1f}%) - gradients weakened")
    elif pct_data > 5:
        print(f"  ⚠️  Data moderate ({pct_data:.1f}%) - may affect gradients")
    elif data_weight > 0:
        print(f"  ✅ Data minor ({pct_data:.1f}%) - gradients mostly preserved")
    else:
        print(f"  ✅ No data constraints - full gradient strength")
    
    if data_loss < 1e-4:
        print(f"  ✅ Good data fit (loss={data_loss:.2e})")
    elif data_loss < 1e-3:
        print(f"  ⚠️  Moderate data fit (loss={data_loss:.2e})")
    else:
        print(f"  ❌ Poor data fit (loss={data_loss:.2e})")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nKey findings:")
print("  - data_weight scales contribution quadratically (weight²)")
print("  - With 100 data rows, even small weights (0.1-1.0) contribute significantly")
print("  - Data rows have NO mu dependence → suppress mu gradients")
print("  - As data_weight increases:")
print("    * Data fit improves (lower MSE)")
print("    * But gradient strength decreases (weaker learning)")
print("\nRecommendation:")
print("  - For INVERSE problems: Use bc_weight=200, data_weight=0")
print("  - For FORWARD validation: data_weight can show fit quality")
print("  - Hybrid approach needs careful tuning (data_weight < 0.1)")
