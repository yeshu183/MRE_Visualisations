"""Test: Pure data constraints (bc_weight=0, data_weight>0).

This tests whether we can solve the inverse problem using ONLY data fitting,
without any boundary condition constraints.
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ForwardMREModel
from pielm_solver import pielm_solve
import matplotlib.pyplot as plt

device = torch.device('cpu')

with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

print("="*70)
print("DATA ONLY APPROACH (bc_weight=0)")
print("="*70)

# Setup
n_points = 100
x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
mu_true = 1.0 + 1.0 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))

# Generate reference data
print("\n1. Generating reference data with TRUE mu...")
torch.manual_seed(0)
model_gen = ForwardMREModel(n_neurons_wave=60, input_dim=1, seed=0).to(device)

bc_indices = torch.tensor([0, n_points - 1], dtype=torch.long, device=device)
u_bc_vals = torch.tensor([[0.01], [0.0]], device=device)

phi, phi_lap = model_gen.get_basis_and_laplacian(x)
H_pde_true = mu_true * phi_lap + config['rho_omega2'] * phi
b_pde = torch.zeros(n_points, 1, device=device)
H_bc = phi[bc_indices, :]
b_bc = u_bc_vals
H_true = torch.cat([H_pde_true, 200.0 * H_bc], dim=0)
b_true = torch.cat([b_pde, 200.0 * b_bc], dim=0)
C_true = pielm_solve(H_true, b_true, verbose=False)
u_true = phi @ C_true
u_meas = u_true + 0.001 * torch.randn_like(u_true)

print(f"   True mu range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
print(f"   True u range: [{u_true.min():.6f}, {u_true.max():.6f}]")
print(f"   Measurements range: [{u_meas.min():.6f}, {u_meas.max():.6f}]")

# Test with constant mu
mu_test = torch.ones_like(mu_true) * 1.5

print("\n" + "="*70)
print("2. TESTING DATA WEIGHTS (bc_weight=0)")
print("="*70)

results = []

for data_weight in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
    print(f"\n{'─'*70}")
    print(f"data_weight = {data_weight}")
    print(f"{'─'*70}")
    
    # Build system with data constraints ONLY
    H_pde = mu_test * phi_lap + config['rho_omega2'] * phi
    b_pde = torch.zeros(n_points, 1, device=device)
    H_data = phi * data_weight
    b_data = u_meas * data_weight
    
    H = torch.cat([H_pde, H_data], dim=0)
    b = torch.cat([b_pde, b_data], dim=0)
    
    print(f"System size: H {H.shape}, b {b.shape}")
    print(f"   PDE rows:  {H_pde.shape[0]}")
    print(f"   Data rows: {H_data.shape[0]}")
    
    # Energies
    E_pde = torch.norm(H_pde).item()
    E_data = torch.norm(H_data).item()
    scale_pde = E_pde**2
    scale_data = E_data**2
    
    print(f"\nBlock energies:")
    print(f"   ||H_pde||² = {scale_pde:.3e}  ({100*scale_pde/(scale_pde+scale_data):.1f}%)")
    print(f"   ||H_data||² = {scale_data:.3e}  ({100*scale_data/(scale_pde+scale_data):.1f}%)")
    
    # Solve
    try:
        C = pielm_solve(H, b, verbose=False)
        u_pred = phi @ C
        
        # Losses
        data_loss = torch.mean((u_pred - u_meas) ** 2).item()
        pde_residual = torch.mean((H_pde @ C) ** 2).item()
        
        # Conditioning
        condH = torch.linalg.cond(H).item()
        
        # Gradient test
        mu_test_grad = mu_test.clone().requires_grad_(True)
        H_pde_grad = mu_test_grad * phi_lap + config['rho_omega2'] * phi
        H_grad = torch.cat([H_pde_grad, H_data], dim=0)
        C_grad = pielm_solve(H_grad, b, verbose=False)
        u_pred_grad = phi @ C_grad
        loss_grad = torch.mean((u_pred_grad - u_meas) ** 2)
        loss_grad.backward()
        grad_norm = mu_test_grad.grad.norm().item()
        
        print(f"\nResults:")
        print(f"   u_pred range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
        print(f"   Data loss:    {data_loss:.6e}")
        print(f"   PDE residual: {pde_residual:.6e}")
        print(f"   cond(H):      {condH:.3e}")
        print(f"   Gradient norm: {grad_norm:.6e}")
        
        # Check if BCs are satisfied
        u_bc_actual = u_pred[bc_indices]
        bc_error = torch.mean((u_bc_actual - u_bc_vals)**2).item()
        print(f"   BC error:     {bc_error:.6e}  (should be near 0 if BCs satisfied)")
        print(f"   u[0]={u_pred[0].item():.6f} (should be {u_bc_vals[0].item():.6f})")
        print(f"   u[-1]={u_pred[-1].item():.6f} (should be {u_bc_vals[-1].item():.6f})")
        
        if scale_data > 0.5 * scale_pde:
            print(f"   ⚠️  Data dominates PDE ({100*scale_data/(scale_pde+scale_data):.0f}%)")
        
        results.append({
            'weight': data_weight,
            'data_loss': data_loss,
            'grad_norm': grad_norm,
            'bc_error': bc_error,
            'u_pred': u_pred.detach(),
            'pde_residual': pde_residual
        })
        
    except Exception as e:
        print(f"   ❌ Solve failed: {e}")

print("\n" + "="*70)
print("3. VISUALIZATION")
print("="*70)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, res in enumerate(results[:6]):
    ax = axes.flat[idx]
    ax.plot(x.cpu(), u_true.cpu(), 'k-', label='True u', linewidth=2)
    ax.plot(x.cpu(), u_meas.cpu(), 'k.', label='Measurements', markersize=3, alpha=0.5)
    ax.plot(x.cpu(), res['u_pred'].cpu(), 'r-', label='Predicted u', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(f"data_weight={res['weight']}\nData loss={res['data_loss']:.2e}, BC error={res['bc_error']:.2e}")
    ax.set_xlabel('Position x')
    ax.set_ylabel('Displacement u')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'data_only_test.png'), dpi=150, bbox_inches='tight')
print("   Saved: approach/data_only_test.png")

print("\n" + "="*70)
print("4. SUMMARY")
print("="*70)

print("\nGradient strength comparison:")
for res in results:
    print(f"   data_weight={res['weight']:5.1f}: grad={res['grad_norm']:.3e}, BC error={res['bc_error']:.3e}")

print("\n🔍 KEY OBSERVATIONS:")
print("\nWithout BC constraints (bc_weight=0):")
print("  - Solution is UNDERCONSTRAINED (infinite solutions)")
print("  - BCs are NOT enforced (check BC error)")
print("  - Data fitting may interpolate wrong solution")
print("  - Gradients may be misleading")
print("\nFor inverse problem to work:")
print("  - Need UNIQUE solution (requires BCs)")
print("  - mu must affect solution through physics")
print("  - Gradients must point toward correct mu")
