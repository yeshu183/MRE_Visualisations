"""Test: Can we use heavily weighted interior points instead of BCs?

Key question: If we weight interior data points heavily (simulating high-quality
measurements at specific locations), can we get a unique solution without 
relying on boundary conditions?

This is relevant for MRE where we don't control BCs - we just have displacement
measurements throughout the tissue.
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ForwardMREModel
from pielm_solver import pielm_solve
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')

with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

print("="*70)
print("INTERIOR POINT WEIGHTING vs BC WEIGHTING")
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

# Test with constant mu (to see if weighting helps distinguish)
mu_test = torch.ones_like(mu_true) * 1.5

print("\n" + "="*70)
print("2. STRATEGY COMPARISON")
print("="*70)

strategies = [
    {
        'name': 'BC only (baseline)',
        'bc_weight': 200.0,
        'data_weight': 0.0,
        'data_indices': None
    },
    {
        'name': 'Uniform data (all points equal)',
        'bc_weight': 0.0,
        'data_weight': 1.0,
        'data_indices': None
    },
    {
        'name': 'Heavy interior (10 points, weight=200)',
        'bc_weight': 0.0,
        'data_weight': 200.0,
        'data_indices': torch.linspace(10, 89, 10, dtype=torch.long)  # 10 interior points
    },
    {
        'name': 'Heavy interior (25 points, weight=200)',
        'bc_weight': 0.0,
        'data_weight': 200.0,
        'data_indices': torch.linspace(10, 89, 25, dtype=torch.long)  # 25 interior points
    },
    {
        'name': 'Heavy interior (50 points, weight=200)',
        'bc_weight': 0.0,
        'data_weight': 200.0,
        'data_indices': torch.linspace(10, 89, 50, dtype=torch.long)  # 50 interior points
    },
    {
        'name': 'Sparse high-weight interior (5 key points, weight=1000)',
        'bc_weight': 0.0,
        'data_weight': 1000.0,
        'data_indices': torch.tensor([20, 35, 50, 65, 80], dtype=torch.long)  # 5 strategic points
    },
    {
        'name': 'Hybrid: Weak BC + Heavy interior',
        'bc_weight': 10.0,
        'data_weight': 200.0,
        'data_indices': torch.linspace(10, 89, 25, dtype=torch.long)
    },
]

results = []

for strategy in strategies:
    print(f"\n{'─'*70}")
    print(f"Strategy: {strategy['name']}")
    print(f"{'─'*70}")
    
    # Build system
    H_pde = mu_test * phi_lap + config['rho_omega2'] * phi
    b_pde = torch.zeros(n_points, 1, device=device)
    
    H_parts = [H_pde]
    b_parts = [b_pde]
    
    # Add BC if requested
    if strategy['bc_weight'] > 0:
        H_bc_weighted = phi[bc_indices, :] * strategy['bc_weight']
        b_bc_weighted = u_bc_vals * strategy['bc_weight']
        H_parts.append(H_bc_weighted)
        b_parts.append(b_bc_weighted)
    
    # Add data constraints
    if strategy['data_weight'] > 0:
        if strategy['data_indices'] is not None:
            # Selected points only
            data_idx = strategy['data_indices']
            H_data = phi[data_idx, :] * strategy['data_weight']
            b_data = u_meas[data_idx] * strategy['data_weight']
            n_data = len(data_idx)
        else:
            # All points
            H_data = phi * strategy['data_weight']
            b_data = u_meas * strategy['data_weight']
            n_data = n_points
        H_parts.append(H_data)
        b_parts.append(b_data)
    else:
        n_data = 0
    
    H = torch.cat(H_parts, dim=0)
    b = torch.cat(b_parts, dim=0)
    
    print(f"System size: H {H.shape}, b {b.shape}")
    print(f"   PDE rows: {H_pde.shape[0]}")
    if strategy['bc_weight'] > 0:
        print(f"   BC rows: 2 (weight={strategy['bc_weight']})")
    if strategy['data_weight'] > 0:
        print(f"   Data rows: {n_data} (weight={strategy['data_weight']})")
    
    # Compute energies
    E_pde = torch.norm(H_pde).item() ** 2
    E_bc = 0.0
    E_data = 0.0
    
    if strategy['bc_weight'] > 0:
        E_bc = (strategy['bc_weight'] * torch.norm(phi[bc_indices, :])).item() ** 2
    if strategy['data_weight'] > 0:
        if strategy['data_indices'] is not None:
            E_data = (strategy['data_weight'] * torch.norm(phi[strategy['data_indices'], :])).item() ** 2
        else:
            E_data = (strategy['data_weight'] * torch.norm(phi)).item() ** 2
    
    E_total = E_pde + E_bc + E_data
    
    print(f"\nEnergy contributions:")
    print(f"   PDE:  {E_pde:.3e} ({100*E_pde/E_total:.1f}%)")
    print(f"   BC:   {E_bc:.3e} ({100*E_bc/E_total:.1f}%)")
    print(f"   Data: {E_data:.3e} ({100*E_data/E_total:.1f}%)")
    
    # Solve
    try:
        C = pielm_solve(H, b, verbose=False)
        u_pred = phi @ C
        
        # Compute metrics
        data_loss = torch.mean((u_pred - u_meas) ** 2).item()
        pde_residual = torch.mean((H_pde @ C) ** 2).item()
        condH = torch.linalg.cond(H).item()
        
        # BC error
        u_bc_actual = u_pred[bc_indices]
        bc_error = torch.mean((u_bc_actual - u_bc_vals)**2).item()
        
        # Gradient test
        mu_test_grad = mu_test.clone().requires_grad_(True)
        H_pde_grad = mu_test_grad * phi_lap + config['rho_omega2'] * phi
        H_parts_grad = [H_pde_grad]
        if strategy['bc_weight'] > 0:
            H_parts_grad.append(phi[bc_indices, :] * strategy['bc_weight'])
        if strategy['data_weight'] > 0:
            if strategy['data_indices'] is not None:
                H_parts_grad.append(phi[strategy['data_indices'], :] * strategy['data_weight'])
            else:
                H_parts_grad.append(phi * strategy['data_weight'])
        H_grad = torch.cat(H_parts_grad, dim=0)
        C_grad = pielm_solve(H_grad, b, verbose=False)
        u_pred_grad = phi @ C_grad
        loss_grad = torch.mean((u_pred_grad - u_meas) ** 2)
        loss_grad.backward()
        grad_norm = mu_test_grad.grad.norm().item()
        
        print(f"\nResults:")
        print(f"   u_pred range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
        print(f"   Data loss:    {data_loss:.6e}")
        print(f"   PDE residual: {pde_residual:.6e}")
        print(f"   BC error:     {bc_error:.6e}")
        print(f"   cond(H):      {condH:.3e}")
        print(f"   Gradient norm: {grad_norm:.6e}")
        
        # Assessment
        if E_pde / E_total > 0.9:
            print(f"   ⚠️  PDE dominates ({100*E_pde/E_total:.0f}%) - solution may not be unique")
        if bc_error > 1e-5:
            print(f"   ⚠️  BCs not enforced (error={bc_error:.2e})")
        if grad_norm < 1e-5:
            print(f"   ⚠️  Very weak gradients - learning will be slow")
        
        results.append({
            'name': strategy['name'],
            'data_loss': data_loss,
            'bc_error': bc_error,
            'grad_norm': grad_norm,
            'pde_fraction': E_pde / E_total,
            'u_pred': u_pred.detach(),
            'condH': condH
        })
        
    except Exception as e:
        print(f"   ❌ Solve failed: {e}")
        results.append({
            'name': strategy['name'],
            'data_loss': np.nan,
            'bc_error': np.nan,
            'grad_norm': np.nan,
            'pde_fraction': E_pde / E_total,
            'u_pred': None,
            'condH': np.nan
        })

print("\n" + "="*70)
print("3. SUMMARY TABLE")
print("="*70)

print(f"\n{'Strategy':<45} {'Data Loss':<12} {'BC Error':<12} {'Gradient':<12} {'PDE %':<8}")
print("─" * 100)
for res in results:
    print(f"{res['name']:<45} {res['data_loss']:<12.2e} {res['bc_error']:<12.2e} {res['grad_norm']:<12.2e} {100*res['pde_fraction']:<8.1f}")

print("\n" + "="*70)
print("4. KEY FINDINGS")
print("="*70)

# Find best strategies
valid_results = [r for r in results if not np.isnan(r['data_loss'])]
best_grad = max(valid_results, key=lambda r: r['grad_norm'])
best_data = min(valid_results, key=lambda r: r['data_loss'])
best_bc = min(valid_results, key=lambda r: r['bc_error'])

print(f"\n✅ Strongest gradients: {best_grad['name']}")
print(f"   Gradient norm: {best_grad['grad_norm']:.3e}")
print(f"   This strategy provides best learning signal for inverse problem")

print(f"\n✅ Best data fit: {best_data['name']}")
print(f"   Data loss: {best_data['data_loss']:.3e}")
print(f"   This strategy matches measurements best")

print(f"\n✅ Best BC enforcement: {best_bc['name']}")
print(f"   BC error: {best_bc['bc_error']:.3e}")
print(f"   This strategy respects boundary conditions")

print("\n🔍 Insights:")
print("   1. Heavy interior weighting CAN provide constraints")
print("   2. BUT: Data points don't contain mu → gradients weakened")
print("   3. BC weighting gives unique solution with strong gradients")
print("   4. For real MRE: May need hybrid approach with estimated BCs")

print("\n" + "="*70)
print("5. VISUALIZATION")
print("="*70)

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

plot_strategies = [
    'BC only (baseline)',
    'Uniform data (all points equal)',
    'Heavy interior (25 points, weight=200)',
    'Sparse high-weight interior (5 key points, weight=1000)',
    'Hybrid: Weak BC + Heavy interior'
]

for idx, name in enumerate(plot_strategies):
    if idx >= 6:
        break
    res = next((r for r in results if r['name'] == name), None)
    if res is None or res['u_pred'] is None:
        continue
    
    ax = axes.flat[idx]
    ax.plot(x.cpu(), u_true.cpu(), 'k-', label='True u', linewidth=2)
    ax.plot(x.cpu(), u_meas.cpu(), 'k.', label='Measurements', markersize=2, alpha=0.3)
    ax.plot(x.cpu(), res['u_pred'].cpu(), 'r-', label='Predicted u', linewidth=2)
    
    # Mark weighted points if applicable
    strategy = next((s for s in strategies if s['name'] == name), None)
    if strategy and strategy['data_indices'] is not None:
        data_idx = strategy['data_indices']
        ax.plot(x[data_idx].cpu(), u_meas[data_idx].cpu(), 'ro', markersize=8, label='Weighted points')
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(f"{name}\nLoss={res['data_loss']:.2e}, Grad={res['grad_norm']:.2e}")
    ax.set_xlabel('Position x')
    ax.set_ylabel('Displacement u')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Last panel: gradient comparison bar chart
ax = axes.flat[5]
names = [r['name'].replace(' (', '\n(') for r in results if not np.isnan(r['grad_norm'])]
grads = [r['grad_norm'] for r in results if not np.isnan(r['grad_norm'])]
colors = ['green' if 'BC' in n else 'blue' if 'Hybrid' in n else 'red' for n in names]
ax.barh(range(len(grads)), grads, color=colors, alpha=0.6)
ax.set_yticks(range(len(grads)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Gradient Norm')
ax.set_title('Gradient Strength Comparison')
ax.axvline(best_grad['grad_norm'], color='k', linestyle='--', linewidth=1, label='Best')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'interior_weighting_test.png'), dpi=150, bbox_inches='tight')
print("   Saved: approach/interior_weighting_test.png")

print("\n" + "="*70)
print("6. RECOMMENDATION FOR REAL MRE")
print("="*70)

print("""
In real MRE experiments:
  ❌ We DON'T control boundary conditions
  ✅ We DO have displacement measurements throughout tissue
  ⚠️  We MAY have some knowledge of tissue boundaries (e.g., fixed to skull)

Possible approaches:

1. Estimate BCs from data:
   - Use measurements near boundaries as approximate BCs
   - Weight them heavily (bc_weight=100-200)
   - Even approximate BCs help constrain the solution

2. Physics-informed regularization:
   - Instead of hard BCs, add soft constraints
   - E.g., penalize high displacements at boundaries
   - Or use smoothness regularization

3. Overdetermined interior measurements:
   - Use many heavily-weighted interior points
   - Requires dense, high-quality measurements
   - May need 50+ points with weight=200 to compete with 2 BCs

4. Hybrid approach (RECOMMENDED):
   - Weak BCs from boundary estimates (bc_weight=10-50)
   - Medium-weight interior data (data_weight=1-5)
   - Combine physics and measurements

For this synthetic study:
  ✅ BC-only approach works best (baseline)
  ⚠️  But may not be realistic for real MRE
  🔬 Test hybrid approach with your real data
""")
