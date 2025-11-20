"""Forward-only solve: Test if data constraints allow wrong mu to fit data.

This tests the hypothesis: Can we fit measurements well even with wrong mu
when using data constraints? If yes, that explains why gradients fail.
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
print("FORWARD SOLVE: Can Wrong Mu Fit Data?")
print("="*70)

# Setup
n_points = 100
x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)

# True mu (Gaussian bump)
mu_true = 1.0 + 1.0 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))

print("\n1. Generating reference data with TRUE mu...")
print(f"   True mu range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")

torch.manual_seed(config['seed'])
model = ForwardMREModel(
    n_neurons_wave=config['n_wave_neurons'],
    input_dim=1,
    seed=config['seed']
).to(device)

phi, phi_lap = model.get_basis_and_laplacian(x)

bc_indices = torch.tensor([0, n_points - 1], dtype=torch.long, device=device)
u_bc_vals = torch.tensor([[0.01], [0.0]], device=device)

# Solve with true mu (BC only, no data)
H_pde_true = mu_true * phi_lap + config['rho_omega2'] * phi
b_pde = torch.zeros(n_points, 1, device=device)
H_bc = phi[bc_indices, :]

H_true = torch.cat([H_pde_true, 200.0 * H_bc], dim=0)
b_true = torch.cat([b_pde, 200.0 * u_bc_vals], dim=0)

C_true = pielm_solve(H_true, b_true, verbose=False)
u_true = phi @ C_true

print(f"   True u range: [{u_true.min():.6f}, {u_true.max():.6f}]")

# Add noise
u_meas = u_true + 0.001 * torch.randn_like(u_true)
print(f"   Measurements range: [{u_meas.min():.6f}, {u_meas.max():.6f}]")

print("\n" + "="*70)
print("2. Testing Different WRONG Mu Values")
print("="*70)

# Test several wrong mu values
test_mus = [
    ("Constant 1.0", torch.ones_like(mu_true) * 1.0),
    ("Constant 1.5", torch.ones_like(mu_true) * 1.5),
    ("Constant 2.0", torch.ones_like(mu_true) * 2.0),
    ("Linear ramp", torch.linspace(1.0, 2.0, n_points, device=device).reshape(-1, 1)),
    ("Inverted Gaussian", 1.0 + 1.0 * torch.exp(-((x - 0.8) ** 2) / (2 * 0.1**2))),
]

results = []

for name, mu_test in test_mus:
    print(f"\n{'─'*70}")
    print(f"Testing: {name}")
    print(f"  Mu range: [{mu_test.min():.3f}, {mu_test.max():.3f}]")
    print(f"  Mu error: MSE={torch.mean((mu_test - mu_true)**2).item():.3f}")
    
    H_pde_test = mu_test * phi_lap + config['rho_omega2'] * phi
    
    # Test 3 scenarios
    scenarios = [
        ("BC only (weight=200)", 200.0, 0.0),
        ("BC + Data (weak)", 200.0, 0.1),
        ("BC + Data (medium)", 200.0, 1.0),
        ("BC + Data (strong)", 200.0, 5.0),
    ]
    
    for scenario_name, bc_w, data_w in scenarios:
        if data_w > 0:
            H_data = phi
            b_data = u_meas
            H = torch.cat([H_pde_test, bc_w * H_bc, data_w * H_data], dim=0)
            b = torch.cat([b_pde, bc_w * u_bc_vals, data_w * b_data], dim=0)
        else:
            H = torch.cat([H_pde_test, bc_w * H_bc], dim=0)
            b = torch.cat([b_pde, bc_w * u_bc_vals], dim=0)
        
        C_test = pielm_solve(H, b, verbose=False)
        u_pred = phi @ C_test
        
        data_loss = torch.mean((u_pred - u_meas) ** 2).item()
        pde_residual = torch.mean((H_pde_test @ C_test) ** 2).item()
        
        print(f"  {scenario_name:25s}: Data loss={data_loss:.2e}, PDE residual={pde_residual:.2e}")
        
        results.append({
            'mu_name': name,
            'mu': mu_test,
            'scenario': scenario_name,
            'u_pred': u_pred,
            'data_loss': data_loss,
            'pde_residual': pde_residual,
        })

print("\n" + "="*70)
print("3. KEY FINDINGS")
print("="*70)

print("\n🔍 Analysis:")
print("   If data loss is LOW with WRONG mu:")
print("   → Data constraints let wrong mu fit measurements")
print("   → Inverse problem cannot distinguish correct mu")
print("   → Gradients will not improve mu")
print()
print("   If data loss is HIGH with wrong mu:")
print("   → Only correct mu can fit measurements")
print("   → Inverse problem can work")
print("   → Gradients will guide toward correct mu")

# Find best and worst for each scenario
print("\n" + "="*70)
print("4. COMPARISON BY SCENARIO")
print("="*70)

scenarios_list = list(set([r['scenario'] for r in results]))
for scenario in scenarios_list:
    print(f"\n{scenario}:")
    scenario_results = [r for r in results if r['scenario'] == scenario]
    scenario_results.sort(key=lambda r: r['data_loss'])
    
    print("  Best fit (lowest data loss):")
    best = scenario_results[0]
    print(f"    {best['mu_name']}: loss={best['data_loss']:.3e}")
    
    print("  Worst fit (highest data loss):")
    worst = scenario_results[-1]
    print(f"    {worst['mu_name']}: loss={worst['data_loss']:.3e}")
    
    print(f"  Ratio (worst/best): {worst['data_loss']/best['data_loss']:.1f}×")
    
    if worst['data_loss'] / best['data_loss'] < 2.0:
        print("  ❌ PROBLEM: All mu values fit similarly well!")
        print("     → Inverse problem cannot work with this setup")
    else:
        print("  ✅ GOOD: Wrong mu gives worse fit")
        print("     → Inverse problem has gradient signal")

# Visualize one case
print("\n" + "="*70)
print("5. VISUALIZATION: Constant mu=1.5 vs True mu")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Get results for constant mu=1.5
const_results = [r for r in results if r['mu_name'] == "Constant 1.5"]

for idx, result in enumerate(const_results):
    ax = axes.flat[idx]
    ax.plot(x.cpu(), u_meas.cpu(), 'k.', label='Measurements', alpha=0.5, markersize=2)
    ax.plot(x.cpu(), u_true.cpu(), 'k-', label='True u', linewidth=2)
    ax.plot(x.cpu(), result['u_pred'].detach().cpu(), 'r-', label='Predicted u', linewidth=1.5)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Displacement u')
    ax.set_title(f"{result['scenario']}\nData loss={result['data_loss']:.3e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'forward_solve_data_test.png'), dpi=150, bbox_inches='tight')
print("  Saved: approach/forward_solve_data_test.png")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
If you see that:
  - BC only: Different mu → very different data loss (good!)
  - BC + Data: Different mu → similar data loss (bad!)

Then data constraints are SHORT-CIRCUITING the physics:
  → The solver fits data regardless of mu
  → Gradients to mu are meaningless
  → Inverse problem cannot work

This explains why your plots showed flat mu with data constraints!
""")
