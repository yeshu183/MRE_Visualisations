"""Debug gradient flow through the PIELM forward model.

Tests whether changing mu affects the loss through the solver.
"""
import torch
from models import ForwardMREModel

torch.manual_seed(42)
device = "cpu"

# Tiny problem
N = 20
M = 10

x = torch.linspace(0, 1, N).reshape(-1, 1).to(device)
bc_indices = torch.tensor([0, N-1], dtype=torch.long, device=device)
u_bc_vals = torch.zeros(2, 1, device=device)

# True stiffness with inclusion
mu_true = (1.0 + 2.0 * torch.exp(-50 * (x - 0.5) ** 2)).to(device)

# Generate "measured" data with different basis
gen_model = ForwardMREModel(n_neurons_wave=M, input_dim=1, seed=999).to(device)
u_meas, _ = gen_model.solve_given_mu(x, mu_true, bc_indices, u_bc_vals, 400.0, bc_weight=200.0)
u_meas = u_meas.detach()  # Stop gradients from data generation

print(f"Measured wavefield shape: {u_meas.shape}, range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")

# Training model with different basis
model = ForwardMREModel(n_neurons_wave=M, input_dim=1, seed=42).to(device)

# Initial mu (should be different from mu_true)
print("\n=== Initial Forward Pass ===")
u_pred_0, mu_pred_0 = model(x, bc_indices, u_bc_vals, 400.0, bc_weight=200.0, verbose=True)
loss_0 = torch.mean((u_pred_0 - u_meas) ** 2)
print(f"Initial mu range: [{mu_pred_0.min():.3f}, {mu_pred_0.max():.3f}]")
print(f"Initial loss: {loss_0.item():.6e}")

# Compute gradients
loss_0.backward()
grad_norms = [(name, p.grad.norm().item()) for name, p in model.mu_net.named_parameters() if p.grad is not None]
print(f"\nGradient norms:")
for name, gnorm in grad_norms:
    print(f"  {name}: {gnorm:.6e}")

# Manual perturbation test
print("\n=== Manual Perturbation Test ===")
with torch.no_grad():
    # Save original params
    orig_params = [p.clone() for p in model.mu_net.parameters()]
    
    # Perturb parameters
    for p in model.mu_net.parameters():
        p.add_(torch.randn_like(p) * 0.1)
    
u_pred_1, mu_pred_1 = model(x, bc_indices, u_bc_vals, 400.0, bc_weight=200.0)
loss_1 = torch.mean((u_pred_1 - u_meas) ** 2)
print(f"Perturbed mu range: [{mu_pred_1.min():.3f}, {mu_pred_1.max():.3f}]")
print(f"Perturbed loss: {loss_1.item():.6e}")
print(f"Loss change: {(loss_1 - loss_0).item():.6e}")

if abs(loss_1.item() - loss_0.item()) < 1e-10:
    print("\n❌ PROBLEM: Loss unchanged despite mu perturbation!")
    print("This means u_pred doesn't depend on mu_pred.")
else:
    print("\n✓ Loss changes with mu perturbation (gradient chain should work)")
