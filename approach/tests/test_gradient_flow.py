"""Debug script to test gradient flow through the modular code."""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_generators import generate_gaussian_bump
from models import ForwardMREModel

# Load config
with open(os.path.join(os.path.dirname(__file__), '..', 'config_forward.json'), 'r') as f:
    config = json.load(f)

device = torch.device('cpu')

print("="*70)
print("GRADIENT FLOW TEST")
print("="*70)

# Generate data
print("\n1. Generating synthetic data...")
x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_gaussian_bump(
    n_points=config['n_points'],
    n_wave_neurons=config['n_wave_neurons'],
    device=device,
    seed=config['seed']
)
print(f"   Data generated: {len(x)} points")
print(f"   Ground truth mu range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
print(f"   Measurement range: [{u_meas.min():.6f}, {u_meas.max():.6f}]")

# Create training model with SAME seed
print("\n2. Creating training model...")
model = ForwardMREModel(
    n_neurons_wave=config['n_wave_neurons'],
    input_dim=1,
    seed=config['seed']
).to(device)
print(f"   Model created with seed={config['seed']}")
print(f"   Number of parameters: {sum(p.numel() for p in model.mu_net.parameters())}")

# Forward pass
print("\n3. Forward pass...")
u_pred, mu_pred = model(x, bc_indices, u_bc_vals, 
                       config['rho_omega2'], 
                       bc_weight=config['bc_weight'],
                       verbose=True)
print(f"   u_pred range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
print(f"   mu_pred range: [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
print(f"   mu_pred unique values: {torch.unique(mu_pred).shape[0]}")

# Compute loss
loss = torch.mean((u_pred - u_meas) ** 2)
print(f"\n4. Loss computation...")
print(f"   Data loss (MSE): {loss.item():.6e}")

# Backward pass
print("\n5. Backward pass...")
loss.backward()

# Check gradients
print("\n6. Gradient analysis...")
has_grads = False
for name, param in model.mu_net.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        print(f"   {name:20s}: norm={grad_norm:.6e}, mean={grad_mean:.6e}, std={grad_std:.6e}")
        if grad_norm > 0:
            has_grads = True
    else:
        print(f"   {name:20s}: NO GRADIENT")

print("\n"+"="*70)
if has_grads:
    print("✅ GRADIENT FLOW: WORKING")
else:
    print("❌ GRADIENT FLOW: BROKEN (all gradients are zero or None)")
print("="*70)

# Additional diagnostics
print("\n7. Additional diagnostics...")
print(f"   requires_grad for mu_pred: {mu_pred.requires_grad}")
print(f"   requires_grad for u_pred: {u_pred.requires_grad}")
print(f"   requires_grad for loss: {loss.requires_grad}")

# Check if mu_pred is actually connected to the computation graph
print(f"\n   Checking if mu_pred has grad_fn: {mu_pred.grad_fn is not None}")
print(f"   Checking if u_pred has grad_fn: {u_pred.grad_fn is not None}")
