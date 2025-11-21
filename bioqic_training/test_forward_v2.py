"""
Quick test of fixed forward model v2.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model_v2 import ForwardMREModel as ForwardMREModelV2

print("="*80)
print("TESTING FIXED FORWARD MODEL V2")
print("="*80)

# Load small dataset
loader = BIOQICDataLoader(
    data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
    subsample=500,
    displacement_mode='z_component',
    seed=42
)

raw_data = loader.load()
data = loader.to_tensors(raw_data, device='cpu')
physics_params = loader.get_physics_params(strategy='effective')

x = data['x']
u_meas = data['u_meas']
rho_omega2 = physics_params['rho_omega2']

print(f"\nData:")
print(f"  Points: {len(x)}")
print(f"  u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")
print(f"  ρω²: {rho_omega2}")

# Create constant mu network
mu_net = FlexibleStiffnessNetwork(
    input_dim=3,
    hidden_dim=64,
    n_layers=3,
    output_strategy='direct',
    mu_min=0.49,
    mu_max=0.51
)

# Create fixed forward model
model = ForwardMREModelV2(
    mu_network=mu_net,
    n_wave_neurons=100,
    input_dim=3,
    physics_mode='effective'
)

print(f"\n📊 Forward solve with DATA constraints (weight=100)...")
with torch.no_grad():
    u_pred, mu_pred = model(
        x, rho_omega2,
        bc_indices=None,
        u_bc_vals=None,
        bc_weight=0.0,
        u_data=u_meas,
        data_weight=100.0,
        verbose=True
    )

mse = torch.mean((u_pred - u_meas) ** 2).item()
mae = torch.mean(torch.abs(u_pred - u_meas)).item()

print(f"\n✅ Results:")
print(f"  MSE: {mse:.6e}")
print(f"  MAE: {mae:.6e}")
print(f"  u_pred range: [{u_pred.min():.3e}, {u_pred.max():.3e}]")
print(f"  u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")

if mse < 1e-3:
    print(f"\n  ✅ SUCCESS: MSE < 1e-3, forward solver working!")
else:
    print(f"\n  ❌ FAIL: MSE = {mse:.3e} still too high")
