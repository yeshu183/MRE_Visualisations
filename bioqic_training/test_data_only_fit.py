"""
Test if we can fit BIOQIC data using ONLY data constraints (no PDE).
This should work if the forward solver is correct.
"""

import torch
import numpy as np
from pathlib import Path

from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model import ForwardMREModel


def test_data_only_fit():
    """Test pure data fitting without PDE constraints."""
    print("\n" + "="*80)
    print("DATA-ONLY FITTING TEST")
    print("="*80)
    print("Goal: Fit u_meas using ONLY data constraints (no PDE)")
    print("Expected: MSE < 1e-3 if forward solver works\n")
    
    # Load data
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
    
    print(f"Data:")
    print(f"  Points: {len(x)}")
    print(f"  u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")
    print(f"  ρω²: {physics_params['rho_omega2']:.1f}")
    
    # Create constant μ network (for testing)
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3,
        hidden_dim=64,
        n_layers=3,
        output_strategy='direct',
        mu_min=0.49,
        mu_max=0.51,  # Essentially constant
        seed=42
    )
    
    # Freeze network to keep μ constant
    for param in mu_net.parameters():
        param.requires_grad = False
    
    # Create forward model
    model = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=200,  # More basis functions
        input_dim=3,
        physics_mode='effective',
        omega_basis=10.0,  # Higher frequency content
        seed=42
    )
    
    mu_const = mu_net(x)
    print(f"\nConstant μ:")
    print(f"  Range: [{mu_const.min():.3f}, {mu_const.max():.3f}]")
    
    # Forward solve with DATA ONLY (no PDE, minimal BC)
    print(f"\n📊 Forward solve with DATA-ONLY (weight=1000, no PDE)...")
    
    # Minimal BC (3 points for uniqueness)
    bc_indices = torch.tensor([0, 250, 499])
    u_bc_vals = u_meas[bc_indices]
    
    with torch.no_grad():
        u_pred, _ = model(
            x, physics_params['rho_omega2'],
            bc_indices=bc_indices,
            u_bc_vals=u_bc_vals,
            bc_weight=1.0,      # Minimal
            u_data=u_meas,
            data_weight=1000.0,  # VERY HIGH - only data matters
            verbose=True
        )
    
    # Compute error
    mse = torch.mean((u_pred - u_meas) ** 2).item()
    mae = torch.mean(torch.abs(u_pred - u_meas)).item()
    
    print(f"\n✅ Results:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  u_pred range: [{u_pred.min():.3e}, {u_pred.max():.3e}]")
    print(f"  u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")
    
    if mse < 1e-3:
        print(f"\n  ✅ SUCCESS: Data fitting works!")
        print(f"     → Forward solver is correct")
        print(f"     → Problem was trying to use PDE on PDE-inconsistent data")
    else:
        print(f"\n  ❌ FAIL: MSE = {mse:.3e} still too high")
        print(f"     → Forward solver has deeper issues")
    
    return mse


if __name__ == '__main__':
    test_data_only_fit()
