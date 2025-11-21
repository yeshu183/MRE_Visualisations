"""
Test different basis function frequencies to find what works for BIOQIC data.
"""

import torch
import numpy as np
from pathlib import Path

from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model import ForwardMREModel


def test_frequency(omega_basis, n_neurons=200):
    """Test data fitting with specific basis frequency."""
    
    # Load data
    loader = BIOQICDataLoader(
        data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
        subsample=500,
        displacement_mode='z_component',
        seed=42
    )
    
    # Suppress output
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    raw_data = loader.load()
    data = loader.to_tensors(raw_data, device='cpu')
    physics_params = loader.get_physics_params(strategy='effective')
    
    sys.stdout = old_stdout
    
    x = data['x']
    u_meas = data['u_meas']
    
    # Constant μ network
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.49, mu_max=0.51, seed=42
    )
    for param in mu_net.parameters():
        param.requires_grad = False
    
    # Forward model with specific omega_basis
    model = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=n_neurons,
        input_dim=3,
        physics_mode='effective',
        omega_basis=omega_basis,
        seed=42
    )
    
    # Minimal BC
    bc_indices = torch.tensor([0, 250, 499])
    u_bc_vals = u_meas[bc_indices]
    
    # Forward solve with data-only
    with torch.no_grad():
        u_pred, _ = model(
            x, physics_params['rho_omega2'],
            bc_indices=bc_indices,
            u_bc_vals=u_bc_vals,
            bc_weight=1.0,
            u_data=u_meas,
            data_weight=1000.0,
            verbose=False
        )
    
    # Compute error
    mse = torch.mean((u_pred - u_meas) ** 2).item()
    mae = torch.mean(torch.abs(u_pred - u_meas)).item()
    
    # Compute scale match
    u_pred_std = u_pred.std().item()
    u_meas_std = u_meas.std().item()
    scale_ratio = u_pred_std / u_meas_std
    
    return mse, mae, scale_ratio, u_pred.min().item(), u_pred.max().item()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING DIFFERENT BASIS FREQUENCIES")
    print("="*80)
    print("Goal: Find omega_basis that allows fitting BIOQIC displacement")
    print()
    
    # Test range of frequencies
    omega_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    print(f"{'omega_basis':<12} {'MSE':<12} {'MAE':<12} {'Scale Ratio':<12} {'u_pred range'}")
    print("-" * 80)
    
    best_mse = float('inf')
    best_omega = None
    
    for omega in omega_values:
        mse, mae, scale, u_min, u_max = test_frequency(omega)
        print(f"{omega:<12.1f} {mse:<12.6f} {mae:<12.6f} {scale:<12.3f} [{u_min:+.3f}, {u_max:+.3f}]")
        
        if mse < best_mse:
            best_mse = mse
            best_omega = omega
    
    print("-" * 80)
    print(f"\n✅ Best omega_basis: {best_omega} (MSE = {best_mse:.6f})")
    
    if best_mse < 1e-2:
        print(f"   SUCCESS: Can fit data with appropriate basis frequencies!")
    else:
        print(f"   ❌ FAIL: Even best omega gives MSE = {best_mse:.3e}")
        print(f"   → Wave basis fundamentally incompatible with this data")
        print(f"   → Need different basis (polynomial, RBF, or learned)")
