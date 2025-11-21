"""
Test ForwardMREModelV3 with BIOQIC data.

Goal: Pure data-driven interpolation should achieve MSE < 0.01
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model_v3 import ForwardMREModelV3
import sys


def test_data_only_v3():
    """Test pure data fitting with approach folder's method."""
    
    print("="*70)
    print("Testing ForwardMREModelV3: Data-only fitting")
    print("="*70)
    
    # Load BIOQIC data
    print("\n1. Loading BIOQIC data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None  # Will subsample later
    )
    
    sys.stdout.flush()
    data = loader.load()
    
    # Subsample for speed
    subsample = 2000
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    
    print(f"   Using {subsample} points")
    print(f"   u_meas range: [{u_meas.min():.4f}, {u_meas.max():.4f}]")
    
    # Boundary conditions (minimal - just uniqueness)
    bc_indices = torch.randperm(subsample)[:20]  # Just 20 random points
    u_bc_vals = u_meas[bc_indices]
    
    # Create networks
    print("\n2. Creating networks...")
    
    mu_network = FlexibleStiffnessNetwork(
        input_dim=3,
        hidden_dim=64,
        n_layers=3,
        output_strategy='direct',
        mu_min=0.2,
        mu_max=1.2,
        seed=42
    )
    
    # Try different omega_basis values
    omega_values = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    print("\n3. Testing different basis frequencies...")
    print("\n" + "="*70)
    print(f"{'omega':<10} {'MSE':<12} {'u_pred range':<20}")
    print("="*70)
    
    best_mse = float('inf')
    best_omega = None
    
    for omega in omega_values:
        # Create model
        model = ForwardMREModelV3(
            mu_network=mu_network,
            n_wave_neurons=200,
            omega_basis=omega,
            seed=42
        )
        
        # Forward pass - pure data-driven
        with torch.no_grad():
            u_pred, mu_pred = model(
                x,
                bc_indices=bc_indices,
                u_bc_vals=u_bc_vals,
                bc_weight=1.0,          # Low - just uniqueness
                u_data=u_meas,          # Fit all data
                data_weight=1000.0,     # High - main constraint
                use_pde=False,          # No PDE!
                verbose=(omega == 20.0)  # Verbose for omega=20
            )
        
        # Compute MSE
        mse = torch.mean((u_pred - u_meas)**2).item()
        u_min = u_pred.min().item()
        u_max = u_pred.max().item()
        
        print(f"{omega:<10.1f} {mse:<12.6f} [{u_min:.3f}, {u_max:.3f}]")
        
        if mse < best_mse:
            best_mse = mse
            best_omega = omega
    
    print("="*70)
    print(f"\n✅ Best result: omega={best_omega}, MSE={best_mse:.6f}")
    
    if best_mse < 0.01:
        print("   SUCCESS! MSE < 0.01 - data-driven approach works! 🎉")
    else:
        print(f"   Still high MSE. Expected < 0.01, got {best_mse:.6f}")
        print("   Possible issues:")
        print("   - Wave basis still incompatible")
        print("   - Need more basis functions")
        print("   - Need different basis type (polynomial, RBF)")
    
    return best_mse, best_omega


if __name__ == '__main__':
    test_data_only_v3()
