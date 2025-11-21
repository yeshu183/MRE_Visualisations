"""
Test proper PIELM with polynomial basis on BIOQIC data.

Key setup:
- PDE: μ∇²u + ρω²u = 0 (main physics constraint)
- BC: High weight (e.g., 100) - enforce boundary conditions strongly
- Data: Low weight (e.g., 0.1) - just regularization to guide solution
- μ: Predicted from network (not constant!)
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from pielm_polynomial import PIELMPolyModel
import sys


def test_pielm_polynomial():
    """Test proper PIELM with polynomial basis."""
    
    print("="*70)
    print("Testing PIELM with Polynomial Basis (PROPER)")
    print("="*70)
    
    # Load data
    print("\n1. Loading BIOQIC data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None
    )
    sys.stdout.flush()
    data = loader.load()
    
    # Get physics parameters
    rho = 1000.0  # kg/m³
    omega = data['scales']['omega']  # rad/s
    rho_omega2 = rho * omega**2
    
    print(f"\n   Physics parameters:")
    print(f"     ρ = {rho} kg/m³")
    print(f"     ω = {omega:.1f} rad/s ({data['scales']['frequency_hz']} Hz)")
    print(f"     ρω² = {rho_omega2:.1f}")
    
    # Subsample
    subsample = 2000
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    
    print(f"\n   Using {subsample} points")
    print(f"   u_meas range: [{u_meas.min():.4f}, {u_meas.max():.4f}]")
    
    # Boundary conditions (random sample)
    n_bc = 50
    bc_indices = torch.randperm(subsample)[:n_bc]
    u_bc_vals = u_meas[bc_indices]
    
    print(f"   BC: {n_bc} points")
    
    # Create networks
    print("\n2. Creating PIELM model...")
    
    mu_network = FlexibleStiffnessNetwork(
        input_dim=3,
        hidden_dim=64,
        n_layers=3,
        output_strategy='direct',
        mu_min=0.0,  # Normalized [0, 1]
        mu_max=1.0,
        seed=42
    )
    
    # Test different configurations
    configs = [
        {'degree': 4, 'bc_weight': 100.0, 'data_weight': 0.0, 'name': 'Pure Physics'},
        {'degree': 5, 'bc_weight': 100.0, 'data_weight': 0.1, 'name': 'Physics + Low Data'},
        {'degree': 5, 'bc_weight': 100.0, 'data_weight': 1.0, 'name': 'Physics + Med Data'},
        {'degree': 6, 'bc_weight': 100.0, 'data_weight': 0.1, 'name': 'Degree 6 + Low Data'},
    ]
    
    print("\n3. Testing different PIELM configurations...")
    print("\n" + "="*80)
    print(f"{'Config':<25} {'Degree':<8} {'BC wt':<8} {'Data wt':<10} {'MSE':<12}")
    print("="*80)
    
    best_mse = float('inf')
    best_config = None
    
    for config in configs:
        # Create model
        model = PIELMPolyModel(
            mu_network=mu_network,
            poly_degree=config['degree'],
            seed=42
        )
        
        # Forward pass
        with torch.no_grad():
            try:
                u_pred, mu_pred = model(
                    x,
                    rho_omega2=rho_omega2,
                    bc_indices=bc_indices,
                    u_bc_vals=u_bc_vals,
                    bc_weight=config['bc_weight'],
                    u_data=u_meas,
                    data_weight=config['data_weight'],
                    verbose=(config == configs[1])  # Verbose for 2nd config
                )
                
                # Compute MSE
                mse = torch.mean((u_pred - u_meas)**2).item()
                
                print(f"{config['name']:<25} {config['degree']:<8} {config['bc_weight']:<8.1f} "
                      f"{config['data_weight']:<10.3f} {mse:<12.6f}")
                
                if mse < best_mse:
                    best_mse = mse
                    best_config = config
                    
            except Exception as e:
                print(f"{config['name']:<25} ERROR: {str(e)[:40]}")
    
    print("="*80)
    
    print(f"\n✅ Best configuration: {best_config['name']}")
    print(f"   Degree: {best_config['degree']}")
    print(f"   BC weight: {best_config['bc_weight']}")
    print(f"   Data weight: {best_config['data_weight']}")
    print(f"   MSE: {best_mse:.6f}")
    
    if best_mse < 0.01:
        print(f"\n   🎉 SUCCESS! MSE < 0.01 with PHYSICS-INFORMED approach!")
    elif best_mse < 0.05:
        print(f"\n   ✓ Good! MSE < 0.05")
    else:
        print(f"\n   Note: MSE still high. Possible issues:")
        print(f"   - BIOQIC data has 85% PDE residual (viscoelastic vs elastic)")
        print(f"   - May need complex μ support")
        print(f"   - Or data-driven approach is more appropriate")
    
    return best_mse, best_config


if __name__ == '__main__':
    test_pielm_polynomial()
