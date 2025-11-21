"""
Test polynomial basis forward model with BIOQIC data.

Expected: MSE < 0.01 (polynomial should work much better than wave!)
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model_polynomial import ForwardMREModelPolynomial
import sys


def test_polynomial_basis():
    """Test polynomial basis for data fitting."""
    
    print("="*70)
    print("Testing Polynomial Basis: Data-only fitting")
    print("="*70)
    
    # Load BIOQIC data
    print("\n1. Loading BIOQIC data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None
    )
    
    sys.stdout.flush()
    data = loader.load()
    
    # Subsample
    subsample = 2000
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    
    print(f"   Using {subsample} points")
    print(f"   u_meas range: [{u_meas.min():.4f}, {u_meas.max():.4f}]")
    
    # Minimal BC (just uniqueness)
    bc_indices = torch.randperm(subsample)[:20]
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
    
    # Test different polynomial degrees
    degrees = [2, 3, 4, 5, 6, 7, 8]
    
    print("\n3. Testing different polynomial degrees...")
    print("\n" + "="*70)
    print(f"{'Degree':<10} {'N basis':<12} {'MSE':<12} {'u_pred range':<20}")
    print("="*70)
    
    best_mse = float('inf')
    best_degree = None
    
    for degree in degrees:
        # Create model
        model = ForwardMREModelPolynomial(
            mu_network=mu_network,
            poly_degree=degree,
            seed=42
        )
        
        print(f"\n  Testing degree {degree}...")
        
        # Forward pass - pure data-driven
        with torch.no_grad():
            try:
                u_pred, mu_pred = model(
                    x,
                    bc_indices=bc_indices,
                    u_bc_vals=u_bc_vals,
                    bc_weight=1.0,
                    u_data=u_meas,
                    data_weight=1000.0,
                    verbose=(degree == 3)  # Verbose for degree 3
                )
                
                # Compute MSE
                mse = torch.mean((u_pred - u_meas)**2).item()
                u_min = u_pred.min().item()
                u_max = u_pred.max().item()
                
                print(f"{degree:<10} {model.n_basis:<12} {mse:<12.6f} [{u_min:.3f}, {u_max:.3f}]")
                
                if mse < best_mse:
                    best_mse = mse
                    best_degree = degree
                    
            except Exception as e:
                print(f"{degree:<10} {'ERROR':<12} {str(e)[:40]}")
    
    print("="*70)
    print(f"\n✅ Best result: degree={best_degree}, MSE={best_mse:.6f}")
    
    if best_mse < 0.01:
        print(f"   🎉 SUCCESS! MSE < 0.01 - Polynomial basis works!")
        print(f"   This is {0.166/best_mse:.1f}× better than wave basis (MSE=0.166)")
    elif best_mse < 0.05:
        print(f"   ✓ Good! MSE < 0.05 - Much better than wave basis (0.166)")
        print(f"   Improvement: {0.166/best_mse:.1f}×")
    else:
        print(f"   Still high MSE. Expected < 0.01, got {best_mse:.6f}")
        print(f"   But still better than wave basis: {0.166/best_mse:.1f}× improvement")
    
    return best_mse, best_degree


if __name__ == '__main__':
    test_polynomial_basis()
