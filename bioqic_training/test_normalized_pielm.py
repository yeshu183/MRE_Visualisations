"""
Test NORMALIZED PIELM with BIOQIC data.

This should finally give non-zero solutions!
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from pielm_normalized import PIELMNormalized
import sys


class ConstantMuNetwork:
    def __init__(self, mu_values):
        self.mu_values = mu_values
    def __call__(self, x):
        return self.mu_values


def test_normalized_pielm():
    print("="*70)
    print("TESTING NORMALIZED PIELM (with ρω² scaling)")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None
    )
    sys.stdout.flush()
    data = loader.load()
    
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    print(f"\n   Physics: ρω² = {rho_omega2:.2e}")
    
    # Subsample
    subsample = 500
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    mu_true = torch.tensor(data['mu_data'][indices], dtype=torch.float32)
    
    print(f"   Subsample: {subsample} points")
    print(f"   u_meas: range [{u_meas.min():.3f}, {u_meas.max():.3f}]")
    
    # BC
    n_bc = 50
    bc_indices = torch.randperm(subsample)[:n_bc]
    u_bc_vals = u_meas[bc_indices]
    
    print(f"   BC: {n_bc} points")
    
    # Create model with ground truth mu
    print("\n2. Testing NORMALIZED PIELM...")
    mu_network = ConstantMuNetwork(mu_true)
    model = PIELMNormalized(mu_network=mu_network, poly_degree=5, seed=42)
    
    with torch.no_grad():
        u_pred, _ = model(
            x,
            rho_omega2=rho_omega2,
            bc_indices=bc_indices,
            u_bc_vals=u_bc_vals,
            bc_weight=100.0,
            u_data=u_meas,
            data_weight=0.1,
            verbose=True
        )
    
    mse = torch.mean((u_pred - u_meas)**2).item()
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"u_pred range: [{u_pred.min().item():.4f}, {u_pred.max().item():.4f}]")
    print(f"u_meas range: [{u_meas.min().item():.4f}, {u_meas.max().item():.4f}]")
    print(f"MSE: {mse:.6f}")
    
    if u_pred.abs().max() > 0.01:
        print(f"\n✅ SUCCESS! Solution is NON-ZERO!")
        print(f"   Normalization fixed the magnitude imbalance!")
        
        if mse < 0.05:
            print(f"   ✅ MSE < 0.05 - Reasonable fit!")
        elif mse < 0.20:
            print(f"   ⚠️  MSE moderate ({mse:.3f})")
            print(f"      Still better than before (was 0.198 with zero solution)")
        else:
            print(f"   ❌ MSE still high ({mse:.3f})")
            print(f"      But at least we have non-zero solution now!")
    else:
        print(f"\n❌ Solution still collapsed to zero")
        print(f"   Need even more normalization or different approach")
    
    return mse


if __name__ == '__main__':
    test_normalized_pielm()
