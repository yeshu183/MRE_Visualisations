"""
Test the forward solver with simple cases to diagnose issues.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add approach folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'approach'))

from stiffness_network import FlexibleStiffnessNetwork
from forward_model import ForwardMREModel
from data_loader import BIOQICDataLoader


def test_homogeneous_mu():
    """Test with constant stiffness - should perfectly solve the wave equation."""
    print("\n" + "="*80)
    print("TEST 1: Homogeneous Stiffness (μ = constant)")
    print("="*80)
    
    # Load real data
    loader = BIOQICDataLoader(
        data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
        subsample=500,  # Small subset
        displacement_mode='z_component',
        seed=42
    )
    
    raw_data = loader.load()
    data = loader.to_tensors(raw_data, device='cpu')
    
    x = data['x']
    u_meas = data['u_meas']
    
    print(f"\nData:")
    print(f"  Points: {x.shape[0]}")
    print(f"  u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")
    
    # Create model with constant μ
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3,
        hidden_dim=64,
        n_layers=3,
        output_strategy='direct',
        mu_min=0.49,  # Almost constant
        mu_max=0.51   # Small range around 0.5
    )
    
    model = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=100,
        input_dim=3,
        physics_mode='effective'
    )
    
    # Get constant mu
    mu_pred = mu_net(x)
    print(f"\nConstant μ:")
    print(f"  Range: [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
    print(f"  Should be exactly 0.5")
    
    # Try to fit displacement with data constraints
    print("\n📊 Forward solve with DATA constraints (high weight)...")
    u_pred, _ = model(
        x, rho_omega2=400.0,
        bc_indices=None, u_bc_vals=None, bc_weight=0.0,
        u_data=u_meas, data_weight=100.0,
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
    
    if mse < 1e-4:
        print(f"\n  ✅ PASS: Can fit data with constant μ (MSE < 1e-4)")
    else:
        print(f"\n  ❌ FAIL: Cannot fit data even with constant μ!")
        print(f"          This suggests fundamental forward solver issue")
    
    return mse


def test_with_pde_only():
    """Test with PDE residual only (no data, no BC) - should give smooth solution."""
    print("\n" + "="*80)
    print("TEST 2: PDE Residual Only (No Data, No BC)")
    print("="*80)
    
    # Create simple grid
    N = 100
    x = torch.zeros(N, 3)
    x[:, 0] = torch.linspace(0, 1, N)
    x[:, 1] = torch.linspace(0, 1, N)
    x[:, 2] = 0.5  # Mid Z-slice
    
    # Constant mu
    mu = torch.ones(N, 1) * 0.5
    
    print(f"\nData:")
    print(f"  Points: {N}")
    print(f"  μ: constant = 0.5")
    
    # Create model
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.5, mu_max=0.5
    )
    
    model = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=50,
        input_dim=3,
        physics_mode='effective'
    )
    
    # Solve PDE only
    print("\n📊 Forward solve with PDE residual only...")
    u_pred, _ = model(
        x, rho_omega2=400.0,
        bc_indices=None, u_bc_vals=None, bc_weight=0.0,
        u_data=None, data_weight=0.0,
        verbose=True
    )
    
    print(f"\n✅ Results:")
    print(f"  u_pred range: [{u_pred.min():.3e}, {u_pred.max():.3e}]")
    print(f"  u_pred mean: {u_pred.mean():.3e}")
    print(f"  u_pred std: {u_pred.std():.3e}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x[:, 0].numpy(), u_pred.detach().numpy(), 'b-', label='u_pred')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Displacement (PDE only, no constraints)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(u_pred.detach().numpy(), bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('u')
    plt.ylabel('Frequency')
    plt.title('Distribution')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'outputs' / 'baseline' / 'test_pde_only.png', dpi=150)
    print(f"\n  💾 Saved: test_pde_only.png")
    plt.close()


def test_with_ground_truth_mu():
    """Test if we can fit data when given the TRUE stiffness field."""
    print("\n" + "="*80)
    print("TEST 3: Forward Solve with TRUE Stiffness (Best Case)")
    print("="*80)
    
    # Load data
    loader = BIOQICDataLoader(
        data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
        subsample=500,
        displacement_mode='z_component',
        seed=42
    )
    
    raw_data = loader.load()
    data = loader.to_tensors(raw_data, device='cpu')
    
    x = data['x']
    u_meas = data['u_meas']
    mu_true = data['mu_true']
    
    print(f"\nData:")
    print(f"  Points: {x.shape[0]}")
    print(f"  μ_true range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
    print(f"  u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")
    
    # Create a "cheat" network that outputs ground truth
    class GroundTruthNetwork(torch.nn.Module):
        def __init__(self, mu_values):
            super().__init__()
            self.mu_values = mu_values
            self.counter = 0
            
        def forward(self, x):
            return self.mu_values
    
    mu_net = GroundTruthNetwork(mu_true)
    
    model = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=100,
        input_dim=3,
        physics_mode='effective'
    )
    
    # Try to fit with TRUE mu and high data weight
    print("\n📊 Forward solve with TRUE μ and data constraints...")
    u_pred, _ = model(
        x, rho_omega2=400.0,
        bc_indices=None, u_bc_vals=None, bc_weight=0.0,
        u_data=u_meas, data_weight=100.0,
        verbose=True
    )
    
    # Compute error
    mse = torch.mean((u_pred - u_meas) ** 2).item()
    mae = torch.mean(torch.abs(u_pred - u_meas)).item()
    rel_error = mae / u_meas.abs().mean().item()
    
    print(f"\n✅ Results:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  Relative error: {rel_error:.2%}")
    
    if mse < 1e-3:
        print(f"\n  ✅ PASS: Can fit data with TRUE μ")
    else:
        print(f"\n  ❌ FAIL: Cannot fit data even with TRUE μ!")
        print(f"          Forward model is fundamentally broken")
    
    return mse


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FORWARD SOLVER DIAGNOSTIC TESTS")
    print("="*80)
    
    # Run tests
    mse1 = test_homogeneous_mu()
    test_with_pde_only()
    mse3 = test_with_ground_truth_mu()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test 1 (Constant μ): MSE = {mse1:.6e}")
    print(f"Test 3 (True μ):     MSE = {mse3:.6e}")
    
    if mse1 < 1e-4 and mse3 < 1e-3:
        print("\n✅ Forward solver is working!")
        print("   Problem is likely in μ network or optimization")
    else:
        print("\n❌ Forward solver has fundamental issues!")
        print("   Need to fix PDE formulation or solver")
