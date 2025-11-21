"""
Test PIELM with GROUND TRUTH μ distribution.

This is the critical test:
- Use TRUE μ values: 3000 Pa (background), 10000 Pa (4 circles)
- Solve forward problem: Given μ, predict u
- Check if we can fit measured displacement

If this STILL gives high MSE → PDE is wrong (viscoelastic vs elastic issue)
If this gives low MSE → Problem is μ network, not PDE
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from pielm_polynomial import PIELMPolyModel
import sys


class ConstantMuNetwork:
    """Dummy network that returns ground truth μ values."""
    
    def __init__(self, mu_true: torch.Tensor):
        self.mu_true = mu_true
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Return ground truth μ for the sampled indices
        # Note: This is a hack - ideally we'd look up by coordinates
        # But since we're using the same indices, this works
        return self.mu_true


def test_with_ground_truth_mu():
    """Test PIELM forward solver with TRUE μ distribution."""
    
    print("="*70)
    print("CRITICAL TEST: PIELM with GROUND TRUTH μ")
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
    
    # Physics parameters
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    print(f"\n   Physics: ρω² = {rho_omega2:.1f}")
    
    # Subsample
    subsample = 2000
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    mu_true = torch.tensor(data['mu_data'][indices], dtype=torch.float32)  # GROUND TRUTH!
    
    print(f"\n   Using {subsample} points")
    print(f"   u_meas range: [{u_meas.min():.4f}, {u_meas.max():.4f}]")
    print(f"   μ_true range: [{mu_true.min():.4f}, {mu_true.max():.4f}] (normalized)")
    print(f"   μ_true actual: {mu_true.min()*7000+3000:.0f} to {mu_true.max()*7000+3000:.0f} Pa")
    
    # BC
    n_bc = 50
    bc_indices = torch.randperm(subsample)[:n_bc]
    u_bc_vals = u_meas[bc_indices]
    
    # Create model with GROUND TRUTH μ
    print("\n2. Creating PIELM with GROUND TRUTH μ...")
    
    # Use constant mu network that returns ground truth
    mu_network = ConstantMuNetwork(mu_true)
    
    # Test different polynomial degrees
    degrees = [3, 4, 5, 6, 7]
    
    print("\n3. Testing with TRUE μ distribution...")
    print("\n" + "="*70)
    print(f"{'Degree':<10} {'N basis':<12} {'BC wt':<10} {'Data wt':<12} {'MSE':<12}")
    print("="*70)
    
    best_mse = float('inf')
    best_config = None
    
    # Configuration: PDE-driven with some data regularization
    configs = [
        {'bc_weight': 100.0, 'data_weight': 0.0},   # Pure physics
        {'bc_weight': 100.0, 'data_weight': 0.01},  # Tiny data regularization
        {'bc_weight': 100.0, 'data_weight': 0.1},   # Low data regularization
    ]
    
    for degree in degrees:
        for config in configs:
            model = PIELMPolyModel(
                mu_network=mu_network,
                poly_degree=degree,
                seed=42
            )
            
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
                        verbose=(degree == 5 and config['data_weight'] == 0.1)  # Verbose once
                    )
                    
                    # Verify we're using ground truth μ
                    assert torch.allclose(mu_pred, mu_true, rtol=1e-5), "Not using ground truth μ!"
                    
                    mse = torch.mean((u_pred - u_meas)**2).item()
                    
                    print(f"{degree:<10} {model.n_basis:<12} {config['bc_weight']:<10.1f} "
                          f"{config['data_weight']:<12.3f} {mse:<12.6f}")
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_config = {'degree': degree, **config}
                        
                except Exception as e:
                    print(f"{degree:<10} {'ERROR':<12} {str(e)[:40]}")
    
    print("="*70)
    
    print(f"\n✅ Best result with GROUND TRUTH μ:")
    print(f"   Degree: {best_config['degree']}")
    print(f"   BC weight: {best_config['bc_weight']}")
    print(f"   Data weight: {best_config['data_weight']}")
    print(f"   MSE: {best_mse:.6f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    
    if best_mse < 0.01:
        print("✅ MSE < 0.01 → PDE is CORRECT!")
        print("   Problem was the μ network (wrong predictions)")
        print("   Solution: Train μ network better")
    elif best_mse < 0.05:
        print("⚠️  MSE moderate (0.01-0.05)")
        print("   PDE might be approximately correct but needs tuning")
        print("   Or polynomial degree insufficient")
    else:
        print("❌ MSE > 0.05 → PDE is WRONG!")
        print(f"   Even with PERFECT μ, we get MSE = {best_mse:.6f}")
        print("   ROOT CAUSE: BIOQIC uses viscoelastic (complex μ)")
        print("              We use elastic (real μ only)")
        print("   SOLUTION: Either:")
        print("     1. Use data-driven approach (no PDE)")
        print("     2. Implement complex μ support")
        print("     3. Use different dataset with elastic physics")
    
    return best_mse, best_config


if __name__ == '__main__':
    test_with_ground_truth_mu()
