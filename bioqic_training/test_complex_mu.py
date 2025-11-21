"""
Test PIELM with COMPLEX μ (Voigt viscoelastic model).

BIOQIC uses: μ* = μ_storage + iωη
- Background: μ* = 3000 + 377i Pa  
- Targets: μ* = 10000 + 377i Pa
- We've been using only REAL part → that's why MSE = 0.198!

Now test with COMPLEX μ to see if it improves.
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from pielm_polynomial import PIELMPolyModel
import sys


class ComplexMuNetwork:
    """Return ground truth COMPLEX μ values."""
    
    def __init__(self, mu_true_complex: torch.Tensor):
        """
        Args:
            mu_true_complex: Complex stiffness (N, 1) 
        """
        self.mu_true = mu_true_complex
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu_true


def test_with_complex_mu():
    """Test PIELM with COMPLEX ground truth μ."""
    
    print("="*70)
    print("TEST: PIELM with COMPLEX μ (Voigt viscoelastic)")
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
    
    # Physics
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    print(f"\n   Physics: ω = {omega:.1f} rad/s, ρω² = {rho_omega2:.1f}")
    
    # Subsample
    subsample = 2000
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    mu_norm = torch.tensor(data['mu_data'][indices], dtype=torch.float32)  # Normalized real part
    
    # Reconstruct COMPLEX μ in physical units
    # Storage modulus (real part)
    mu_storage = mu_norm * 7000 + 3000  # Denormalize: [0,1] → [3000, 10000] Pa
    
    # Loss modulus (imaginary part) - constant everywhere!
    eta = 1.0  # Pa·s
    mu_loss = omega * eta * torch.ones_like(mu_storage)  # 377 Pa everywhere
    
    # Complex stiffness
    mu_complex = mu_storage + 1j * mu_loss
    
    # For now, test with just the MAGNITUDE of complex μ
    mu_magnitude = torch.abs(mu_complex)
    
    print(f"\n   Stiffness (complex):")
    print(f"     Storage (μ'): [{mu_storage.min().item():.0f}, {mu_storage.max().item():.0f}] Pa")
    print(f"     Loss (μ''): {mu_loss[0].item():.0f} Pa (constant)")
    print(f"     Magnitude |μ*|: [{mu_magnitude.min().item():.0f}, {mu_magnitude.max().item():.0f}] Pa")
    
    # Normalize magnitude for network
    mu_mag_min = 3000
    mu_mag_max = np.sqrt(10000**2 + 377**2)  # ~10007 Pa
    mu_mag_norm = (mu_magnitude - mu_mag_min) / (mu_mag_max - mu_mag_min)
    
    print(f"     Normalized |μ*|: [{mu_mag_norm.min().item():.4f}, {mu_mag_norm.max().item():.4f}]")
    
    # BC
    n_bc = 50
    bc_indices = torch.randperm(subsample)[:n_bc]
    u_bc_vals = u_meas[bc_indices]
    
    # Test 1: Using REAL part only (baseline)
    print("\n2. Test 1: REAL μ only (baseline - should give MSE ~0.20)...")
    mu_network_real = ComplexMuNetwork(mu_norm)
    model_real = PIELMPolyModel(mu_network=mu_network_real, poly_degree=5, seed=42)
    
    with torch.no_grad():
        u_pred_real, _ = model_real(
            x, rho_omega2=rho_omega2,
            bc_indices=bc_indices, u_bc_vals=u_bc_vals, bc_weight=100.0,
            u_data=u_meas, data_weight=0.1, verbose=False
        )
    
    mse_real = torch.mean((u_pred_real - u_meas)**2).item()
    print(f"   MSE (real μ only): {mse_real:.6f}")
    
    # Test 2: Using MAGNITUDE |μ*|
    print("\n3. Test 2: MAGNITUDE |μ*| (includes loss modulus effect)...")
    mu_network_mag = ComplexMuNetwork(mu_mag_norm)
    model_mag = PIELMPolyModel(mu_network=mu_network_mag, poly_degree=5, seed=42)
    
    with torch.no_grad():
        u_pred_mag, _ = model_mag(
            x, rho_omega2=rho_omega2,
            bc_indices=bc_indices, u_bc_vals=u_bc_vals, bc_weight=100.0,
            u_data=u_meas, data_weight=0.1, verbose=True
        )
    
    mse_mag = torch.mean((u_pred_mag - u_meas)**2).item()
    print(f"   MSE (|μ*|): {mse_mag:.6f}")
    
    # Compare
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Real μ only:      MSE = {mse_real:.6f}")
    print(f"Magnitude |μ*|:   MSE = {mse_mag:.6f}")
    
    if mse_mag < mse_real:
        improvement = (mse_real - mse_mag) / mse_real * 100
        print(f"\n✅ Using |μ*| improves by {improvement:.1f}%!")
    else:
        print(f"\n⚠️  No improvement - magnitude alone not sufficient")
    
    if mse_mag > 0.05:
        print(f"\n❌ MSE still high ({mse_mag:.3f})")
        print("   ROOT CAUSE: Our PDE is REAL, but BIOQIC used COMPLEX PDE")
        print("   Our PDE:    μ∇²u + ρω²u = 0  (elastic)")
        print("   BIOQIC PDE: μ*∇²u + ρω²u = 0 (viscoelastic, μ* complex)")
        print("\n   SOLUTION OPTIONS:")
        print("   1. Implement complex-valued PIELM (μ*, u both complex)")
        print("   2. Use data-driven approach (no PDE enforcement)")
        print("   3. Use different dataset with purely elastic physics")
    
    return mse_real, mse_mag


if __name__ == '__main__':
    test_with_complex_mu()
