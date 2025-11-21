"""
Debug PIELM system construction - check if data constraints work.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_pure_data_fitting():
    """
    Test if PIELM can fit arbitrary data when data_weight >> PDE weight.
    If this fails, the problem is in the solver, not the physics.
    """
    print("\n" + "="*80)
    print("PURE DATA FITTING TEST (No Physics)")
    print("="*80)
    print("Goal: Fit arbitrary function u(x) = sin(2πx) using only data constraints")
    print("Expected: MSE < 1e-6 if solver works correctly\n")
    
    # Create 1D problem
    N = 100
    M = 50  # Basis functions
    
    # Spatial points
    x_1d = torch.linspace(0, 1, N)
    
    # Target function: u = sin(2πx)
    u_target = torch.sin(2 * np.pi * x_1d).reshape(-1, 1)
    
    # Random wave basis functions (for approximating any smooth function)
    omega = torch.randn(M) * 10  # Random frequencies
    phi = torch.sin(x_1d.reshape(-1, 1) @ omega.reshape(1, -1))  # (N, M)
    
    print(f"Setup:")
    print(f"  Points: {N}")
    print(f"  Basis functions: {M}")
    print(f"  Target: u = sin(2πx)")
    print(f"  Basis: φ_i = sin(ω_i·x), ω_i ~ N(0, 100)")
    
    # Build system with ONLY data constraints (like our forward model does)
    data_weight = 100.0
    
    H = data_weight * (phi.t() @ phi)  # (M, M)
    b = data_weight * (phi.t() @ u_target)  # (M, 1)
    
    print(f"\nSystem:")
    print(f"  H shape: {H.shape}")
    print(f"  H condition number: {torch.linalg.cond(H).item():.2e}")
    print(f"  b shape: {b.shape}")
    
    # Solve using Cholesky (same as our forward model)
    try:
        L = torch.linalg.cholesky(H)
        y = torch.linalg.solve_triangular(L, b, upper=False)
        c = torch.linalg.solve_triangular(L.t(), y, upper=True)
    except RuntimeError as e:
        print(f"\n❌ Cholesky failed: {e}")
        print("   Trying least squares instead...")
        c, _ = torch.lstsq(b, H)
    
    # Reconstruct
    u_pred = phi @ c
    
    # Compute error
    mse = torch.mean((u_pred - u_target)**2).item()
    mae = torch.mean(torch.abs(u_pred - u_target)).item()
    
    print(f"\n📊 Results:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  u_target range: [{u_target.min().item():.3f}, {u_target.max().item():.3f}]")
    print(f"  u_pred range: [{u_pred.min().item():.3f}, {u_pred.max().item():.3f}]")
    
    if mse < 1e-4:
        print(f"\n✅ PASS: Data fitting works correctly!")
    else:
        print(f"\n❌ FAIL: Cannot fit data even without physics!")
        print(f"        This indicates a fundamental problem with the basis or solver.")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(x_1d, u_target, 'b-', label='Target', linewidth=2)
    ax.plot(x_1d, u_pred, 'r--', label='Predicted', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Pure Data Fitting Test')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    error = (u_pred - u_target).numpy().flatten()
    ax.plot(x_1d, error, 'k-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Error')
    ax.set_title(f'Pointwise Error (MSE={mse:.2e})')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_pielm_data_fitting.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: debug_pielm_data_fitting.png")
    plt.show()
    
    return mse


def test_bioqic_data_fitting():
    """
    Test data fitting with actual BIOQIC displacement data.
    """
    print("\n" + "="*80)
    print("BIOQIC DATA FITTING TEST")
    print("="*80)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'approach'))
    
    from data_loader import BIOQICDataLoader
    
    # Load small subset
    loader = BIOQICDataLoader(
        data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
        subsample=200,  # Small for fast test
        displacement_mode='z_component',
        seed=42
    )
    
    raw_data = loader.load()
    data = loader.to_tensors(raw_data, 'cpu')
    
    x = data['x']  # (200, 3)
    u_meas = data['u_meas']  # (200, 1)
    
    N = x.shape[0]
    M = 100  # Basis functions
    
    print(f"\nSetup:")
    print(f"  Points: {N}")
    print(f"  Basis functions: {M}")
    print(f"  u_meas range: [{u_meas.min().item():.3f}, {u_meas.max().item():.3f}]")
    
    # Create wave basis
    omega_basis = torch.randn(3, M) * 1.0  # Standard scale
    z = x @ omega_basis  # (N, M)
    phi = torch.sin(z)
    
    # Data fitting only
    data_weight = 100.0
    H = data_weight * (phi.t() @ phi)
    b = data_weight * (phi.t() @ u_meas)
    
    print(f"  H condition number: {torch.linalg.cond(H).item():.2e}")
    
    # Solve
    L = torch.linalg.cholesky(H)
    y = torch.linalg.solve_triangular(L, b, upper=False)
    c = torch.linalg.solve_triangular(L.t(), y, upper=True)
    
    u_pred = phi @ c
    
    mse = torch.mean((u_pred - u_meas)**2).item()
    mae = torch.mean(torch.abs(u_pred - u_meas)).item()
    
    print(f"\n📊 Results:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  u_pred range: [{u_pred.min().item():.3f}, {u_pred.max().item():.3f}]")
    
    if mse < 0.01:
        print(f"\n✅ PASS: Can fit BIOQIC data with simple basis")
    else:
        print(f"\n⚠️  WARNING: High MSE - may need more basis functions or better ω scale")
    
    return mse


if __name__ == '__main__':
    mse1 = test_pure_data_fitting()
    mse2 = test_bioqic_data_fitting()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Simple 1D test MSE: {mse1:.6e}")
    print(f"BIOQIC 3D test MSE: {mse2:.6e}")
    
    if mse1 < 1e-4 and mse2 < 0.01:
        print("\n✅ Data fitting works - problem is likely in PDE formulation or scaling")
    elif mse1 < 1e-4:
        print("\n⚠️  1D works but 3D fails - likely need better basis coverage for 3D")
    else:
        print("\n❌ Even 1D fails - fundamental solver issue!")
