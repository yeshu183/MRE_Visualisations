"""
Check if BIOQIC data actually satisfies the Helmholtz PDE.

The data should satisfy: ∇·(μ∇u) + ρω²u = 0

If it doesn't, then trying to fit it with PDE constraints will fail!
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

from data_loader import BIOQICDataLoader


def compute_laplacian_fd(u_3d, dx=0.001):
    """
    Compute Laplacian using finite differences.
    
    Args:
        u_3d: (nx, ny, nz) displacement field
        dx: grid spacing (meters)
        
    Returns:
        lap_u: (nx, ny, nz) Laplacian
    """
    # Use scipy's Laplace operator (6-point stencil)
    lap_u = laplace(u_3d, mode='constant', cval=0.0) / (dx**2)
    return lap_u


def check_pde_residual():
    """Check PDE residual for BIOQIC data."""
    
    print("\n" + "="*80)
    print("CHECKING PDE COMPLIANCE OF BIOQIC DATA")
    print("="*80)
    
    # Load full data (no normalization yet)
    data_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box'
    
    coords = np.load(data_dir / 'coordinates.npy')  # (80000, 3)
    displacement = np.load(data_dir / 'displacement.npy')  # (80000, 3) complex
    stiffness = np.load(data_dir / 'stiffness_ground_truth.npy')  # (80000, 1)
    
    # Use Z-component (dominant)
    u_z = np.abs(displacement[:, 2])  # (80000,)
    mu = stiffness[:, 0]  # (80000,)
    
    # Reshape to 3D grid
    grid_shape = (100, 80, 10)  # X, Y, Z
    u_3d = u_z.reshape(grid_shape)
    mu_3d = mu.reshape(grid_shape)
    
    print(f"\n📊 Data shapes:")
    print(f"   Displacement (Z): {u_3d.shape}")
    print(f"   Stiffness: {mu_3d.shape}")
    print(f"   u range: [{u_3d.min():.3e}, {u_3d.max():.3e}] m")
    print(f"   μ range: [{mu_3d.min():.0f}, {mu_3d.max():.0f}] Pa")
    
    # Physical parameters
    omega = 2 * np.pi * 60  # 60 Hz
    rho = 1000.0  # kg/m³
    rho_omega2 = rho * omega**2
    dx = 0.001  # 1mm voxel size
    
    print(f"\n⚙️  Physics:")
    print(f"   ω = {omega:.1f} rad/s")
    print(f"   ρ = {rho:.0f} kg/m³")
    print(f"   ρω² = {rho_omega2:.2e} Pa/m²")
    print(f"   Grid spacing: {dx*1000:.1f} mm")
    
    # Compute Laplacian of u
    print(f"\n🔧 Computing ∇²u using finite differences...")
    lap_u = compute_laplacian_fd(u_3d, dx)
    
    print(f"   ∇²u range: [{lap_u.min():.3e}, {lap_u.max():.3e}] m⁻²")
    
    # Compute PDE residual: R = μ∇²u + ρω²u
    # (Simplified - ignoring ∇μ·∇u term for now)
    print(f"\n🧮 Computing PDE residual: R = μ∇²u + ρω²u")
    
    R = mu_3d * lap_u + rho_omega2 * u_3d
    
    print(f"\n📊 Residual statistics:")
    print(f"   R range: [{R.min():.3e}, {R.max():.3e}]")
    print(f"   R mean: {R.mean():.3e}")
    print(f"   R std: {R.std():.3e}")
    print(f"   R RMS: {np.sqrt(np.mean(R**2)):.3e}")
    
    # Normalize by typical terms
    term1_magnitude = np.abs(mu_3d * lap_u).mean()
    term2_magnitude = np.abs(rho_omega2 * u_3d).mean()
    
    print(f"\n📊 Term magnitudes:")
    print(f"   |μ∇²u| (mean): {term1_magnitude:.3e}")
    print(f"   |ρω²u| (mean): {term2_magnitude:.3e}")
    print(f"   Ratio (term1/term2): {term1_magnitude/term2_magnitude:.3f}")
    
    # Relative residual
    total_magnitude = term1_magnitude + term2_magnitude
    relative_residual = np.abs(R).mean() / total_magnitude if total_magnitude > 0 else np.inf
    
    print(f"\n🎯 Relative PDE residual: {relative_residual:.3e}")
    
    if relative_residual < 0.01:
        print(f"   ✅ GOOD: Data satisfies PDE (residual < 1%)")
    elif relative_residual < 0.1:
        print(f"   ⚠️  OKAY: Data approximately satisfies PDE (residual < 10%)")
    else:
        print(f"   ❌ BAD: Data does NOT satisfy PDE (residual > 10%)")
        print(f"   → Forward solver will struggle to fit this data!")
    
    # Visualize residual
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Middle slice (Z=5)
    z_mid = 5
    
    # Row 1: Data
    im1 = axes[0, 0].imshow(u_3d[:, :, z_mid].T, cmap='viridis', origin='lower')
    axes[0, 0].set_title(f'Displacement u (Z={z_mid})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(mu_3d[:, :, z_mid].T, cmap='jet', origin='lower')
    axes[0, 1].set_title(f'Stiffness μ (Z={z_mid})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(lap_u[:, :, z_mid].T, cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title(f'Laplacian ∇²u (Z={z_mid})')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Terms and residual
    term1 = mu_3d * lap_u
    term2 = rho_omega2 * u_3d
    
    im4 = axes[1, 0].imshow(term1[:, :, z_mid].T, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title(f'μ∇²u (Z={z_mid})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(term2[:, :, z_mid].T, cmap='RdBu_r', origin='lower')
    axes[1, 1].set_title(f'ρω²u (Z={z_mid})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(R[:, :, z_mid].T, cmap='RdBu_r', origin='lower')
    axes[1, 2].set_title(f'Residual R = μ∇²u + ρω²u (Z={z_mid})')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / 'outputs' / 'baseline' / 'pde_compliance_check.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: {save_path}")
    
    plt.show()
    
    return relative_residual


if __name__ == '__main__':
    check_pde_residual()
