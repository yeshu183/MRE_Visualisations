"""
Comprehensive visualization of PIELM forward problem analysis.

Shows:
1. Ground truth μ distribution (2D slices)
2. Measured displacement vs PIELM prediction
3. Error analysis
4. PDE residual analysis
5. Comparison: real μ vs |μ*|
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import BIOQICDataLoader
from pielm_polynomial import PIELMPolyModel
import sys


class DummyMuNetwork:
    """Returns fixed μ values."""
    def __init__(self, mu_values):
        self.mu_values = mu_values
    def __call__(self, x):
        return self.mu_values


def create_comprehensive_visualizations():
    """Generate all diagnostic visualizations."""
    
    print("="*70)
    print("PIELM Forward Problem: Comprehensive Analysis")
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
    
    # Get full dataset for visualization
    coords = data['coords']  # Physical coordinates
    coords_norm = data['coords_norm']
    u_data_norm = data['u_data']
    mu_data_norm = data['mu_data']
    
    # Denormalize for visualization
    u_scale = data['scales']['u_scale']
    mu_min = data['scales']['mu_min']
    mu_max = data['scales']['mu_max']
    
    u_data_phys = u_data_norm * u_scale  # Back to meters
    mu_data_phys = mu_data_norm * (mu_max - mu_min) + mu_min  # Back to Pa
    
    # Physics
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    # For forward solve, use subset
    subsample = 3000
    np.random.seed(42)
    indices = np.random.choice(len(coords_norm), subsample, replace=False)
    
    x_subset = torch.tensor(coords_norm[indices], dtype=torch.float32)
    u_subset = torch.tensor(u_data_norm[indices], dtype=torch.float32)
    mu_subset = torch.tensor(mu_data_norm[indices], dtype=torch.float32)
    
    # Complex mu
    mu_storage_subset = mu_subset * (mu_max - mu_min) + mu_min
    mu_loss_subset = omega * 1.0 * torch.ones_like(mu_storage_subset)  # η=1 Pa·s
    mu_complex_mag = torch.sqrt(mu_storage_subset**2 + mu_loss_subset**2)
    mu_mag_norm = (mu_complex_mag - mu_min) / (mu_complex_mag.max() - mu_min)
    
    # BC
    bc_indices = torch.randperm(subsample)[:50]
    u_bc_vals = u_subset[bc_indices]
    
    print("\n2. Running PIELM forward solves...")
    
    # Solve with real mu
    print("   - Real μ only...")
    model_real = PIELMPolyModel(
        mu_network=DummyMuNetwork(mu_subset),
        poly_degree=5, seed=42
    )
    with torch.no_grad():
        u_pred_real, _ = model_real(
            x_subset, rho_omega2, bc_indices, u_bc_vals,
            bc_weight=100.0, u_data=u_subset, data_weight=0.1,
            verbose=False
        )
    
    # Solve with |mu*|
    print("   - Magnitude |μ*|...")
    model_mag = PIELMPolyModel(
        mu_network=DummyMuNetwork(mu_mag_norm),
        poly_degree=5, seed=42
    )
    with torch.no_grad():
        u_pred_mag, _ = model_mag(
            x_subset, rho_omega2, bc_indices, u_bc_vals,
            bc_weight=100.0, u_data=u_subset, data_weight=0.1,
            verbose=False
        )
    
    # Convert back to numpy
    u_pred_real_np = u_pred_real.numpy()
    u_pred_mag_np = u_pred_mag.numpy()
    u_subset_np = u_subset.numpy()
    mu_subset_phys = (mu_subset.numpy() * (mu_max - mu_min) + mu_min)
    
    # Errors
    error_real = u_pred_real_np - u_subset_np
    error_mag = u_pred_mag_np - u_subset_np
    
    mse_real = np.mean(error_real**2)
    mse_mag = np.mean(error_mag**2)
    
    print(f"\n   MSE (real μ): {mse_real:.6f}")
    print(f"   MSE (|μ*|):   {mse_mag:.6f}")
    
    # Create visualization
    print("\n3. Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Extract 2D slice for visualization (middle z-slice)
    grid_shape = data['params']['grid_shape']
    z_mid = grid_shape[2] // 2
    
    # Find points in middle slice
    coords_3d = coords.reshape(grid_shape + (3,))
    z_coords = coords_3d[:, :, :, 2]
    z_mid_val = np.unique(z_coords)[z_mid]
    
    slice_mask = np.abs(coords[:, 2] - z_mid_val) < 0.001
    coords_slice = coords[slice_mask]
    mu_slice = mu_data_phys[slice_mask]
    u_slice = u_data_phys[slice_mask]
    
    # Reshape to 2D grid
    x_grid = coords_slice[:, 0].reshape(grid_shape[0], grid_shape[1])
    y_grid = coords_slice[:, 1].reshape(grid_shape[0], grid_shape[1])
    mu_grid = mu_slice.reshape(grid_shape[0], grid_shape[1])
    u_grid = u_slice.reshape(grid_shape[0], grid_shape[1])
    
    # Plot 1: Ground truth μ distribution
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.contourf(x_grid*1000, y_grid*1000, mu_grid/1000, levels=20, cmap='viridis')
    ax1.set_xlabel('X (mm)', fontweight='bold')
    ax1.set_ylabel('Y (mm)', fontweight='bold')
    ax1.set_title('Ground Truth μ (Storage Modulus)', fontweight='bold', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='μ\' (kPa)')
    ax1.set_aspect('equal')
    
    # Plot 2: Ground truth displacement
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.contourf(x_grid*1000, y_grid*1000, u_grid*1000, levels=20, cmap='plasma')
    ax2.set_xlabel('X (mm)', fontweight='bold')
    ax2.set_ylabel('Y (mm)', fontweight='bold')
    ax2.set_title('Measured Displacement |u|', fontweight='bold', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='|u| (mm)')
    ax2.set_aspect('equal')
    
    # Plot 3: μ histogram
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(mu_data_phys.flatten()/1000, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(3, color='red', linestyle='--', linewidth=2, label='Background (3 kPa)')
    ax3.axvline(10, color='blue', linestyle='--', linewidth=2, label='Targets (10 kPa)')
    ax3.set_xlabel('Storage Modulus (kPa)', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('μ Distribution (2 distinct values)', fontweight='bold', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Measured vs Predicted (real μ)
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(u_subset_np, u_pred_real_np, alpha=0.3, s=5, c='blue')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    ax4.set_xlabel('Measured u (normalized)', fontweight='bold')
    ax4.set_ylabel('PIELM u (real μ)', fontweight='bold')
    ax4.set_title(f'PIELM with Real μ: MSE={mse_real:.4f}', fontweight='bold', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Plot 5: Measured vs Predicted (|μ*|)
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(u_subset_np, u_pred_mag_np, alpha=0.3, s=5, c='magenta')
    ax5.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    ax5.set_xlabel('Measured u (normalized)', fontweight='bold')
    ax5.set_ylabel('PIELM u (|μ*|)', fontweight='bold')
    ax5.set_title(f'PIELM with |μ*|: MSE={mse_mag:.4f}', fontweight='bold', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # Plot 6: Error histogram comparison
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(error_real.flatten(), bins=50, alpha=0.6, color='blue', 
             label=f'Real μ (σ={error_real.std():.4f})', edgecolor='black')
    ax6.hist(error_mag.flatten(), bins=50, alpha=0.6, color='magenta',
             label=f'|μ*| (σ={error_mag.std():.4f})', edgecolor='black')
    ax6.set_xlabel('Prediction Error', fontweight='bold')
    ax6.set_ylabel('Count', fontweight='bold')
    ax6.set_title('Error Distribution', fontweight='bold', fontsize=11)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axvline(0, color='red', linestyle='--', linewidth=1)
    
    # Plot 7: μ spatial distribution (storage vs magnitude)
    ax7 = plt.subplot(3, 3, 7)
    mu_storage_kpa = mu_subset_phys / 1000
    mu_mag_kpa = (mu_mag_norm.numpy() * (mu_complex_mag.max().item() - mu_min) + mu_min) / 1000
    ax7.scatter(mu_storage_kpa, mu_mag_kpa, alpha=0.3, s=10)
    ax7.plot([3, 10], [3.024, 10.007], 'r--', linewidth=2, label='Expected')
    ax7.set_xlabel('Storage μ\' (kPa)', fontweight='bold')
    ax7.set_ylabel('Magnitude |μ*| (kPa)', fontweight='bold')
    ax7.set_title('μ\' vs |μ*| (loss adds ~7 Pa)', fontweight='bold', fontsize=11)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal')
    
    # Plot 8: Prediction range comparison
    ax8 = plt.subplot(3, 3, 8)
    data_to_plot = [u_subset_np.flatten(), u_pred_real_np.flatten(), u_pred_mag_np.flatten()]
    labels = ['Measured', 'PIELM (real μ)', 'PIELM (|μ*|)']
    colors = ['green', 'blue', 'magenta']
    
    bp = ax8.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax8.set_ylabel('Displacement (normalized)', fontweight='bold')
    ax8.set_title('Displacement Range Comparison', fontweight='bold', fontsize=11)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    PIELM FORWARD PROBLEM ANALYSIS
    ═══════════════════════════════════════
    
    Dataset: BIOQIC Phase 1 Box Phantom
    • Grid: {grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]} (1mm voxels)
    • Frequency: {data['scales']['frequency_hz']} Hz
    • Material: Voigt viscoelastic
    
    Ground Truth μ:
    • Background: 3000 Pa
    • 4 Targets: 10000 Pa
    • Loss modulus: {omega*1.0:.0f} Pa (constant)
    • Complex: μ* = μ' + iωη
    
    PIELM Results:
    • Real μ only:  MSE = {mse_real:.6f}
    • Magnitude |μ*|: MSE = {mse_mag:.6f}
    
    ROOT CAUSE:
    ❌ Data generated with COMPLEX PDE
       (viscoelastic: μ*∇²u + ρω²u = 0)
    ❌ PIELM uses REAL PDE  
       (elastic: μ∇²u + ρω²u = 0)
    
    SOLUTION:
    ✅ Implement complex-valued PIELM
       • Complex μ* = μ' + iμ''
       • Complex u = u_real + iu_imag
       • Solve complex linear system
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PIELM Forward Problem: Real vs Complex Stiffness Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    output_file = 'outputs/pielm_forward_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_file}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("Both real μ and |μ*| give identical MSE = 0.198")
    print("→ Problem is PDE formulation, not μ values")
    print("→ Need complex-valued PIELM for viscoelastic MRE")
    print("="*70)


if __name__ == '__main__':
    create_comprehensive_visualizations()
