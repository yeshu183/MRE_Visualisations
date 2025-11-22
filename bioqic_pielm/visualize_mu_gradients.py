"""
Visualize μ(x) field and ∇μ computed via finite differences.

This script helps assess the quality of the gradient estimation used in
the gradient term test.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from data_loader import BIOQICDataLoader


def compute_mu_gradient_fd(x, mu_field):
    """Compute ∇μ using finite differences (same as in test_gradient_term_effect.py)."""
    N = x.shape[0]
    input_dim = x.shape[1]
    grad_mu = torch.zeros(N, input_dim, device=x.device, dtype=mu_field.dtype)

    # For each spatial dimension
    for d in range(input_dim):
        for i in range(N):
            xi = x[i:i+1, :]

            # Find nearest points in + and - directions along dimension d
            mask_plus = x[:, d] > xi[0, d]
            mask_minus = x[:, d] < xi[0, d]

            if mask_plus.sum() > 0 and mask_minus.sum() > 0:
                # Central difference
                dist_plus = torch.abs(x[mask_plus, d] - xi[0, d])
                dist_minus = torch.abs(x[mask_minus, d] - xi[0, d])

                idx_plus = torch.where(mask_plus)[0][torch.argmin(dist_plus)]
                idx_minus = torch.where(mask_minus)[0][torch.argmin(dist_minus)]

                dx = x[idx_plus, d] - x[idx_minus, d]
                if dx > 1e-8:
                    grad_mu[i, d] = (mu_field[idx_plus, 0] - mu_field[idx_minus, 0]) / dx
            elif mask_plus.sum() > 0:
                # Forward difference
                dist_plus = torch.abs(x[mask_plus, d] - xi[0, d])
                idx_plus = torch.where(mask_plus)[0][torch.argmin(dist_plus)]
                dx = x[idx_plus, d] - xi[0, d]
                if dx > 1e-8:
                    grad_mu[i, d] = (mu_field[idx_plus, 0] - mu_field[i, 0]) / dx
            elif mask_minus.sum() > 0:
                # Backward difference
                dist_minus = torch.abs(x[mask_minus, d] - xi[0, d])
                idx_minus = torch.where(mask_minus)[0][torch.argmin(dist_minus)]
                dx = xi[0, d] - x[idx_minus, d]
                if dx > 1e-8:
                    grad_mu[i, d] = (mu_field[i, 0] - mu_field[idx_minus, 0]) / dx

    return grad_mu


def main():
    print("="*80)
    print("VISUALIZING MU GRADIENTS")
    print("="*80)

    device = torch.device('cpu')

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'gradient_term_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    loader = BIOQICDataLoader(
        data_dir=str(data_dir),
        displacement_mode='z_component',
        subsample=5000,
        seed=42,
        adaptive_sampling=False
    )
    data = loader.load()

    coords = data['coords']
    mu_raw = data['mu_raw']

    x = torch.from_numpy(coords).float().to(device)
    mu = torch.from_numpy(mu_raw).float().to(device)

    print(f"  Points: {len(x)}")
    print(f"  μ range: [{mu.min():.0f}, {mu.max():.0f}] Pa")

    # Compute gradients
    print("\nComputing ∇μ via finite differences...")
    grad_mu = compute_mu_gradient_fd(x, mu)

    grad_mu_np = grad_mu.cpu().numpy()
    mu_np = mu.cpu().numpy().flatten()
    coords_np = x.cpu().numpy()

    # Compute gradient magnitude
    grad_mu_mag = np.linalg.norm(grad_mu_np, axis=1)

    print(f"\n∇μ Statistics:")
    print(f"  ∇μ_x: [{grad_mu_np[:, 0].min():.2f}, {grad_mu_np[:, 0].max():.2f}] Pa/m")
    print(f"  ∇μ_y: [{grad_mu_np[:, 1].min():.2f}, {grad_mu_np[:, 1].max():.2f}] Pa/m")
    print(f"  ∇μ_z: [{grad_mu_np[:, 2].min():.2f}, {grad_mu_np[:, 2].max():.2f}] Pa/m")
    print(f"  |∇μ|: [{grad_mu_mag.min():.2f}, {grad_mu_mag.max():.2f}] Pa/m")
    print(f"  Mean |∇μ|: {grad_mu_mag.mean():.2f} Pa/m")

    # Identify blob regions
    blob_threshold = 8000.0
    is_blob = mu_np > blob_threshold

    if is_blob.sum() > 0:
        print(f"\nBlob region gradient statistics:")
        print(f"  Blob points: {is_blob.sum()} ({100*is_blob.sum()/len(mu_np):.1f}%)")
        print(f"  Mean |∇μ| in blob: {grad_mu_mag[is_blob].mean():.2f} Pa/m")
        print(f"  Mean |∇μ| in background: {grad_mu_mag[~is_blob].mean():.2f} Pa/m")

    # Create visualizations
    print("\nCreating visualizations...")

    # Find middle z-slice for 2D visualization
    z_mid = (coords_np[:, 2].min() + coords_np[:, 2].max()) / 2
    z_tol = 0.005  # 5mm tolerance
    z_slice_mask = np.abs(coords_np[:, 2] - z_mid) < z_tol

    if z_slice_mask.sum() > 100:
        print(f"  Using z-slice at z={z_mid:.4f}m ({z_slice_mask.sum()} points)")

        x_slice = coords_np[z_slice_mask, 0]
        y_slice = coords_np[z_slice_mask, 1]
        mu_slice = mu_np[z_slice_mask]
        grad_x_slice = grad_mu_np[z_slice_mask, 0]
        grad_y_slice = grad_mu_np[z_slice_mask, 1]
        grad_mag_slice = grad_mu_mag[z_slice_mask]

        # Create 2x3 subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. μ field
        ax = axes[0, 0]
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=mu_slice/1000,
                           cmap='viridis', s=20, edgecolors='none')
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title('Stiffness Field μ(x,y)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('μ (kPa)', fontsize=10)

        # 2. ∇μ_x
        ax = axes[0, 1]
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=grad_x_slice,
                           cmap='RdBu_r', s=20, edgecolors='none',
                           vmin=-grad_mu_np[:, 0].std()*3, vmax=grad_mu_np[:, 0].std()*3)
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title('∂μ/∂x (Finite Difference)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('∂μ/∂x (Pa/m)', fontsize=10)

        # 3. ∇μ_y
        ax = axes[0, 2]
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=grad_y_slice,
                           cmap='RdBu_r', s=20, edgecolors='none',
                           vmin=-grad_mu_np[:, 1].std()*3, vmax=grad_mu_np[:, 1].std()*3)
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title('∂μ/∂y (Finite Difference)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('∂μ/∂y (Pa/m)', fontsize=10)

        # 4. Gradient magnitude |∇μ|
        ax = axes[1, 0]
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=grad_mag_slice,
                           cmap='hot', s=20, edgecolors='none')
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title('Gradient Magnitude |∇μ|', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('|∇μ| (Pa/m)', fontsize=10)

        # 5. Quiver plot of gradient direction
        ax = axes[1, 1]
        # Subsample for clearer quiver plot
        subsample = 4
        x_quiv = x_slice[::subsample]
        y_quiv = y_slice[::subsample]
        gx_quiv = grad_x_slice[::subsample]
        gy_quiv = grad_y_slice[::subsample]
        mu_quiv = mu_slice[::subsample]

        # Background: mu field
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=mu_slice/1000,
                           cmap='viridis', s=10, alpha=0.3, edgecolors='none')

        # Quiver: gradient direction
        quiv = ax.quiver(x_quiv*1000, y_quiv*1000, gx_quiv, gy_quiv,
                        angles='xy', scale_units='xy', scale=5000,
                        color='red', alpha=0.7, width=0.003)
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title('∇μ Direction (Red Arrows)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

        # 6. Histogram of gradient magnitude
        ax = axes[1, 2]
        ax.hist(grad_mu_mag, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(grad_mu_mag.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {grad_mu_mag.mean():.1f} Pa/m')
        ax.axvline(np.median(grad_mu_mag), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(grad_mu_mag):.1f} Pa/m')
        ax.set_xlabel('|∇μ| (Pa/m)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Gradient Magnitude', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'mu_gradient_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: mu_gradient_visualization.png")

    # Create 3D scatter plot showing gradient magnitude
    print("\nCreating 3D gradient magnitude plot...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample for performance
    subsample_3d = 5
    x_3d = coords_np[::subsample_3d, 0] * 1000
    y_3d = coords_np[::subsample_3d, 1] * 1000
    z_3d = coords_np[::subsample_3d, 2] * 1000
    grad_mag_3d = grad_mu_mag[::subsample_3d]

    scatter = ax.scatter(x_3d, y_3d, z_3d, c=grad_mag_3d,
                        cmap='hot', s=5, alpha=0.6, edgecolors='none')

    ax.set_xlabel('x (mm)', fontsize=11)
    ax.set_ylabel('y (mm)', fontsize=11)
    ax.set_zlabel('z (mm)', fontsize=11)
    ax.set_title('3D Gradient Magnitude |∇μ|', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('|∇μ| (Pa/m)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'mu_gradient_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mu_gradient_3d.png")

    # Analyze gradient estimation quality
    print("\n" + "="*80)
    print("GRADIENT ESTIMATION QUALITY ASSESSMENT")
    print("="*80)

    # Check for numerical artifacts
    zero_grad = (grad_mu_mag < 1e-6).sum()
    large_grad = (grad_mu_mag > 10000).sum()

    print(f"\nNumerical artifacts:")
    print(f"  Zero gradients: {zero_grad} ({100*zero_grad/len(grad_mu_mag):.2f}%)")
    print(f"  Very large gradients (>10k Pa/m): {large_grad} ({100*large_grad/len(grad_mu_mag):.2f}%)")

    # Compare gradient at blob boundaries
    blob_boundary_tol = 500  # Points within 500 Pa of blob threshold
    near_boundary = np.abs(mu_np - blob_threshold) < blob_boundary_tol

    if near_boundary.sum() > 0:
        print(f"\nGradient at blob boundaries (μ ≈ {blob_threshold:.0f} Pa):")
        print(f"  Boundary points: {near_boundary.sum()}")
        print(f"  Mean |∇μ|: {grad_mu_mag[near_boundary].mean():.1f} Pa/m")
        print(f"  Max |∇μ|: {grad_mu_mag[near_boundary].max():.1f} Pa/m")

    # Estimate expected gradient magnitude from μ range
    mu_range = mu.max() - mu.min()
    domain_size = np.linalg.norm([
        coords_np[:, 0].max() - coords_np[:, 0].min(),
        coords_np[:, 1].max() - coords_np[:, 1].min(),
        coords_np[:, 2].max() - coords_np[:, 2].min()
    ])
    expected_grad_order = mu_range.item() / domain_size

    print(f"\nExpected gradient order of magnitude:")
    print(f"  μ range: {mu_range:.0f} Pa")
    print(f"  Domain size: {domain_size:.3f} m")
    print(f"  Expected |∇μ|: ~{expected_grad_order:.0f} Pa/m")
    print(f"  Actual mean |∇μ|: {grad_mu_mag.mean():.1f} Pa/m")
    print(f"  Ratio: {grad_mu_mag.mean() / expected_grad_order:.2f}x")

    print("\n" + "="*80)
    print(f"Visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
