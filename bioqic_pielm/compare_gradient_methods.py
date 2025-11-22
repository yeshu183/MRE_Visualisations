"""
Compare three gradient computation methods for ∇μ:

1. Grid-based: Compute on original voxel grid, interpolate to sample points
2. Dense sampling: Use 20,000 random points for better neighbor coverage
3. RBF interpolation: Fit smooth RBF to μ(x), take analytical derivatives

This will show whether better gradient estimation improves the gradient term's effectiveness.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import sobel
import pandas as pd

from data_loader import BIOQICDataLoader


def compute_gradient_finite_diff(x, mu_field):
    """Method 1: Finite differences on sparse random sampling (current broken method)."""
    N = x.shape[0]
    input_dim = x.shape[1]
    grad_mu = torch.zeros(N, input_dim, device=x.device, dtype=mu_field.dtype)

    for d in range(input_dim):
        for i in range(N):
            xi = x[i:i+1, :]
            mask_plus = x[:, d] > xi[0, d]
            mask_minus = x[:, d] < xi[0, d]

            if mask_plus.sum() > 0 and mask_minus.sum() > 0:
                dist_plus = torch.abs(x[mask_plus, d] - xi[0, d])
                dist_minus = torch.abs(x[mask_minus, d] - xi[0, d])
                idx_plus = torch.where(mask_plus)[0][torch.argmin(dist_plus)]
                idx_minus = torch.where(mask_minus)[0][torch.argmin(dist_minus)]
                dx = x[idx_plus, d] - x[idx_minus, d]
                if dx > 1e-8:
                    grad_mu[i, d] = (mu_field[idx_plus, 0] - mu_field[idx_minus, 0]) / dx
            elif mask_plus.sum() > 0:
                dist_plus = torch.abs(x[mask_plus, d] - xi[0, d])
                idx_plus = torch.where(mask_plus)[0][torch.argmin(dist_plus)]
                dx = x[idx_plus, d] - xi[0, d]
                if dx > 1e-8:
                    grad_mu[i, d] = (mu_field[idx_plus, 0] - mu_field[i, 0]) / dx
            elif mask_minus.sum() > 0:
                dist_minus = torch.abs(x[mask_minus, d] - xi[0, d])
                idx_minus = torch.where(mask_minus)[0][torch.argmin(dist_minus)]
                dx = xi[0, d] - x[idx_minus, d]
                if dx > 1e-8:
                    grad_mu[i, d] = (mu_field[i, 0] - mu_field[idx_minus, 0]) / dx

    return grad_mu


def compute_gradient_from_grid(coords_sample, mu_sample, data_dir):
    """Method 2: Compute gradients on original voxel grid, interpolate to sample points."""
    print("  Loading full voxel grid...")

    # Load original grid data
    coords_grid = np.load(data_dir / 'coordinates.npy')  # (N, 3)
    mu_grid_flat = np.load(data_dir / 'stiffness_ground_truth.npy')  # (N,)

    # Infer grid shape from coordinates
    # Assume regular grid structure
    x_unique = np.unique(coords_grid[:, 0])
    y_unique = np.unique(coords_grid[:, 1])
    z_unique = np.unique(coords_grid[:, 2])

    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    grid_shape = (nx, ny, nz)
    print(f"  Grid shape: {grid_shape}")

    # Reshape mu to grid
    mu_grid = mu_grid_flat.reshape(grid_shape, order='F')  # Fortran order (x varies fastest)

    # Compute gradients using Sobel filter on the grid
    print("  Computing gradients on grid using Sobel filter...")
    grad_mu_x = sobel(mu_grid, axis=0, mode='constant')
    grad_mu_y = sobel(mu_grid, axis=1, mode='constant')
    grad_mu_z = sobel(mu_grid, axis=2, mode='constant')

    # Create coordinate arrays for the grid
    x_grid = x_unique
    y_grid = y_unique
    z_grid = z_unique

    # Interpolate gradients to sample points
    print("  Interpolating gradients to sample points...")
    from scipy.interpolate import RegularGridInterpolator

    interp_x = RegularGridInterpolator((x_grid, y_grid, z_grid), grad_mu_x,
                                        method='linear', bounds_error=False, fill_value=0)
    interp_y = RegularGridInterpolator((x_grid, y_grid, z_grid), grad_mu_y,
                                        method='linear', bounds_error=False, fill_value=0)
    interp_z = RegularGridInterpolator((x_grid, y_grid, z_grid), grad_mu_z,
                                        method='linear', bounds_error=False, fill_value=0)

    grad_x_sample = interp_x(coords_sample)
    grad_y_sample = interp_y(coords_sample)
    grad_z_sample = interp_z(coords_sample)

    # Scale gradients (Sobel gives derivatives in voxel units, convert to Pa/m)
    dx_voxel = x_grid[1] - x_grid[0]
    dy_voxel = y_grid[1] - y_grid[0]
    dz_voxel = z_grid[1] - z_grid[0]

    grad_x_sample /= dx_voxel
    grad_y_sample /= dy_voxel
    grad_z_sample /= dz_voxel

    grad_mu = np.stack([grad_x_sample, grad_y_sample, grad_z_sample], axis=1)

    return torch.from_numpy(grad_mu).float()


def compute_gradient_rbf(coords_sample, mu_sample, rbf_neighbors=1000):
    """Method 3: RBF interpolation with analytical derivatives."""
    print(f"  Fitting RBF with {min(rbf_neighbors, len(coords_sample))} neighbors...")

    coords_np = coords_sample.cpu().numpy() if torch.is_tensor(coords_sample) else coords_sample
    mu_np = mu_sample.cpu().numpy().flatten() if torch.is_tensor(mu_sample) else mu_sample.flatten()

    # Subsample for RBF fitting if needed (RBF is expensive)
    if len(coords_np) > rbf_neighbors:
        idx = np.random.choice(len(coords_np), rbf_neighbors, replace=False)
        coords_fit = coords_np[idx]
        mu_fit = mu_np[idx]
    else:
        coords_fit = coords_np
        mu_fit = mu_np

    # Fit RBF (use thin_plate_spline for smoothness)
    print("  Building RBF interpolator...")
    rbf = RBFInterpolator(coords_fit, mu_fit, kernel='thin_plate_spline', epsilon=1.0)

    # Compute gradients using finite differences on RBF
    print("  Computing analytical gradients from RBF...")
    h = 1e-4  # Small perturbation (0.1mm)
    grad_mu = np.zeros((len(coords_np), 3))

    for d in range(3):
        coords_plus = coords_np.copy()
        coords_minus = coords_np.copy()
        coords_plus[:, d] += h
        coords_minus[:, d] -= h

        mu_plus = rbf(coords_plus)
        mu_minus = rbf(coords_minus)

        grad_mu[:, d] = (mu_plus - mu_minus) / (2 * h)

    return torch.from_numpy(grad_mu).float()


def analyze_gradient_quality(grad_mu, mu_field, method_name, coords=None):
    """Compute statistics for gradient quality assessment."""
    grad_mu_np = grad_mu.cpu().numpy() if torch.is_tensor(grad_mu) else grad_mu
    mu_np = mu_field.cpu().numpy().flatten() if torch.is_tensor(mu_field) else mu_field.flatten()

    grad_mag = np.linalg.norm(grad_mu_np, axis=1)

    # Identify blob regions
    blob_threshold = 8000.0
    is_blob = mu_np > blob_threshold

    stats = {
        'method': method_name,
        'mean_mag': grad_mag.mean(),
        'median_mag': np.median(grad_mag),
        'max_mag': grad_mag.max(),
        'min_mag': grad_mag.min(),
        'std_mag': grad_mag.std(),
        'pct_zero': (grad_mag < 1e-6).sum() / len(grad_mag) * 100,
        'pct_large': (grad_mag > 10000).sum() / len(grad_mag) * 100,
    }

    if is_blob.sum() > 0:
        stats['blob_mean_mag'] = grad_mag[is_blob].mean()
        stats['background_mean_mag'] = grad_mag[~is_blob].mean()
        stats['blob_ratio'] = stats['blob_mean_mag'] / stats['background_mean_mag'] if stats['background_mean_mag'] > 0 else np.nan

    return stats, grad_mag


def visualize_gradient_comparison(coords, mu_field, grad_dict, output_dir):
    """Create comprehensive visualizations comparing gradient methods."""

    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
    mu_np = mu_field.cpu().numpy().flatten() if torch.is_tensor(mu_field) else mu_field.flatten()

    # Find middle z-slice
    z_mid = (coords_np[:, 2].min() + coords_np[:, 2].max()) / 2
    z_tol = 0.005
    z_slice_mask = np.abs(coords_np[:, 2] - z_mid) < z_tol

    if z_slice_mask.sum() < 50:
        print("  Not enough points in z-slice for visualization")
        return

    x_slice = coords_np[z_slice_mask, 0]
    y_slice = coords_np[z_slice_mask, 1]
    mu_slice = mu_np[z_slice_mask]

    # Create 2x4 subplot (1 for mu field, 3 for gradient methods)
    n_methods = len(grad_dict)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(6 * (n_methods + 1), 12))

    # First column: μ field
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

    # Histogram placeholder (will use for comparison later)
    axes[1, 0].axis('off')

    # Each method gets two panels: magnitude map and quiver plot
    for idx, (method_name, grad_data) in enumerate(grad_dict.items(), start=1):
        grad_np = grad_data['grad'].cpu().numpy() if torch.is_tensor(grad_data['grad']) else grad_data['grad']
        grad_mag = grad_data['mag']

        grad_mag_slice = grad_mag[z_slice_mask]
        grad_x_slice = grad_np[z_slice_mask, 0]
        grad_y_slice = grad_np[z_slice_mask, 1]

        # Top: Gradient magnitude
        ax = axes[0, idx]
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=grad_mag_slice,
                           cmap='hot', s=20, edgecolors='none', vmin=0, vmax=grad_mag.max())
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title(f'{method_name}\n|∇μ|', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('|∇μ| (Pa/m)', fontsize=10)

        # Bottom: Quiver plot
        ax = axes[1, idx]
        subsample = 4
        x_quiv = x_slice[::subsample]
        y_quiv = y_slice[::subsample]
        gx_quiv = grad_x_slice[::subsample]
        gy_quiv = grad_y_slice[::subsample]

        # Background: mu field
        scatter = ax.scatter(x_slice*1000, y_slice*1000, c=mu_slice/1000,
                           cmap='viridis', s=10, alpha=0.3, edgecolors='none')

        # Quiver: gradient direction
        scale = np.percentile(np.abs(grad_np[:, :2]), 95) * 5 if np.percentile(np.abs(grad_np[:, :2]), 95) > 0 else 5000
        quiv = ax.quiver(x_quiv*1000, y_quiv*1000, gx_quiv, gy_quiv,
                        angles='xy', scale_units='xy', scale=scale,
                        color='red', alpha=0.7, width=0.003)
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_title(f'{method_name}\n∇μ Direction', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_methods_comparison.png")

    # Create histogram comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'green', 'red', 'purple']

    # Histogram of gradient magnitudes (linear scale)
    ax = axes[0, 0]
    for idx, (method_name, grad_data) in enumerate(grad_dict.items()):
        grad_mag = grad_data['mag']
        ax.hist(grad_mag, bins=50, alpha=0.5, label=method_name, color=colors[idx % len(colors)])
    ax.set_xlabel('|∇μ| (Pa/m)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Gradient Magnitude Distribution (Linear)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Histogram (log scale)
    ax = axes[0, 1]
    for idx, (method_name, grad_data) in enumerate(grad_dict.items()):
        grad_mag = grad_data['mag']
        ax.hist(grad_mag[grad_mag > 0], bins=50, alpha=0.5, label=method_name, color=colors[idx % len(colors)])
    ax.set_xlabel('|∇μ| (Pa/m)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Gradient Magnitude Distribution (Log)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot comparison
    ax = axes[1, 0]
    data_for_box = [grad_data['mag'] for grad_data in grad_dict.values()]
    labels_for_box = list(grad_dict.keys())
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('|∇μ| (Pa/m)', fontsize=11)
    ax.set_title('Gradient Magnitude Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    # Statistics comparison
    ax = axes[1, 1]
    stats_names = ['Mean', 'Median', 'Std', '% Zero\n(×1000)', '% Large\n(×100)']
    x_pos = np.arange(len(stats_names))
    width = 0.8 / len(grad_dict)

    for idx, (method_name, grad_data) in enumerate(grad_dict.items()):
        stats = grad_data['stats']
        values = [
            stats['mean_mag'],
            stats['median_mag'],
            stats['std_mag'],
            stats['pct_zero'] * 1000,  # Scale for visibility
            stats['pct_large'] * 100
        ]
        ax.bar(x_pos + idx * width, values, width, label=method_name,
               color=colors[idx % len(colors)], alpha=0.7)

    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Gradient Statistics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width * (len(grad_dict) - 1) / 2)
    ax.set_xticklabels(stats_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_statistics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_statistics_comparison.png")


def main():
    print("="*80)
    print("GRADIENT COMPUTATION METHOD COMPARISON")
    print("="*80)

    device = torch.device('cpu')

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'gradient_method_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (5,000 points as in the original test)
    print("\nLoading data (5,000 sample points)...")
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

    # Dictionary to store results
    gradient_results = {}

    # Method 1: Finite differences (current broken method)
    print("\n" + "="*80)
    print("METHOD 1: Finite Differences on Sparse Random Sampling (Current)")
    print("="*80)
    try:
        grad_fd = compute_gradient_finite_diff(x, mu)
        stats_fd, mag_fd = analyze_gradient_quality(grad_fd, mu, "Finite Diff (Sparse)")
        gradient_results['Finite Diff\n(Sparse)'] = {
            'grad': grad_fd,
            'mag': mag_fd,
            'stats': stats_fd
        }
        print(f"  Mean |∇μ|: {stats_fd['mean_mag']:.1f} Pa/m")
        print(f"  % Zero: {stats_fd['pct_zero']:.1f}%")
        print(f"  % Large (>10k): {stats_fd['pct_large']:.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Method 2: Grid-based gradients
    print("\n" + "="*80)
    print("METHOD 2: Grid-Based Gradients (Sobel on Original Voxel Grid)")
    print("="*80)
    try:
        grad_grid = compute_gradient_from_grid(coords, mu_raw, data_dir)
        grad_grid = grad_grid.to(device)
        stats_grid, mag_grid = analyze_gradient_quality(grad_grid, mu, "Grid-Based (Sobel)")
        gradient_results['Grid-Based\n(Sobel)'] = {
            'grad': grad_grid,
            'mag': mag_grid,
            'stats': stats_grid
        }
        print(f"  Mean |∇μ|: {stats_grid['mean_mag']:.1f} Pa/m")
        print(f"  % Zero: {stats_grid['pct_zero']:.1f}%")
        print(f"  % Large (>10k): {stats_grid['pct_large']:.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Method 3: RBF interpolation
    print("\n" + "="*80)
    print("METHOD 3: RBF Interpolation with Analytical Derivatives")
    print("="*80)
    try:
        grad_rbf = compute_gradient_rbf(coords, mu_raw, rbf_neighbors=1000)
        grad_rbf = grad_rbf.to(device)
        stats_rbf, mag_rbf = analyze_gradient_quality(grad_rbf, mu, "RBF Interpolation")
        gradient_results['RBF\nInterpolation'] = {
            'grad': grad_rbf,
            'mag': mag_rbf,
            'stats': stats_rbf
        }
        print(f"  Mean |∇μ|: {stats_rbf['mean_mag']:.1f} Pa/m")
        print(f"  % Zero: {stats_rbf['pct_zero']:.1f}%")
        print(f"  % Large (>10k): {stats_rbf['pct_large']:.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Save statistics
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    stats_list = [grad_data['stats'] for grad_data in gradient_results.values()]
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_dir / 'gradient_method_statistics.csv', index=False)
    print(f"  Saved: gradient_method_statistics.csv")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_gradient_comparison(x, mu, gradient_results, output_dir)

    # Compare to expected gradient
    print("\n" + "="*80)
    print("COMPARISON TO EXPECTED GRADIENT MAGNITUDE")
    print("="*80)

    mu_range = mu.max() - mu.min()
    domain_size = np.linalg.norm([
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        coords[:, 2].max() - coords[:, 2].min()
    ])
    expected_grad = mu_range.item() / domain_size

    print(f"\nExpected gradient order of magnitude: ~{expected_grad:.0f} Pa/m")
    print(f"(Based on μ range / domain size)")
    print()

    for method_name, grad_data in gradient_results.items():
        stats = grad_data['stats']
        ratio = stats['mean_mag'] / expected_grad
        print(f"{method_name.replace(chr(10), ' ')}:")
        print(f"  Mean |∇μ|: {stats['mean_mag']:.1f} Pa/m")
        print(f"  Ratio to expected: {ratio:.2f}x")
        print(f"  Assessment: {'✓ Reasonable' if 0.5 < ratio < 5 else '✗ Unrealistic'}")
        print()

    # Final recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)

    best_method = None
    best_score = -np.inf

    for method_name, grad_data in gradient_results.items():
        stats = grad_data['stats']
        # Score based on: low % zero, reasonable magnitude, low % large spikes
        score = (100 - stats['pct_zero']) - stats['pct_large'] * 2
        ratio = stats['mean_mag'] / expected_grad
        if ratio > 5 or ratio < 0.1:
            score -= 50  # Penalty for unrealistic magnitude

        print(f"{method_name.replace(chr(10), ' ')}: score = {score:.1f}")

        if score > best_score:
            best_score = score
            best_method = method_name

    print(f"\nBest method: {best_method.replace(chr(10), ' ')}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
