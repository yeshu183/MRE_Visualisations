"""
Forward Model Testing with Sin Basis
======================================

Tests the PIELM forward model with optimized parameters:
1. Sin (Fourier) basis functions
2. Physical rho_omega2 = 1.42e8 (fixed)
3. Optimal parameters: 10000 points, 1000 neurons, bc_weight=10
4. Visualization of u_meas vs u_pred comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


def compute_region_metrics(u_pred, u_meas, mu_true, blob_threshold=8000.0):
    """Compute error metrics separately for blob and background regions."""
    mu_np = mu_true.cpu().numpy().flatten()
    u_pred_np = u_pred.cpu().numpy().flatten()
    u_meas_np = u_meas.cpu().numpy().flatten()

    # Classify regions
    is_blob = (mu_np > blob_threshold).astype(bool)
    is_background = ~is_blob

    # Compute metrics for each region
    def compute_metrics(pred, meas):
        error = pred - meas
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        max_err = np.max(np.abs(error))
        var_u = np.var(meas)
        r2 = 1 - mse / var_u if var_u > 0 else 0
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_err,
            'r2': r2,
            'n_points': len(pred)
        }

    metrics = {
        'overall': compute_metrics(u_pred_np, u_meas_np),
        'blob': compute_metrics(u_pred_np[is_blob], u_meas_np[is_blob]) if is_blob.sum() > 0 else None,
        'background': compute_metrics(u_pred_np[is_background], u_meas_np[is_background]) if is_background.sum() > 0 else None
    }

    return metrics


def test_forward_given_mu(
    model: ForwardMREModel,
    x: torch.Tensor,
    u_meas: torch.Tensor,
    mu_true: torch.Tensor,
    bc_indices: torch.Tensor,
    u_bc_vals: torch.Tensor,
    rho_omega2: float,
    bc_weight: float
):
    """Test forward solve with ground truth stiffness."""
    with torch.no_grad():
        u_pred, _ = model.solve_given_mu(
            x, mu_true, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )

        # Compute region-specific metrics
        region_metrics = compute_region_metrics(u_pred, u_meas, mu_true)

        # Add legacy fields for backward compatibility
        error = u_pred - u_meas
        result = {
            'mse': region_metrics['overall']['mse'],
            'mae': region_metrics['overall']['mae'],
            'max_error': region_metrics['overall']['max_error'],
            'r2': region_metrics['overall']['r2'],
            'u_pred': u_pred.cpu().numpy(),
            'error': error.cpu().numpy(),
            'region_metrics': region_metrics
        }

    return result


def plot_u_comparison(x, u_meas, mu_true, bc_indices, u_bc_vals,
                      rho_omega2, device, output_dir, bc_weight, omega_basis=10, n_neurons=100, basis_type='sin'):
    """Simple plot: u_meas(x,y) vs u_pred(x,y)."""

    print("\n" + "="*60)
    print("VISUALIZATION: u_meas vs u_pred")
    print("="*60)

    model = ForwardMREModel(
        n_wave_neurons=n_neurons,
        input_dim=3,
        omega_basis=omega_basis,
        mu_min=3000.0,
        mu_max=10000.0,
        seed=42,
        basis_type=basis_type
    ).to(device)

    metrics = test_forward_given_mu(
        model, x, u_meas, mu_true, bc_indices, u_bc_vals,
        rho_omega2, bc_weight=bc_weight
    )

    u_pred = metrics['u_pred'].flatten()
    u_true = u_meas.cpu().numpy().flatten()
    mu_true_np = mu_true.cpu().numpy().flatten()
    coords = x.cpu().numpy()

    # Print region-specific metrics
    region_metrics = metrics['region_metrics']

    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"  MSE: {region_metrics['overall']['mse']:.6e}")
    print(f"  MAE: {region_metrics['overall']['mae']:.6e}")
    print(f"  R²: {region_metrics['overall']['r2']:.4f}")
    print(f"  Max Error: {region_metrics['overall']['max_error']:.6e}")
    print(f"  Points: {region_metrics['overall']['n_points']}")

    if region_metrics['blob'] is not None:
        print(f"\n{'='*60}")
        print("BLOB REGION METRICS (μ > 8 kPa)")
        print(f"{'='*60}")
        print(f"  MSE: {region_metrics['blob']['mse']:.6e}")
        print(f"  MAE: {region_metrics['blob']['mae']:.6e}")
        print(f"  R²: {region_metrics['blob']['r2']:.4f}")
        print(f"  Max Error: {region_metrics['blob']['max_error']:.6e}")
        print(f"  Points: {region_metrics['blob']['n_points']} ({100*region_metrics['blob']['n_points']/region_metrics['overall']['n_points']:.1f}%)")

    if region_metrics['background'] is not None:
        print(f"\n{'='*60}")
        print("BACKGROUND REGION METRICS (μ ≤ 8 kPa)")
        print(f"{'='*60}")
        print(f"  MSE: {region_metrics['background']['mse']:.6e}")
        print(f"  MAE: {region_metrics['background']['mae']:.6e}")
        print(f"  R²: {region_metrics['background']['r2']:.4f}")
        print(f"  Max Error: {region_metrics['background']['max_error']:.6e}")
        print(f"  Points: {region_metrics['background']['n_points']} ({100*region_metrics['background']['n_points']/region_metrics['overall']['n_points']:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  u_meas range: [{u_true.min():.6e}, {u_true.max():.6e}]")
    print(f"  u_pred range: [{u_pred.min():.6e}, {u_pred.max():.6e}]")

    # Create figure with 4x2 subplots (top slice + middle slice comparisons)
    fig, axes = plt.subplots(4, 2, figsize=(14, 24))

    # Get top and middle Z slices
    z_max_val = coords[:, 2].max()
    z_mid = np.median(coords[:, 2])
    z_range = coords[:, 2].max() - coords[:, 2].min()
    z_tol = max(0.01, z_range * 0.1)  # Tighter tolerance for clearer slices
    
    z_mask_top = np.abs(coords[:, 2] - z_max_val) < z_tol
    z_mask_mid = np.abs(coords[:, 2] - z_mid) < z_tol

    # === ROW 1: TOP SLICE (Actuator face) ===
    # Fixed colorbar scale for all displacement plots
    u_vmin = -0.025
    u_vmax = 0.025

    ax1 = axes[0, 0]
    # Plot displacement with stiffness contours overlaid
    sc1 = ax1.scatter(coords[z_mask_top, 0], coords[z_mask_top, 1], c=u_true[z_mask_top],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    # Overlay stiffness contours
    ax1.tricontour(coords[z_mask_top, 0], coords[z_mask_top, 1], mu_true_np[z_mask_top],
                   levels=[5000, 7000, 9000], colors='red', linewidths=1.5, alpha=0.6)
    plt.colorbar(sc1, ax=ax1, label='u (m)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'u_measured TOP (z≈{z_max_val:.4f}m) + μ contours')
    ax1.set_aspect('equal')

    ax2 = axes[0, 1]
    sc2 = ax2.scatter(coords[z_mask_top, 0], coords[z_mask_top, 1], c=u_pred[z_mask_top],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    plt.colorbar(sc2, ax=ax2, label='u (m)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'u_predicted TOP (z≈{z_max_val:.4f}m - Actuator)')
    ax2.set_aspect('equal')

    # === ROW 2: MIDDLE SLICE (Interior propagation) ===
    ax3 = axes[1, 0]
    sc3 = ax3.scatter(coords[z_mask_mid, 0], coords[z_mask_mid, 1], c=u_true[z_mask_mid],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    # Overlay stiffness contours
    ax3.tricontour(coords[z_mask_mid, 0], coords[z_mask_mid, 1], mu_true_np[z_mask_mid],
                   levels=[5000, 7000, 9000], colors='red', linewidths=1.5, alpha=0.6)
    plt.colorbar(sc3, ax=ax3, label='u (m)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'u_measured MIDDLE (z≈{z_mid:.4f}m) + μ contours')
    ax3.set_aspect('equal')

    ax4 = axes[1, 1]
    sc4 = ax4.scatter(coords[z_mask_mid, 0], coords[z_mask_mid, 1], c=u_pred[z_mask_mid],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    plt.colorbar(sc4, ax=ax4, label='u (m)')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title(f'u_predicted MIDDLE (z≈{z_mid:.4f}m)')
    ax4.set_aspect('equal')

    # === ROW 3: Scatter plot and Error ===
    ax5 = axes[2, 0]
    ax5.scatter(u_true, u_pred, alpha=0.3, s=1)
    lims = [min(u_true.min(), u_pred.min()), max(u_true.max(), u_pred.max())]
    ax5.plot(lims, lims, 'r--', lw=2, label='Perfect fit')
    ax5.set_xlabel('u_measured (m)')
    ax5.set_ylabel('u_predicted (m)')
    ax5.set_title(f'Displacement Comparison (R²={metrics["r2"]:.4f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Error distribution in middle slice
    ax6 = axes[2, 1]
    error = np.abs(u_pred - u_true)
    sc6 = ax6.scatter(coords[z_mask_mid, 0], coords[z_mask_mid, 1], c=error[z_mask_mid],
                      cmap='hot', s=5, alpha=0.8)
    # Overlay stiffness contours
    ax6.tricontour(coords[z_mask_mid, 0], coords[z_mask_mid, 1], mu_true_np[z_mask_mid],
                   levels=[5000, 7000, 9000], colors='cyan', linewidths=1.5, alpha=0.7)
    plt.colorbar(sc6, ax=ax6, label='|error| (m)')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title(f'Absolute Error MIDDLE (z≈{z_mid:.4f}m) + μ contours')
    ax6.set_aspect('equal')

    # === ROW 4: Ground truth stiffness and histogram ===
    ax7 = axes[3, 0]
    sc7 = ax7.scatter(coords[z_mask_mid, 0], coords[z_mask_mid, 1], c=mu_true_np[z_mask_mid],
                      cmap='coolwarm', s=8, alpha=0.8)
    plt.colorbar(sc7, ax=ax7, label='μ (Pa)')
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title(f'Ground Truth Stiffness (z≈{z_mid:.4f}m)')
    ax7.set_aspect('equal')

    ax8 = axes[3, 1]
    ax8.hist(mu_true_np, bins=50, alpha=0.7, edgecolor='black')
    ax8.axvline(mu_true_np.min(), color='r', linestyle='--', linewidth=2, label=f'Min: {mu_true_np.min():.0f} Pa')
    ax8.axvline(mu_true_np.max(), color='b', linestyle='--', linewidth=2, label=f'Max: {mu_true_np.max():.0f} Pa')
    ax8.set_xlabel('Stiffness μ (Pa)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Stiffness Distribution')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle(f'Forward Model: u_meas vs u_pred (Box BC)\n'
                 f'bc_weight={bc_weight:.0e}, rho_omega2={rho_omega2:.2e}, sin basis, R²={metrics["r2"]:.4f}',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'u_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return metrics


def main():
    print("\n" + "="*70)
    print("FORWARD MODEL TESTING - OPTIMIZED PARAMETERS")
    print("="*70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Get absolute paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'

    output_dir = script_dir / 'outputs' / 'forward_model_tests'
    output_dir.mkdir(parents=True, exist_ok=True)

    # OPTIMIZED PARAMETERS (from grid sweep results)
    n_points = 10000
    n_neurons = 1000
    bc_weight = 10
    omega_basis = 170.0
    basis_type = 'sin'

    # ADAPTIVE SAMPLING PARAMETERS
    use_adaptive = False  # Set to False for uniform sampling comparison
    blob_ratio = 0.2     # 20% blob samples (reduced from 50%)
    boundary_ratio = 0.1 # 10% boundary samples (reduced from 30%)
    # background_ratio = 0.7 (70% - implicit)

    print(f"\nOptimized parameters:")
    print(f"  Sampling points: {n_points}")
    print(f"  Number of neurons: {n_neurons}")
    print(f"  BC weight: {bc_weight}")
    print(f"  Omega basis: {omega_basis}")
    print(f"  Basis type: {basis_type}")
    print(f"  Adaptive sampling: {use_adaptive}")
    if use_adaptive:
        print(f"    Blob ratio: {blob_ratio}")
        print(f"    Boundary ratio: {boundary_ratio}")
        print(f"    Background ratio: {1.0 - blob_ratio - boundary_ratio}")

    # Load data
    loader = BIOQICDataLoader(
        data_dir=str(data_dir),
        displacement_mode='z_component',
        subsample=n_points,
        seed=42,
        adaptive_sampling=use_adaptive,
        blob_sample_ratio=blob_ratio,
        boundary_sample_ratio=boundary_ratio
    )
    data = loader.load()

    # Use RAW SI data
    coords = data['coords']
    u_raw = data['u_raw']
    mu_raw = data['mu_raw']

    x = torch.from_numpy(coords).float().to(device)
    u_meas = torch.from_numpy(u_raw).float().to(device)
    mu_true = torch.from_numpy(mu_raw).float().to(device)
    #mu_true = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)  # Homogeneous

    # Boundary conditions - FULL BOX (all 6 faces)
    print("\nSelecting FULL BOX boundary indices...")
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    tol = 1e-4

    mask_x = (np.abs(coords[:, 0] - x_min) < tol) | (np.abs(coords[:, 0] - x_max) < tol)
    mask_y = (np.abs(coords[:, 1] - y_min) < tol) | (np.abs(coords[:, 1] - y_max) < tol)
    mask_z = (np.abs(coords[:, 2] - z_min) < tol) | (np.abs(coords[:, 2] - z_max) < tol)
    bc_mask = mask_x | mask_y | mask_z
    bc_indices_np = np.where(bc_mask)[0]
    bc_indices = torch.from_numpy(bc_indices_np).long().to(device)
    u_bc_vals = u_meas[bc_indices]

    print(f"  BC strategy: Box (6 faces)")
    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")

    # Physics parameters
    freq = 60  # Hz
    omega = 2 * np.pi * freq
    rho = 1000  # kg/m³
    rho_omega2 = rho * omega ** 2  # ≈ 1.42e8

    print(f"\nData summary:")
    print(f"  Coords range: x=[{coords[:,0].min():.4f}, {coords[:,0].max():.4f}] m")
    print(f"               y=[{coords[:,1].min():.4f}, {coords[:,1].max():.4f}] m")
    print(f"               z=[{coords[:,2].min():.4f}, {coords[:,2].max():.4f}] m")
    print(f"  u_meas range: [{u_meas.min():.6e}, {u_meas.max():.6e}] m")
    print(f"  mu_true range: [{mu_true.min():.0f}, {mu_true.max():.0f}] Pa")
    print(f"  u_bc_vals range: [{u_bc_vals.min():.6e}, {u_bc_vals.max():.6e}] m")
    print(f"  Physical rho_omega2: {rho_omega2:.2e}")

    # Test with optimized parameters
    metrics = plot_u_comparison(
        x, u_meas, mu_true, bc_indices, u_bc_vals,
        rho_omega2, device, output_dir, bc_weight=bc_weight,
        omega_basis=omega_basis, n_neurons=n_neurons, basis_type=basis_type
    )

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"  Sampling points: {n_points}")
    print(f"  Number of neurons: {n_neurons}")
    print(f"  BC weight: {bc_weight}")
    print(f"  MSE: {metrics['mse']:.6e}")
    print(f"  MAE: {metrics['mae']:.6e}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
