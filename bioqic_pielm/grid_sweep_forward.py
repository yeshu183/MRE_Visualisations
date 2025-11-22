"""
Comprehensive Grid Sweep for Forward Model
==========================================

Tests all combinations of:
- BC weights: [0, 1, 10, 100]
- Neuron counts: [2000, 5000, 10000]
- Sampling points: [1000, 10000, 50000]

Creates a giant visualization matrix showing results for each combination.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


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

        error = u_pred - u_meas
        mse = torch.mean(error ** 2).item()
        mae = torch.mean(torch.abs(error)).item()
        max_err = torch.max(torch.abs(error)).item()
        var_u = torch.var(u_meas).item()
        r2 = 1 - mse / var_u if var_u > 0 else 0

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_err,
        'r2': r2,
        'u_pred': u_pred.cpu().numpy(),
        'error': error.cpu().numpy()
    }


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE GRID SWEEP - BC WEIGHT x NEURONS x SAMPLING")
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

    # Grid sweep parameters
    bc_weights = [0, 1, 10, 100]
    neuron_counts = [2000, 5000, 10000]
    sampling_points = [1000, 10000, 50000]

    # ADAPTIVE SAMPLING PARAMETERS
    use_adaptive = True  # Set to False for uniform sampling comparison
    blob_ratio = 0.5
    boundary_ratio = 0.3

    # Physics parameters
    freq = 60  # Hz
    omega = 2 * np.pi * freq
    rho = 1000  # kg/m³
    rho_omega2 = rho * omega ** 2  # ≈ 1.42e8
    omega_basis = 170.0
    basis_type = 'sin'

    print(f"\nGrid sweep parameters:")
    print(f"  BC weights: {bc_weights}")
    print(f"  Neuron counts: {neuron_counts}")
    print(f"  Sampling points: {sampling_points}")
    print(f"  Adaptive sampling: {use_adaptive}")
    if use_adaptive:
        print(f"    Blob ratio: {blob_ratio}")
        print(f"    Boundary ratio: {boundary_ratio}")
        print(f"    Background ratio: {1.0 - blob_ratio - boundary_ratio}")
    print(f"  Total combinations: {len(bc_weights) * len(neuron_counts) * len(sampling_points)}")

    # Create giant visualization grid
    n_combinations = len(bc_weights) * len(neuron_counts) * len(sampling_points)
    n_cols = 3  # u_meas, u_pred, scatter
    fig, axes = plt.subplots(n_combinations, n_cols, figsize=(18, 6 * n_combinations))
    
    # Ensure axes is 2D
    if n_combinations == 1:
        axes = axes.reshape(1, -1)
    
    results_summary = []
    row_idx = 0

    # Grid sweep
    for n_points in sampling_points:
        for n_neurons in neuron_counts:
            for bc_weight in bc_weights:
                print(f"\n{'='*60}")
                print(f"Testing: n_points={n_points}, n_neurons={n_neurons}, bc_weight={bc_weight}")
                print(f"{'='*60}")

                # Load data with current sampling
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

                # Boundary conditions - FULL BOX
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

                # Create model
                model = ForwardMREModel(
                    n_wave_neurons=n_neurons,
                    input_dim=3,
                    omega_basis=omega_basis,
                    mu_min=3000.0,
                    mu_max=10000.0,
                    seed=42,
                    basis_type=basis_type
                ).to(device)

                # Test
                metrics = test_forward_given_mu(
                    model, x, u_meas, mu_true, bc_indices, u_bc_vals,
                    rho_omega2, bc_weight=bc_weight
                )

                print(f"  MSE: {metrics['mse']:.6e}, R²: {metrics['r2']:.4f}")

                # Store results
                results_summary.append({
                    'n_points': n_points,
                    'n_neurons': n_neurons,
                    'bc_weight': bc_weight,
                    'mse': metrics['mse'],
                    'r2': metrics['r2']
                })

                # Plotting
                u_pred = metrics['u_pred'].flatten()
                u_true = u_meas.cpu().numpy().flatten()
                coords_np = x.cpu().numpy()

                # Get middle Z slice
                z_mid = np.median(coords_np[:, 2])
                z_range = coords_np[:, 2].max() - coords_np[:, 2].min()
                z_tol = max(0.1, z_range * 0.3)
                z_mask = np.abs(coords_np[:, 2] - z_mid) < z_tol

                # Shared color scale
                vmin = min(u_true[z_mask].min(), u_pred[z_mask].min())
                vmax = max(u_true[z_mask].max(), u_pred[z_mask].max())

                # Column 1: u_meas
                ax1 = axes[row_idx, 0]
                sc1 = ax1.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], 
                                 c=u_true[z_mask], cmap='viridis', s=3, alpha=0.8, vmin=vmin, vmax=vmax)
                plt.colorbar(sc1, ax=ax1)
                ax1.set_xlabel('X (m)', fontsize=8)
                ax1.set_ylabel('Y (m)', fontsize=8)
                ax1.set_title(f'u_meas\nN={n_points}, M={n_neurons}, BC={bc_weight}', fontsize=9)
                ax1.set_aspect('equal')

                # Column 2: u_pred
                ax2 = axes[row_idx, 1]
                sc2 = ax2.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], 
                                 c=u_pred[z_mask], cmap='viridis', s=3, alpha=0.8, vmin=vmin, vmax=vmax)
                plt.colorbar(sc2, ax=ax2)
                ax2.set_xlabel('X (m)', fontsize=8)
                ax2.set_ylabel('Y (m)', fontsize=8)
                ax2.set_title(f'u_pred\nMSE={metrics["mse"]:.2e}, R²={metrics["r2"]:.3f}', fontsize=9)
                ax2.set_aspect('equal')

                # Column 3: Scatter plot
                ax3 = axes[row_idx, 2]
                ax3.scatter(u_true, u_pred, alpha=0.3, s=1)
                lims = [min(u_true.min(), u_pred.min()), max(u_true.max(), u_pred.max())]
                ax3.plot(lims, lims, 'r--', lw=2)
                ax3.set_xlabel('u_measured (m)', fontsize=8)
                ax3.set_ylabel('u_predicted (m)', fontsize=8)
                ax3.set_title(f'Displacement Comparison', fontsize=9)
                ax3.grid(True, alpha=0.3)

                row_idx += 1

    plt.suptitle(f'Comprehensive Grid Sweep: BC Weight x Neurons x Sampling\n'
                 f'rho_omega2={rho_omega2:.2e}, omega_basis={omega_basis}, {basis_type} basis',
                 fontsize=16, y=0.9995)
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_grid_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for res in results_summary:
        print(f"N={res['n_points']:5d}, M={res['n_neurons']:5d}, BC={res['bc_weight']:5.0f}: "
              f"MSE={res['mse']:.6e}, R²={res['r2']:.4f}")
    
    # Find best
    best_res = min(results_summary, key=lambda x: x['mse'])
    print(f"\nBest configuration:")
    print(f"  n_points={best_res['n_points']}, n_neurons={best_res['n_neurons']}, bc_weight={best_res['bc_weight']}")
    print(f"  MSE={best_res['mse']:.6e}, R²={best_res['r2']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
