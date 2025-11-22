"""
Simplified Grid Search for Adaptive Sampling Comparison
========================================================

Tests different adaptive sampling strategies to see which best
discriminates between constant and heterogeneous stiffness fields.

Fixed parameters:
- BC weight: 10
- Sampling points: 5000
- Neurons: [100, 500, 1000]

Variable: Sampling strategies (uniform vs 6 adaptive configs)
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


def compute_region_metrics(u_pred, u_meas, mu_field, blob_threshold=8000.0):
    """Compute error metrics separately for blob and background regions."""
    mu_np = mu_field.cpu().numpy().flatten()
    u_pred_np = u_pred.cpu().numpy().flatten()
    u_meas_np = u_meas.cpu().numpy().flatten()

    # Classify regions
    is_blob = (mu_np > blob_threshold).astype(bool)
    is_background = ~is_blob

    # Compute metrics for each region
    def compute_metrics(pred, meas):
        if len(pred) == 0:
            return None
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


def test_forward_config(
    model, x, u_meas, mu, bc_indices, u_bc_vals, rho_omega2, bc_weight
):
    """Test forward solve and return all loss metrics."""
    with torch.no_grad():
        u_pred, _ = model.solve_given_mu(
            x, mu, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )

        # Basic metrics
        error = u_pred - u_meas
        mse = torch.mean(error ** 2).item()
        var_u = torch.var(u_meas).item()
        r2 = 1 - mse / var_u if var_u > 0 else 0

        # Region-specific metrics
        region_metrics = compute_region_metrics(u_pred, u_meas, mu)

        metrics = {
            'mse': mse,
            'r2': r2,
            'blob_r2': region_metrics['blob']['r2'] if region_metrics['blob'] else None,
            'blob_mse': region_metrics['blob']['mse'] if region_metrics['blob'] else None,
            'background_r2': region_metrics['background']['r2'] if region_metrics['background'] else None,
            'background_mse': region_metrics['background']['mse'] if region_metrics['background'] else None,
        }

    return metrics


def main():
    print("\n" + "="*80)
    print("ADAPTIVE SAMPLING GRID SEARCH")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'sampling_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed parameters
    bc_weight = 10
    n_points = 5000
    neurons_list = [100, 500, 1000, 10000]  # Added 10000 neurons
    omega_basis = 170.0

    # Sampling configurations: (use_adaptive, blob_ratio, boundary_ratio, allow_replacement, name)
    sampling_configs = [
        (False, 0.0, 0.0, False, 'uniform'),
        (True, 0.05, 0.25, True, 'adaptive_5_25_70_repl'),
        (True, 0.10, 0.20, True, 'adaptive_10_20_70_repl'),
        (True, 0.20, 0.10, True, 'adaptive_20_10_70_repl'),
        (True, 0.05, 0.25, False, 'adaptive_5_25_70_noRepl'),
        (True, 0.10, 0.20, False, 'adaptive_10_20_70_noRepl'),
        (True, 0.20, 0.10, False, 'adaptive_20_10_70_noRepl'),
    ]

    print(f"\nFixed Parameters:")
    print(f"  BC weight: {bc_weight}")
    print(f"  Sampling points: {n_points}")
    print(f"  Neurons: {neurons_list}")

    # Physics parameters
    freq = 60
    omega = 2 * np.pi * freq
    rho = 1000
    rho_omega2 = rho * omega ** 2

    # Run grid search
    results = []
    total_configs = len(sampling_configs) * len(neurons_list) * 2
    config_count = 0

    print("\n" + "="*80)
    print(f"RUNNING {total_configs} CONFIGURATIONS")
    print("="*80)

    for use_adaptive, blob_ratio, boundary_ratio, allow_replacement, sampling_name in sampling_configs:
        print(f"\n{'='*80}")
        print(f"SAMPLING: {sampling_name}")
        print(f"{'='*80}")

        # Load data with current sampling configuration
        loader = BIOQICDataLoader(
            data_dir=str(data_dir),
            displacement_mode='z_component',
            subsample=n_points,
            seed=42,
            adaptive_sampling=use_adaptive,
            blob_sample_ratio=blob_ratio,
            boundary_sample_ratio=boundary_ratio,
            allow_replacement=allow_replacement
        )
        data = loader.load()

        coords = data['coords']
        u_raw = data['u_raw']
        mu_raw = data['mu_raw']

        x = torch.from_numpy(coords).float().to(device)
        u_meas = torch.from_numpy(u_raw).float().to(device)
        mu_heterogeneous = torch.from_numpy(mu_raw).float().to(device)
        mu_constant = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)

        # Boundary conditions
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        tol = 1e-4

        mask_x = (np.abs(coords[:, 0] - x_min) < tol) | (np.abs(coords[:, 0] - x_max) < tol)
        mask_y = (np.abs(coords[:, 1] - y_min) < tol) | (np.abs(coords[:, 1] - y_max) < tol)
        mask_z = (np.abs(coords[:, 2] - z_min) < tol) | (np.abs(coords[:, 2] - z_max) < tol)
        bc_mask = mask_x | mask_y | mask_z
        bc_indices = torch.from_numpy(np.where(bc_mask)[0]).long().to(device)
        u_bc_vals = u_meas[bc_indices]

        for mu_type, mu_field in [('constant_5000', mu_constant), ('heterogeneous', mu_heterogeneous)]:
            for n_neurons in neurons_list:
                config_count += 1
                print(f"\n[{config_count}/{total_configs}] {sampling_name}, neurons={n_neurons}, mu={mu_type}")

                model = ForwardMREModel(
                    n_wave_neurons=n_neurons,
                    input_dim=3,
                    omega_basis=omega_basis,
                    mu_min=3000.0,
                    mu_max=10000.0,
                    seed=42,
                    basis_type='sin'
                ).to(device)

                metrics = test_forward_config(
                    model, x, u_meas, mu_field, bc_indices, u_bc_vals,
                    rho_omega2, bc_weight=bc_weight
                )

                blob_r2_str = f"{metrics['blob_r2']:.4f}" if metrics['blob_r2'] is not None else "N/A"
                print(f"  Overall R²: {metrics['r2']:.4f}, Blob R²: {blob_r2_str}")

                results.append({
                    'mu_type': mu_type,
                    'sampling_config': sampling_name,
                    'use_adaptive': use_adaptive,
                    'blob_ratio': blob_ratio if use_adaptive else 0.0,
                    'boundary_ratio': boundary_ratio if use_adaptive else 0.0,
                    'allow_replacement': allow_replacement,
                    'neurons': n_neurons,
                    **metrics
                })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'sampling_comparison_results.csv', index=False)

    # Analyze results
    print("\n" + "="*80)
    print("DISCRIMINATION ANALYSIS")
    print("="*80)

    df_const = results_df[results_df['mu_type'] == 'constant_5000']
    df_hetero = results_df[results_df['mu_type'] == 'heterogeneous']

    summary = []
    for samp_cfg in sorted(results_df['sampling_config'].unique()):
        for n_neurons in neurons_list:
            const_row = df_const[(df_const['sampling_config'] == samp_cfg) & (df_const['neurons'] == n_neurons)]
            hetero_row = df_hetero[(df_hetero['sampling_config'] == samp_cfg) & (df_hetero['neurons'] == n_neurons)]

            if len(const_row) == 0 or len(hetero_row) == 0:
                continue

            const_r2 = const_row.iloc[0]['r2']
            hetero_r2 = hetero_row.iloc[0]['r2']
            const_blob_r2 = const_row.iloc[0]['blob_r2']
            hetero_blob_r2 = hetero_row.iloc[0]['blob_r2']

            # Discrimination = how different are const vs hetero
            r2_diff = abs(hetero_r2 - const_r2)
            blob_r2_diff = abs(hetero_blob_r2 - const_blob_r2) if (const_blob_r2 and hetero_blob_r2) else 0

            print(f"\n{samp_cfg}, neurons={n_neurons}:")
            print(f"  Overall R²:  Const={const_r2:.4f}, Hetero={hetero_r2:.4f}, Diff={r2_diff:.4f}")
            if const_blob_r2 and hetero_blob_r2:
                print(f"  Blob R²:     Const={const_blob_r2:.4f}, Hetero={hetero_blob_r2:.4f}, Diff={blob_r2_diff:.4f}")

            summary.append({
                'sampling_config': samp_cfg,
                'neurons': n_neurons,
                'const_r2': const_r2,
                'hetero_r2': hetero_r2,
                'r2_diff': r2_diff,
                'const_blob_r2': const_blob_r2,
                'hetero_blob_r2': hetero_blob_r2,
                'blob_r2_diff': blob_r2_diff
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'discrimination_summary.csv', index=False)

    # Find best configuration
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    # Best for blob R² in heterogeneous case
    hetero_results = df_hetero[df_hetero['blob_r2'].notna()]
    if len(hetero_results) > 0:
        best_blob = hetero_results.loc[hetero_results['blob_r2'].idxmax()]
        print(f"\nBest Blob R² (heterogeneous):")
        print(f"  Config: {best_blob['sampling_config']}")
        print(f"  Neurons: {best_blob['neurons']}")
        print(f"  Blob R²: {best_blob['blob_r2']:.4f}")
        print(f"  Overall R²: {best_blob['r2']:.4f}")

    # Best discrimination (largest blob R² difference)
    if len(summary_df) > 0 and summary_df['blob_r2_diff'].notna().any():
        best_disc = summary_df.loc[summary_df['blob_r2_diff'].idxmax()]
        print(f"\nBest Discrimination (blob R² difference):")
        print(f"  Config: {best_disc['sampling_config']}")
        print(f"  Neurons: {best_disc['neurons']}")
        print(f"  Difference: {best_disc['blob_r2_diff']:.4f}")

    print(f"\n\nResults saved to:")
    print(f"  {output_dir / 'sampling_comparison_results.csv'}")
    print(f"  {output_dir / 'discrimination_summary.csv'}")
    print("="*80)


if __name__ == "__main__":
    main()
