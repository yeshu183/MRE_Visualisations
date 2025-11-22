"""
Forward Model Loss Function Comparison
========================================

Tests which loss function best discriminates between:
- Homogeneous mu field (constant 5000 Pa)
- Heterogeneous mu field (with inclusions from ground truth)

Loss functions tested:
1. MSE (Mean Squared Error)
2. Relative L2 (Normalized MSE)
3. Sobolev Loss (MSE + Gradient term)
4. Correlation Loss (Cosine similarity)

Goal: Find which loss shows largest difference between homogeneous 
and heterogeneous fields, indicating better sensitivity to stiffness variations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
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


def compute_loss_metrics(
    u_pred: torch.Tensor,
    u_meas: torch.Tensor,
    x: torch.Tensor
) -> Dict:
    """Compute all loss metrics for comparison.
    
    Args:
        u_pred: Predicted displacement (N, 1)
        u_meas: Measured displacement (N, 1)
        x: Spatial coordinates (N, 3)
    
    Returns:
        Dictionary with all loss values
    """
    u_pred_flat = u_pred.flatten()
    u_meas_flat = u_meas.flatten()
    
    # 1. MSE (Standard Mean Squared Error)
    mse = torch.mean((u_pred - u_meas) ** 2).item()
    
    # 2. R² Score (for reference)
    var_u = torch.var(u_meas).item()
    r2 = 1 - mse / var_u if var_u > 0 else 0
    
    # 3. Relative L2 (Normalized by measurement magnitude)
    relative_l2 = torch.sqrt(torch.mean((u_pred - u_meas) ** 2)) / (torch.norm(u_meas_flat) + 1e-8)
    relative_l2 = relative_l2.item()
    
    # 4. Correlation (Cosine Similarity)
    u_p_norm = u_pred_flat / (torch.norm(u_pred_flat) + 1e-8)
    u_m_norm = u_meas_flat / (torch.norm(u_meas_flat) + 1e-8)
    correlation = torch.dot(u_p_norm, u_m_norm).item()
    correlation_loss = 1.0 - correlation
    
    # 5. Sobolev Loss (MSE + Gradient term)
    # Approximate gradients using finite differences
    with torch.no_grad():
        # Use forward differences on sorted points
        n_points = min(len(x), 1000)  # Subsample for speed
        indices = torch.randperm(len(x))[:n_points]
        
        if n_points > 10:
            du_pred = u_pred[indices[1:]] - u_pred[indices[:-1]]
            du_meas = u_meas[indices[1:]] - u_meas[indices[:-1]]
            dx = torch.norm(x[indices[1:]] - x[indices[:-1]], dim=1, keepdim=True) + 1e-8
            
            grad_pred = du_pred / dx
            grad_meas = du_meas / dx
            
            loss_grad = torch.mean((grad_pred - grad_meas) ** 2).item()
        else:
            loss_grad = 0.0
    
    # Sobolev = α*MSE + β*grad_loss (α=0.5, β=0.5 for balanced contribution)
    sobolev_loss = 0.5 * mse + 0.5 * loss_grad
    
    return {
        'mse': mse,
        'r2': r2,
        'relative_l2': relative_l2,
        'correlation': correlation,
        'correlation_loss': correlation_loss,
        'sobolev': sobolev_loss,
        'grad_term': loss_grad
    }


def test_forward_config(
    model: ForwardMREModel,
    x: torch.Tensor,
    u_meas: torch.Tensor,
    mu: torch.Tensor,
    bc_indices: torch.Tensor,
    u_bc_vals: torch.Tensor,
    rho_omega2: float,
    bc_weight: float
) -> Dict:
    """Test forward solve and return all loss metrics."""
    with torch.no_grad():
        u_pred, _ = model.solve_given_mu(
            x, mu, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )

        # Compute all metrics
        metrics = compute_loss_metrics(u_pred, u_meas, x)

        # Compute region-specific metrics
        region_metrics = compute_region_metrics(u_pred, u_meas, mu)

        # Add max error
        error = u_pred - u_meas
        metrics['max_error'] = torch.max(torch.abs(error)).item()
        metrics['u_pred'] = u_pred.cpu().numpy()

        # Add region-specific metrics
        metrics['region_metrics'] = region_metrics
        if region_metrics['blob'] is not None:
            metrics['blob_r2'] = region_metrics['blob']['r2']
            metrics['blob_mse'] = region_metrics['blob']['mse']
        else:
            metrics['blob_r2'] = None
            metrics['blob_mse'] = None

        if region_metrics['background'] is not None:
            metrics['background_r2'] = region_metrics['background']['r2']
            metrics['background_mse'] = region_metrics['background']['mse']
        else:
            metrics['background_r2'] = None
            metrics['background_mse'] = None

    return metrics


def analyze_results(results_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualization comparing sampling configurations."""

    # Separate constant vs heterogeneous results
    df_const = results_df[results_df['mu_type'] == 'constant_5000']
    df_hetero = results_df[results_df['mu_type'] == 'heterogeneous']

    sampling_configs = sorted(results_df['sampling_config'].unique())
    neurons = sorted(results_df['neurons'].unique())
    loss_metrics = ['mse', 'relative_l2', 'sobolev', 'correlation_loss']
    loss_labels = ['MSE', 'Relative L2', 'Sobolev', 'Correlation Loss']

    # For each sampling config and neuron, compute discrimination ability
    print("\n" + "="*80)
    print("LOSS FUNCTION DISCRIMINATION ANALYSIS BY SAMPLING CONFIGURATION")
    print("="*80)
    print("\nMetric: Relative Difference = |Loss(heterogeneous) - Loss(constant)| / Loss(constant)")
    print("Higher relative difference = Better discrimination between mu fields\n")

    discrimination_summary = []

    for samp_cfg in sampling_configs:
        for n_neurons in neurons:
            print(f"\nBC Weight={bc_w}, Neurons={n_neurons}:")
            print("-" * 70)
            
            row_const = df_const[(df_const['bc_weight'] == bc_w) & (df_const['neurons'] == n_neurons)].iloc[0]
            row_hetero = df_hetero[(df_hetero['bc_weight'] == bc_w) & (df_hetero['neurons'] == n_neurons)].iloc[0]
            
            for metric, label in zip(loss_metrics, loss_labels):
                val_const = row_const[metric]
                val_hetero = row_hetero[metric]
                abs_diff = abs(val_hetero - val_const)
                rel_diff = abs_diff / (abs(val_const) + 1e-12)  # Relative difference
                
                print(f"  {label:20s}: Const={val_const:.6e}, Hetero={val_hetero:.6e}, "
                      f"RelDiff={rel_diff:.4f}")
                
                discrimination_summary.append({
                    'bc_weight': bc_w,
                    'neurons': n_neurons,
                    'loss_type': label,
                    'const_value': val_const,
                    'hetero_value': val_hetero,
                    'abs_diff': abs_diff,
                    'rel_diff': rel_diff
                })
    
    disc_df = pd.DataFrame(discrimination_summary)
    disc_df.to_csv(output_dir / 'loss_discrimination_analysis.csv', index=False)
    
    # Create visualization comparing loss functions
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Row 1: Loss values for each configuration
    for idx, (metric, label) in enumerate(zip(loss_metrics, loss_labels)):
        ax = fig.add_subplot(gs[0, idx])
        
        for i, bc_w in enumerate(bc_weights):
            const_vals = []
            hetero_vals = []
            for n in neurons:
                const_vals.append(df_const[(df_const['bc_weight'] == bc_w) & 
                                           (df_const['neurons'] == n)][metric].values[0])
                hetero_vals.append(df_hetero[(df_hetero['bc_weight'] == bc_w) & 
                                             (df_hetero['neurons'] == n)][metric].values[0])
            
            ax.plot(neurons, const_vals, 'o--', label=f'Const (BC={bc_w})', 
                   markersize=6, linewidth=2, alpha=0.7)
            ax.plot(neurons, hetero_vals, 's-', label=f'Hetero (BC={bc_w})', 
                   markersize=6, linewidth=2, alpha=0.9)
        
        ax.set_xscale('log')
        ax.set_xlabel('Neurons', fontsize=10)
        ax.set_ylabel(f'{label} Value', fontsize=10)
        ax.set_title(f'{label}', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    
    # Row 2: Relative discrimination ability (bar chart)
    for idx, (metric, label) in enumerate(zip(loss_metrics, loss_labels)):
        ax = fig.add_subplot(gs[1, idx])
        
        # Average relative difference across all configs
        avg_rel_diff = disc_df[disc_df['loss_type'] == label].groupby(['bc_weight', 'neurons'])['rel_diff'].mean()
        
        x_pos = []
        heights = []
        labels_list = []
        colors = []
        
        color_map = {bc_weights[0]: 'skyblue', bc_weights[1]: 'orange', bc_weights[2]: 'green'}
        
        pos = 0
        for bc_w in bc_weights:
            for n in neurons:
                try:
                    val = avg_rel_diff.loc[(bc_w, n)]
                    x_pos.append(pos)
                    heights.append(val)
                    labels_list.append(f'{n}')
                    colors.append(color_map[bc_w])
                    pos += 1
                except KeyError:
                    pos += 1
            pos += 0.5  # Gap between BC groups
        
        ax.bar(x_pos, heights, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, rotation=45, fontsize=8)
        ax.set_ylabel('Relative Difference', fontsize=10)
        ax.set_title(f'{label} Discrimination', fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend for BC weights
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[bc_w], label=f'BC={bc_w}') 
                          for bc_w in bc_weights]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    
    # Row 3: Direct comparison - which loss is best?
    ax = fig.add_subplot(gs[2, :2])
    
    # For each config, find which loss has highest discrimination
    best_loss_counts = disc_df.groupby(['bc_weight', 'neurons', 'loss_type'])['rel_diff'].mean().reset_index()
    best_per_config = best_loss_counts.loc[best_loss_counts.groupby(['bc_weight', 'neurons'])['rel_diff'].idxmax()]
    
    loss_ranking = best_per_config['loss_type'].value_counts()
    
    ax.bar(range(len(loss_ranking)), loss_ranking.values, 
           color=['steelblue', 'coral', 'lightgreen', 'gold'][:len(loss_ranking)], 
           alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(loss_ranking)))
    ax.set_xticklabels(loss_ranking.index, fontsize=11)
    ax.set_ylabel('# Configurations Where Best', fontsize=11)
    ax.set_title('Loss Function Ranking (Best Discrimination Frequency)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text labels on bars
    for i, (idx, val) in enumerate(zip(range(len(loss_ranking)), loss_ranking.values)):
        ax.text(idx, val + 0.1, str(val), ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Row 3: Average discrimination across all configs
    ax = fig.add_subplot(gs[2, 2:])
    
    avg_discrimination = disc_df.groupby('loss_type')['rel_diff'].mean().sort_values(ascending=False)
    
    ax.barh(range(len(avg_discrimination)), avg_discrimination.values,
            color=['steelblue', 'coral', 'lightgreen', 'gold'][:len(avg_discrimination)],
            alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_yticks(range(len(avg_discrimination)))
    ax.set_yticklabels(avg_discrimination.index, fontsize=11)
    ax.set_xlabel('Average Relative Difference', fontsize=11)
    ax.set_title('Overall Discrimination Power', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, val in enumerate(avg_discrimination.values):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=11, weight='bold')
    
    plt.suptitle('Loss Function Comparison: Sensitivity to Stiffness Heterogeneity', 
                 fontsize=16, y=0.995, weight='bold')
    plt.savefig(output_dir / 'loss_function_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("SUMMARY: BEST LOSS FUNCTION")
    print("="*80)
    print(f"\nAverage Discrimination Power (Higher = Better):")
    for loss_name, disc_val in avg_discrimination.items():
        print(f"  {loss_name:20s}: {disc_val:.4f}")
    
    print(f"\nBest Overall: {avg_discrimination.idxmax()} (Rel Diff = {avg_discrimination.max():.4f})")
    print(f"\nVisualization saved: {output_dir / 'loss_function_comparison.png'}")
    print("="*80)


def main():
    print("\n" + "="*80)
    print("FORWARD MODEL LOSS FUNCTION COMPARISON")
    print("="*80)
    print("\nGoal: Find which loss function best distinguishes between:")
    print("  - Homogeneous mu (constant 5000 Pa)")
    print("  - Heterogeneous mu (with inclusions)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'loss_function_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed parameters
    bc_weight = 10
    n_points = 5000
    neurons_list = [100, 500, 1000]
    omega_basis = 170.0

    # ADAPTIVE SAMPLING CONFIGURATIONS TO TEST
    # Each config: (use_adaptive, blob_ratio, boundary_ratio, allow_replacement, name)
    sampling_configs = [
        # Uniform baseline
        (False, 0.0, 0.0, False, 'uniform'),

        # Adaptive with replacement (original approach)
        (True, 0.05, 0.25, True, 'adaptive_5_25_70_repl'),    # Focus on boundaries
        (True, 0.10, 0.20, True, 'adaptive_10_20_70_repl'),   # Balanced
        (True, 0.20, 0.10, True, 'adaptive_20_10_70_repl'),   # More blob interior

        # Adaptive WITHOUT replacement (capped at available points)
        (True, 0.05, 0.25, False, 'adaptive_5_25_70_noRepl'),
        (True, 0.10, 0.20, False, 'adaptive_10_20_70_noRepl'),
        (True, 0.20, 0.10, False, 'adaptive_20_10_70_noRepl'),
    ]

    print(f"\nFixed Parameters:")
    print(f"  BC weight: {bc_weight}")
    print(f"  Sampling points: {n_points}")
    print(f"  Neurons: {neurons_list}")
    print(f"  Omega basis: {omega_basis}")

    print(f"\nSampling Configurations to Test:")
    for i, (use_adap, blob_r, bound_r, allow_repl, name) in enumerate(sampling_configs, 1):
        if use_adap:
            print(f"  {i}. {name}: blob={blob_r*100:.0f}%, boundary={bound_r*100:.0f}%, "
                  f"bg={100*(1-blob_r-bound_r):.0f}%, replacement={allow_repl}")
        else:
            print(f"  {i}. {name}: uniform random sampling")

    print(f"\nLoss functions: MSE, Relative L2, Sobolev, Correlation")

    # Physics parameters
    freq = 60  # Hz
    omega = 2 * np.pi * freq
    rho = 1000  # kg/m³
    rho_omega2 = rho * omega ** 2

    print(f"\nPhysics parameters:")
    print(f"  Frequency: {freq} Hz")
    print(f"  rho_omega2: {rho_omega2:.2e}")

    # Run grid search over sampling configurations
    results = []
    total_configs = len(sampling_configs) * len(neurons_list) * 2  # × 2 for two mu types
    config_count = 0

    print("\n" + "="*80)
    print("RUNNING FORWARD SOLVES")
    print("="*80)

    for use_adaptive, blob_ratio, boundary_ratio, allow_replacement, sampling_name in sampling_configs:
        print(f"\n{'='*80}")
        print(f"SAMPLING CONFIG: {sampling_name.upper()}")
        print(f"{'='*80}")

        # Load data with current sampling configuration
        print(f"Loading data from: {data_dir}")
        loader = BIOQICDataLoader(
            data_dir=str(data_dir),
            displacement_mode='z_component',
            subsample=n_points,
            seed=42,
            adaptive_sampling=use_adaptive,
            blob_sample_ratio=blob_ratio,
            boundary_sample_ratio=boundary_ratio,
            allow_replacement=allow_replacement  # New parameter
        )
        data = loader.load()

        coords = data['coords']
        u_raw = data['u_raw']
        mu_raw = data['mu_raw']

        x = torch.from_numpy(coords).float().to(device)
        u_meas = torch.from_numpy(u_raw).float().to(device)
        mu_heterogeneous = torch.from_numpy(mu_raw).float().to(device)
        mu_constant = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)

        print(f"\nMu field statistics:")
        print(f"  Constant: {mu_constant[0, 0].item():.0f} Pa (uniform)")
        print(f"  Heterogeneous: [{mu_heterogeneous.min():.0f}, {mu_heterogeneous.max():.0f}] Pa")

        # Boundary conditions - FULL BOX
        print("\nSetting up boundary conditions (Box strategy)...")
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

        print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")

        for mu_type, mu_field in [('constant_5000', mu_constant), ('heterogeneous', mu_heterogeneous)]:
            print(f"\n{'-'*80}")
            print(f"MU FIELD: {mu_type.upper()}")
            print(f"{'-'*80}")

            for n_neurons in neurons_list:
                config_count += 1
                print(f"\n[{config_count}/{total_configs}] sampling={sampling_name}, neurons={n_neurons}, mu={mu_type}")

                # Create model
                model = ForwardMREModel(
                    n_wave_neurons=n_neurons,
                    input_dim=3,
                    omega_basis=omega_basis,
                    mu_min=3000.0,
                    mu_max=10000.0,
                    seed=42,
                    basis_type='sin'
                ).to(device)

                # Test and compute all metrics
                metrics = test_forward_config(
                    model, x, u_meas, mu_field, bc_indices, u_bc_vals,
                    rho_omega2, bc_weight=bc_weight
                )

                print(f"  Overall R²:  {metrics['r2']:.4f}")
                print(f"  MSE:         {metrics['mse']:.6e}")
                print(f"  Relative L2: {metrics['relative_l2']:.6e}")
                print(f"  Sobolev:     {metrics['sobolev']:.6e}")
                print(f"  Correlation: {metrics['correlation']:.4f} (loss={metrics['correlation_loss']:.6e})")

                if metrics['blob_r2'] is not None:
                    print(f"  Blob R²:     {metrics['blob_r2']:.4f} (MSE={metrics['blob_mse']:.6e})")
                if metrics['background_r2'] is not None:
                    print(f"  Background R²: {metrics['background_r2']:.4f} (MSE={metrics['background_mse']:.6e})")

                results.append({
                    'mu_type': mu_type,
                    'sampling_config': sampling_name,
                    'use_adaptive': use_adaptive,
                    'blob_ratio': blob_ratio if use_adaptive else 0.0,
                    'boundary_ratio': boundary_ratio if use_adaptive else 0.0,
                    'allow_replacement': allow_replacement,
                    'neurons': n_neurons,
                    'mse': metrics['mse'],
                    'r2': metrics['r2'],
                    'blob_r2': metrics['blob_r2'],
                    'blob_mse': metrics['blob_mse'],
                    'background_r2': metrics['background_r2'],
                    'background_mse': metrics['background_mse'],
                    'relative_l2': metrics['relative_l2'],
                    'correlation': metrics['correlation'],
                    'correlation_loss': metrics['correlation_loss'],
                    'sobolev': metrics['sobolev'],
                    'grad_term': metrics['grad_term'],
                    'max_error': metrics['max_error']
                })

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'loss_comparison_results.csv', index=False)
    print(f"\n\nSaved results to: {output_dir / 'loss_comparison_results.csv'}")

    # Visualize and analyze
    analyze_results(results_df, output_dir)

    # Print full results table
    print("\n" + "="*80)
    print("FULL RESULTS TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
