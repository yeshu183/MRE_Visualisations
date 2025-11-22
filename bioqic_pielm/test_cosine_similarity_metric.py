"""
Test Cosine Similarity vs MSE Loss for Forward Model
=====================================================

Compare two loss metrics:
1. MSE (Mean Squared Error): Penalizes amplitude differences
2. Cosine Similarity: Only cares about shape/phase alignment

Hypothesis: Cosine similarity will better distinguish between:
- Homogeneous μ (wrong) → Plane wave → Poor shape match
- Heterogeneous μ (correct) → Scattered wave → Good shape match

Even when actuator BC causes amplitude mismatch.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


def compute_metrics(u_pred: torch.Tensor, u_meas: torch.Tensor) -> Dict:
    """Compute both MSE and Cosine Similarity metrics."""
    # MSE and R²
    error = u_pred - u_meas
    mse = torch.mean(error ** 2).item()
    var_u = torch.var(u_meas).item()
    r2 = 1 - mse / var_u if var_u > 0 else 0
    
    # Cosine Similarity (Correlation)
    u_p_flat = u_pred.view(-1)
    u_m_flat = u_meas.view(-1)
    
    # Normalize vectors
    u_p_norm = u_p_flat / (torch.norm(u_p_flat) + 1e-8)
    u_m_norm = u_m_flat / (torch.norm(u_m_flat) + 1e-8)
    
    # Dot product
    correlation = torch.dot(u_p_norm, u_m_norm).item()
    
    # Correlation loss (1 - correlation)
    corr_loss = 1.0 - correlation
    
    # Amplitude ratio
    amp_pred = torch.norm(u_p_flat).item()
    amp_meas = torch.norm(u_m_flat).item()
    amp_ratio = amp_pred / amp_meas if amp_meas > 0 else 0
    
    return {
        'mse': mse,
        'r2': r2,
        'correlation': correlation,
        'corr_loss': corr_loss,
        'amp_ratio': amp_ratio
    }


def test_forward_with_mu(
    model: ForwardMREModel,
    x: torch.Tensor,
    u_meas: torch.Tensor,
    mu: torch.Tensor,
    bc_indices: torch.Tensor,
    u_bc_vals: torch.Tensor,
    rho_omega2: float,
    bc_weight: float,
    label: str
):
    """Test forward solve and compute metrics."""
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"{'='*70}")
    
    with torch.no_grad():
        u_pred, _ = model.solve_given_mu(
            x, mu, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )
    
    metrics = compute_metrics(u_pred, u_meas)
    
    print(f"MSE:                {metrics['mse']:.6e}")
    print(f"R²:                 {metrics['r2']:.4f}")
    print(f"Correlation:        {metrics['correlation']:.4f}")
    print(f"Correlation Loss:   {metrics['corr_loss']:.4f}")
    print(f"Amplitude Ratio:    {metrics['amp_ratio']:.4f} (pred/meas)")
    
    return metrics, u_pred


def main():
    print("\n" + "="*70)
    print("COSINE SIMILARITY vs MSE: FORWARD MODEL COMPARISON")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'cosine_similarity_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    n_points = 10000
    print(f"\nLoading data ({n_points} points)...")
    loader = BIOQICDataLoader(
        data_dir=str(data_dir),
        displacement_mode='z_component',
        subsample=n_points,
        seed=42
    )
    data = loader.load()
    
    coords = data['coords']
    u_raw = data['u_raw']
    mu_raw = data['mu_raw']
    
    x = torch.from_numpy(coords).float().to(device)
    u_meas = torch.from_numpy(u_raw).float().to(device)
    mu_heterogeneous = torch.from_numpy(mu_raw).float().to(device)
    mu_homogeneous = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)
    
    # Physics parameters
    freq = 60
    rho_omega2 = 1000 * (2 * np.pi * freq) ** 2
    
    # Test both BC strategies
    bc_strategies = {
        'box': 'Box (all 6 faces)',
        'actuator': 'Actuator (top face only)'
    }
    
    results = {}
    
    for bc_strategy, bc_label in bc_strategies.items():
        print("\n" + "="*70)
        print(f"BC STRATEGY: {bc_label}")
        print("="*70)
        
        # Setup BCs
        if bc_strategy == 'box':
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
            tol = 1e-4
            mask_x = (np.abs(coords[:, 0] - x_min) < tol) | (np.abs(coords[:, 0] - x_max) < tol)
            mask_y = (np.abs(coords[:, 1] - y_min) < tol) | (np.abs(coords[:, 1] - y_max) < tol)
            mask_z = (np.abs(coords[:, 2] - z_min) < tol) | (np.abs(coords[:, 2] - z_max) < tol)
            bc_mask = mask_x | mask_y | mask_z
            bc_weight = 10
        else:  # actuator
            z_max = coords[:, 2].max()
            tol = 1e-4
            bc_mask = np.abs(coords[:, 2] - z_max) < tol
            bc_weight = 1000
        
        bc_indices = torch.from_numpy(np.where(bc_mask)[0]).long().to(device)
        u_bc_vals = u_meas[bc_indices]
        
        print(f"BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")
        print(f"BC weight: {bc_weight}")
        
        # Create model
        model = ForwardMREModel(
            n_wave_neurons=1000,
            input_dim=3,
            omega_basis=170.0,
            mu_min=3000.0,
            mu_max=10000.0,
            seed=42,
            basis_type='sin'
        ).to(device)
        
        # Test 1: Homogeneous mu (wrong)
        metrics_homo, u_pred_homo = test_forward_with_mu(
            model, x, u_meas, mu_homogeneous, bc_indices, u_bc_vals,
            rho_omega2, bc_weight, f"Homogeneous μ=5000 Pa ({bc_strategy})"
        )
        
        # Test 2: Heterogeneous mu (correct)
        metrics_hetero, u_pred_hetero = test_forward_with_mu(
            model, x, u_meas, mu_heterogeneous, bc_indices, u_bc_vals,
            rho_omega2, bc_weight, f"Heterogeneous μ (inclusions) ({bc_strategy})"
        )
        
        # Store results
        results[bc_strategy] = {
            'homo': metrics_homo,
            'hetero': metrics_hetero,
            'u_pred_homo': u_pred_homo,
            'u_pred_hetero': u_pred_hetero
        }
        
        # Analysis
        print(f"\n{'='*70}")
        print(f"METRIC DISCRIMINATION ({bc_strategy})")
        print(f"{'='*70}")
        
        # MSE ratio
        mse_ratio = metrics_homo['mse'] / metrics_hetero['mse']
        print(f"MSE Ratio (homo/hetero):        {mse_ratio:.4f}")
        
        # R² difference
        r2_diff = metrics_hetero['r2'] - metrics_homo['r2']
        print(f"R² Improvement (hetero-homo):   {r2_diff:.4f}")
        
        # Correlation difference
        corr_diff = metrics_hetero['correlation'] - metrics_homo['correlation']
        print(f"Correlation Improvement:        {corr_diff:.4f}")
        
        # Correlation loss difference
        corr_loss_diff = metrics_homo['corr_loss'] - metrics_hetero['corr_loss']
        print(f"Corr Loss Reduction (↓better):  {corr_loss_diff:.4f}")
        
        print(f"\n{'='*70}")
    
    # Visualization
    print("\nCreating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    coords_np = x.cpu().numpy()
    u_meas_np = u_meas.cpu().numpy().flatten()
    mu_hetero_np = mu_heterogeneous.cpu().numpy().flatten()
    
    # Get middle slice
    z_mid = np.median(coords_np[:, 2])
    z_tol = 0.002
    z_mask = np.abs(coords_np[:, 2] - z_mid) < z_tol
    
    u_vmin, u_vmax = -0.025, 0.025
    
    for row, (bc_strategy, bc_label) in enumerate(bc_strategies.items()):
        res = results[bc_strategy]
        
        u_homo_np = res['u_pred_homo'].cpu().numpy().flatten()
        u_hetero_np = res['u_pred_hetero'].cpu().numpy().flatten()
        
        # Measured
        ax = axes[row, 0]
        sc = ax.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=u_meas_np[z_mask],
                        cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
        ax.tricontour(coords_np[z_mask, 0], coords_np[z_mask, 1], mu_hetero_np[z_mask],
                      levels=[5000, 7000, 9000], colors='red', linewidths=1.5, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='u (m)')
        ax.set_title(f'{bc_label}\nu_measured')
        ax.set_aspect('equal')
        
        # Homogeneous
        ax = axes[row, 1]
        sc = ax.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=u_homo_np[z_mask],
                        cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
        plt.colorbar(sc, ax=ax, label='u (m)')
        ax.set_title(f'Homogeneous μ\nR²={res["homo"]["r2"]:.3f}, Corr={res["homo"]["correlation"]:.3f}')
        ax.set_aspect('equal')
        
        # Heterogeneous
        ax = axes[row, 2]
        sc = ax.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=u_hetero_np[z_mask],
                        cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
        plt.colorbar(sc, ax=ax, label='u (m)')
        ax.set_title(f'Heterogeneous μ\nR²={res["hetero"]["r2"]:.3f}, Corr={res["hetero"]["correlation"]:.3f}')
        ax.set_aspect('equal')
        
        # Metrics comparison
        ax = axes[row, 3]
        metrics_labels = ['R²', 'Correlation', '1-Corr Loss']
        homo_vals = [res['homo']['r2'], res['homo']['correlation'], 1 - res['homo']['corr_loss']]
        hetero_vals = [res['hetero']['r2'], res['hetero']['correlation'], 1 - res['hetero']['corr_loss']]
        
        x_pos = np.arange(len(metrics_labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, homo_vals, width, label='Homogeneous', alpha=0.7, color='orange')
        ax.bar(x_pos + width/2, hetero_vals, width, label='Heterogeneous', alpha=0.7, color='green')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylabel('Score')
        ax.set_title(f'{bc_label}\nMetric Comparison')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect')
    
    plt.suptitle('Forward Model: MSE vs Cosine Similarity Discrimination\n' +
                 'Higher correlation = Better shape match (ignoring amplitude)', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_similarity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_dir / 'cosine_similarity_comparison.png'}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for bc_strategy, bc_label in bc_strategies.items():
        res = results[bc_strategy]
        print(f"\n{bc_label}:")
        print(f"  Homogeneous:    R²={res['homo']['r2']:.4f}, Corr={res['homo']['correlation']:.4f}, CorrLoss={res['homo']['corr_loss']:.4f}")
        print(f"  Heterogeneous:  R²={res['hetero']['r2']:.4f}, Corr={res['hetero']['correlation']:.4f}, CorrLoss={res['hetero']['corr_loss']:.4f}")
        
        r2_improvement = res['hetero']['r2'] - res['homo']['r2']
        corr_improvement = res['hetero']['correlation'] - res['homo']['correlation']
        
        print(f"  ΔR²:            {r2_improvement:.4f}")
        print(f"  ΔCorrelation:   {corr_improvement:.4f}")
        
        if abs(corr_improvement) > abs(r2_improvement):
            print(f"  → Correlation is MORE discriminative ({abs(corr_improvement)/abs(r2_improvement) if r2_improvement != 0 else np.inf:.2f}x)")
        else:
            print(f"  → R² is more discriminative")
    
    print("="*70)


if __name__ == "__main__":
    main()
