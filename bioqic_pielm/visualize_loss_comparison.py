"""
Loss Function Visual Comparison: Constant vs Heterogeneous μ
=============================================================

Visualizes forward solve results with:
- BC weight = 10
- Neurons = 100
- Comparing: Constant μ=5000 Pa vs Heterogeneous μ (ground truth)
- All loss functions: MSE, Relative L2, Sobolev, Correlation

Shows:
1. Displacement predictions (u_pred) for both mu fields
2. Loss values for each loss function
3. Stiffness maps (mid Z-slice)
4. Error distributions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


def compute_all_losses(u_pred: torch.Tensor, u_meas: torch.Tensor, x: torch.Tensor) -> Dict:
    """Compute all loss metrics."""
    u_pred_flat = u_pred.flatten()
    u_meas_flat = u_meas.flatten()
    
    # MSE
    mse = torch.mean((u_pred - u_meas) ** 2).item()
    
    # R²
    var_u = torch.var(u_meas).item()
    r2 = 1 - mse / var_u if var_u > 0 else 0
    
    # Relative L2
    relative_l2 = (torch.sqrt(torch.mean((u_pred - u_meas) ** 2)) / 
                   (torch.norm(u_meas_flat) + 1e-8)).item()
    
    # Correlation
    u_p_norm = u_pred_flat / (torch.norm(u_pred_flat) + 1e-8)
    u_m_norm = u_meas_flat / (torch.norm(u_meas_flat) + 1e-8)
    correlation = torch.dot(u_p_norm, u_m_norm).item()
    correlation_loss = 1.0 - correlation
    
    # Sobolev (approximate with random subsampling)
    with torch.no_grad():
        n_points = min(len(x), 1000)
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
    
    sobolev = 0.5 * mse + 0.5 * loss_grad
    
    return {
        'mse': mse,
        'r2': r2,
        'relative_l2': relative_l2,
        'correlation': correlation,
        'correlation_loss': correlation_loss,
        'sobolev': sobolev
    }


def solve_and_visualize():
    """Main visualization function."""
    print("\n" + "="*80)
    print("LOSS FUNCTION COMPARISON: CONSTANT vs HETEROGENEOUS μ")
    print("Configuration: BC=10, Neurons=1000")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'loss_visual_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    bc_weight = 10
    n_neurons = 1000
    n_points = 10000
    omega_basis = 170.0

    # Load data
    print(f"\nLoading data...")
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
    mu_constant = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)

    print(f"  Points: {len(x)}")
    print(f"  Constant μ: 5000 Pa (uniform)")
    print(f"  Heterogeneous μ: [{mu_heterogeneous.min():.0f}, {mu_heterogeneous.max():.0f}] Pa")

    # Boundary conditions
    print("\nSetting up boundary conditions (Box)...")
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    tol = 1e-4

    mask = ((np.abs(coords[:, 0] - x_min) < tol) | (np.abs(coords[:, 0] - x_max) < tol) |
            (np.abs(coords[:, 1] - y_min) < tol) | (np.abs(coords[:, 1] - y_max) < tol) |
            (np.abs(coords[:, 2] - z_min) < tol) | (np.abs(coords[:, 2] - z_max) < tol))
    
    bc_indices = torch.from_numpy(np.where(mask)[0]).long().to(device)
    u_bc_vals = u_meas[bc_indices]

    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")

    # Physics
    freq = 60
    omega = 2 * np.pi * freq
    rho = 1000
    rho_omega2 = rho * omega ** 2

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

    print(f"\nRunning forward solves...")
    
    # Solve with constant mu
    print("\n1. Constant μ = 5000 Pa...")
    with torch.no_grad():
        u_pred_const, _ = model.solve_given_mu(
            x, mu_constant, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )
    losses_const = compute_all_losses(u_pred_const, u_meas, x)
    print(f"   MSE: {losses_const['mse']:.6e}, R²: {losses_const['r2']:.4f}")
    print(f"   Correlation: {losses_const['correlation']:.4f}, Loss: {losses_const['correlation_loss']:.6e}")
    
    # Solve with heterogeneous mu
    print("\n2. Heterogeneous μ (ground truth)...")
    with torch.no_grad():
        u_pred_hetero, _ = model.solve_given_mu(
            x, mu_heterogeneous, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )
    losses_hetero = compute_all_losses(u_pred_hetero, u_meas, x)
    print(f"   MSE: {losses_hetero['mse']:.6e}, R²: {losses_hetero['r2']:.4f}")
    print(f"   Correlation: {losses_hetero['correlation']:.4f}, Loss: {losses_hetero['correlation_loss']:.6e}")

    # Create comprehensive visualization
    print("\nCreating visualizations...")
    create_visualization(
        x, u_meas, 
        u_pred_const, u_pred_hetero,
        mu_constant, mu_heterogeneous,
        losses_const, losses_hetero,
        output_dir
    )

    print(f"\n✓ Visualizations saved to: {output_dir}")
    print("="*80)


def create_visualization(x, u_meas, u_pred_const, u_pred_hetero, 
                        mu_const, mu_hetero, losses_const, losses_hetero, output_dir):
    """Create comprehensive visualization comparing constant vs heterogeneous."""
    
    # Convert to numpy
    coords = x.cpu().numpy()
    u_meas_np = u_meas.cpu().numpy().flatten()
    u_const_np = u_pred_const.cpu().numpy().flatten()
    u_hetero_np = u_pred_hetero.cpu().numpy().flatten()
    mu_const_np = mu_const.cpu().numpy().flatten()
    mu_hetero_np = mu_hetero.cpu().numpy().flatten()
    
    # Get middle Z slice
    z_mid = (coords[:, 2].min() + coords[:, 2].max()) / 2
    z_tol = 0.01
    mid_mask = np.abs(coords[:, 2] - z_mid) < z_tol
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.35)
    
    # Row 1: Loss comparison bars
    ax = fig.add_subplot(gs[0, :2])
    loss_names = ['MSE', 'Relative L2', 'Sobolev', 'Corr Loss']
    const_vals = [losses_const['mse'], losses_const['relative_l2'], 
                  losses_const['sobolev'], losses_const['correlation_loss']]
    hetero_vals = [losses_hetero['mse'], losses_hetero['relative_l2'],
                   losses_hetero['sobolev'], losses_hetero['correlation_loss']]
    
    x_pos = np.arange(len(loss_names))
    width = 0.35
    ax.bar(x_pos - width/2, const_vals, width, label='Const μ=5000', alpha=0.8, color='skyblue')
    ax.bar(x_pos + width/2, hetero_vals, width, label='Hetero μ', alpha=0.8, color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(loss_names)
    ax.set_ylabel('Loss Value', fontsize=11)
    ax.set_title('Loss Function Comparison\n(Lower = Better Fit)', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (c_val, h_val) in enumerate(zip(const_vals, hetero_vals)):
        ax.text(i - width/2, c_val, f'{c_val:.2e}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, h_val, f'{h_val:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Row 1: Delta (Hetero - Const) - shows which is better
    ax = fig.add_subplot(gs[0, 2:])
    deltas = [h - c for h, c in zip(hetero_vals, const_vals)]
    colors = ['green' if d < 0 else 'red' for d in deltas]
    ax.barh(loss_names, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel('Δ Loss (Hetero - Const)', fontsize=11)
    ax.set_title('Loss Improvement\n(Negative = Hetero Better)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (name, delta) in enumerate(zip(loss_names, deltas)):
        sign = '✓' if delta < 0 else '✗'
        ax.text(delta, i, f'  {sign} {delta:.2e}', va='center', fontsize=10, weight='bold')
    
    # Row 2: Stiffness maps (mid Z-slice)
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1], 
                   c=mu_const_np[mid_mask], s=20, cmap='jet', vmin=3000, vmax=10000)
    plt.colorbar(sc, ax=ax, label='μ (Pa)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Constant μ = 5000 Pa\n(Mid Z-slice)', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                   c=mu_hetero_np[mid_mask], s=20, cmap='jet', vmin=3000, vmax=10000)
    plt.colorbar(sc, ax=ax, label='μ (Pa)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Heterogeneous μ (Ground Truth)\n(Mid Z-slice)', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    # Row 2: Stiffness histogram
    ax = fig.add_subplot(gs[1, 2:])
    ax.hist(mu_const_np, bins=30, alpha=0.5, label='Constant', density=True, color='skyblue')
    ax.hist(mu_hetero_np, bins=30, alpha=0.5, label='Heterogeneous', density=True, color='coral')
    ax.set_xlabel('μ (Pa)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Stiffness Distribution', fontsize=12, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Row 3: Displacement predictions (mid Z-slice)
    vmin_u = min(u_meas_np.min(), u_const_np.min(), u_hetero_np.min())
    vmax_u = max(u_meas_np.max(), u_const_np.max(), u_hetero_np.max())
    
    ax = fig.add_subplot(gs[2, 0])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                   c=u_meas_np[mid_mask], s=20, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    plt.colorbar(sc, ax=ax, label='u (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Measured u (Ground Truth)', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(gs[2, 1])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                   c=u_const_np[mid_mask], s=20, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    plt.colorbar(sc, ax=ax, label='u (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Predicted u (Const μ)\nR²={losses_const["r2"]:.3f}', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(gs[2, 2])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                   c=u_hetero_np[mid_mask], s=20, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    plt.colorbar(sc, ax=ax, label='u (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Predicted u (Hetero μ)\nR²={losses_hetero["r2"]:.3f}', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    # Row 3: Error maps
    error_const = u_const_np - u_meas_np
    error_hetero = u_hetero_np - u_meas_np
    vmax_err = max(np.abs(error_const).max(), np.abs(error_hetero).max())
    
    ax = fig.add_subplot(gs[2, 3])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                   c=error_const[mid_mask], s=20, cmap='seismic', vmin=-vmax_err, vmax=vmax_err)
    plt.colorbar(sc, ax=ax, label='Error (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Error (Const μ)', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(gs[2, 4])
    sc = ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1],
                   c=error_hetero[mid_mask], s=20, cmap='seismic', vmin=-vmax_err, vmax=vmax_err)
    plt.colorbar(sc, ax=ax, label='Error (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Error (Hetero μ)', fontsize=11, weight='bold')
    ax.set_aspect('equal')
    
    # Row 4: Scatter plots u_pred vs u_meas
    ax = fig.add_subplot(gs[3, 0:2])
    ax.scatter(u_meas_np, u_const_np, alpha=0.3, s=2, label='Const μ', color='skyblue')
    lims = [min(u_meas_np.min(), u_const_np.min()), max(u_meas_np.max(), u_const_np.max())]
    ax.plot(lims, lims, 'k--', lw=2, label='Perfect')
    ax.set_xlabel('Measured u (m)', fontsize=11)
    ax.set_ylabel('Predicted u (m)', fontsize=11)
    ax.set_title(f'Const μ Fit (R²={losses_const["r2"]:.4f})', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(gs[3, 2:4])
    ax.scatter(u_meas_np, u_hetero_np, alpha=0.3, s=2, label='Hetero μ', color='coral')
    lims = [min(u_meas_np.min(), u_hetero_np.min()), max(u_meas_np.max(), u_hetero_np.max())]
    ax.plot(lims, lims, 'k--', lw=2, label='Perfect')
    ax.set_xlabel('Measured u (m)', fontsize=11)
    ax.set_ylabel('Predicted u (m)', fontsize=11)
    ax.set_title(f'Hetero μ Fit (R²={losses_hetero["r2"]:.4f})', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Row 4: Error distributions
    ax = fig.add_subplot(gs[3, 4])
    ax.hist(error_const, bins=50, alpha=0.5, label='Const μ', density=True, color='skyblue')
    ax.hist(error_hetero, bins=50, alpha=0.5, label='Hetero μ', density=True, color='coral')
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Error (m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Loss Function Comparison: Constant vs Heterogeneous Stiffness\n' +
                 f'BC Weight=10, Neurons=1000, Omega=170 rad/m',
                 fontsize=16, weight='bold', y=0.995)
    
    plt.savefig(output_dir / 'loss_comparison_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    solve_and_visualize()
