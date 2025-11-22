"""
Sobolev Loss Weight Sweep: Optimal α and β
===========================================

Tests different combinations of α (L2 weight) and β (gradient weight)
in Sobolev loss where α + β = 1.0

Sobolev Loss = α * ||u_pred - u_meas||² + β * ||∇u_pred - ∇u_meas||²

Sweeps α from 0.0 to 1.0 in steps of 0.1 (β = 1.0 - α)

Goal: Find which α/β combination shows maximum discrimination between
constant μ and heterogeneous μ fields.

Configuration: BC=10, Neurons=1000
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


def compute_sobolev_loss(
    u_pred: torch.Tensor, 
    u_meas: torch.Tensor, 
    x: torch.Tensor,
    alpha: float,
    beta: float
) -> float:
    """Compute Sobolev loss with specified α and β weights.
    
    Args:
        u_pred: Predicted displacement (N, 1)
        u_meas: Measured displacement (N, 1)
        x: Spatial coordinates (N, 3)
        alpha: Weight for L2 term
        beta: Weight for gradient term
    
    Returns:
        Sobolev loss value
    """
    # L2 term
    loss_l2 = torch.mean((u_pred - u_meas) ** 2).item()
    
    # Gradient term (finite difference approximation)
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
    
    sobolev = alpha * loss_l2 + beta * loss_grad
    
    return sobolev, loss_l2, loss_grad


def run_weight_sweep():
    """Run sweep over α values from 0.0 to 1.0."""
    print("\n" + "="*80)
    print("SOBOLEV LOSS WEIGHT SWEEP")
    print("Configuration: BC=10, Neurons=1000")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'sobolev_weight_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    bc_weight = 10
    n_neurons = 1000
    n_points = 10000
    omega_basis = 170.0
    
    # Alpha sweep (beta = 1 - alpha)
    alpha_values = np.arange(0.0, 1.1, 0.1)
    beta_values = 1.0 - alpha_values

    print(f"\nSweeping α from 0.0 to 1.0 (β = 1 - α)")
    print(f"  α values: {alpha_values}")
    print(f"  β values: {beta_values}")

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
    
    # Solve ONCE for each mu field
    print("  1. Constant μ = 5000 Pa...")
    with torch.no_grad():
        u_pred_const, _ = model.solve_given_mu(
            x, mu_constant, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )
    
    print("  2. Heterogeneous μ (ground truth)...")
    with torch.no_grad():
        u_pred_hetero, _ = model.solve_given_mu(
            x, mu_heterogeneous, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )

    # Now sweep over α/β values
    print(f"\nSweeping α and β weights...")
    results = []
    
    for i, (alpha, beta) in enumerate(zip(alpha_values, beta_values)):
        print(f"\n  [{i+1}/{len(alpha_values)}] α={alpha:.1f}, β={beta:.1f}")
        
        # Compute Sobolev loss for constant mu
        sob_const, l2_const, grad_const = compute_sobolev_loss(
            u_pred_const, u_meas, x, alpha, beta
        )
        print(f"    Const: Sobolev={sob_const:.6e}, L2={l2_const:.6e}, Grad={grad_const:.6e}")
        
        # Compute Sobolev loss for heterogeneous mu
        sob_hetero, l2_hetero, grad_hetero = compute_sobolev_loss(
            u_pred_hetero, u_meas, x, alpha, beta
        )
        print(f"    Hetero: Sobolev={sob_hetero:.6e}, L2={l2_hetero:.6e}, Grad={grad_hetero:.6e}")
        
        # Discrimination: Hetero - Const (more negative = better)
        delta = sob_hetero - sob_const
        abs_delta = abs(delta)
        rel_delta = abs_delta / (abs(sob_const) + 1e-12)
        
        print(f"    Δ(Hetero-Const) = {delta:.6e} (RelDiff={rel_delta:.4f})")
        
        results.append({
            'alpha': alpha,
            'beta': beta,
            'const_sobolev': sob_const,
            'const_l2': l2_const,
            'const_grad': grad_const,
            'hetero_sobolev': sob_hetero,
            'hetero_l2': l2_hetero,
            'hetero_grad': grad_hetero,
            'delta': delta,
            'abs_delta': abs_delta,
            'rel_delta': rel_delta
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'sobolev_weight_sweep_results.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir / 'sobolev_weight_sweep_results.csv'}")

    # Find optimal weights
    best_idx = df['abs_delta'].idxmax()
    best_alpha = df.loc[best_idx, 'alpha']
    best_beta = df.loc[best_idx, 'beta']
    best_delta = df.loc[best_idx, 'delta']
    
    print("\n" + "="*80)
    print("OPTIMAL WEIGHTS")
    print("="*80)
    print(f"Best α = {best_alpha:.1f}, β = {best_beta:.1f}")
    print(f"Maximum discrimination: Δ = {best_delta:.6e}")
    print("="*80)

    # Create visualization
    create_visualization(df, best_alpha, best_beta, output_dir)
    
    print(f"\n✓ Visualization saved to: {output_dir}")


def create_visualization(df: pd.DataFrame, best_alpha: float, best_beta: float, output_dir: Path):
    """Create comprehensive visualization of weight sweep results."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Row 1: Sobolev loss curves
    ax = fig.add_subplot(gs[0, :])
    ax.plot(df['alpha'], df['const_sobolev'], 'o-', linewidth=2, markersize=8, 
            label='Const μ=5000', color='skyblue')
    ax.plot(df['alpha'], df['hetero_sobolev'], 's-', linewidth=2, markersize=8,
            label='Hetero μ', color='coral')
    ax.axvline(best_alpha, color='green', linestyle='--', linewidth=2, 
               label=f'Optimal α={best_alpha:.1f}')
    ax.set_xlabel('α (L2 weight)', fontsize=12)
    ax.set_ylabel('Sobolev Loss', fontsize=12)
    ax.set_title('Sobolev Loss vs α Weight\n(β = 1 - α)', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add secondary x-axis for β
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df['alpha'])
    ax2.set_xticklabels([f'{b:.1f}' for b in df['beta']])
    ax2.set_xlabel('β (Gradient weight)', fontsize=12)
    
    # Row 2, Col 1: Delta (discrimination)
    ax = fig.add_subplot(gs[1, 0])
    colors = ['green' if d < 0 else 'red' for d in df['delta']]
    ax.bar(df['alpha'], df['delta'], color=colors, alpha=0.7, edgecolor='black', width=0.08)
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(best_alpha, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('Δ (Hetero - Const)', fontsize=11)
    ax.set_title('Discrimination\n(More negative = Better)', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate best
    best_row = df[df['alpha'] == best_alpha].iloc[0]
    ax.annotate(f"Best: α={best_alpha:.1f}\nΔ={best_row['delta']:.2e}",
                xy=(best_alpha, best_row['delta']),
                xytext=(best_alpha + 0.15, best_row['delta']),
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Row 2, Col 2: Absolute delta
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df['alpha'], df['abs_delta'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.axvline(best_alpha, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('|Δ| (Absolute Discrimination)', fontsize=11)
    ax.set_title('Absolute Discrimination Power', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Row 2, Col 3: Relative delta
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(df['alpha'], df['rel_delta'], 'o-', linewidth=2, markersize=8, color='orange')
    ax.axvline(best_alpha, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('Relative Difference', fontsize=11)
    ax.set_title('Relative Discrimination\n|Δ| / Const', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Row 3: Component breakdown (L2 and Gradient separately)
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df['alpha'], df['const_l2'], 'o-', linewidth=2, markersize=6, 
            label='Const L2', color='skyblue')
    ax.plot(df['alpha'], df['hetero_l2'], 's-', linewidth=2, markersize=6,
            label='Hetero L2', color='coral')
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('L2 Loss Component', fontsize=11)
    ax.set_title('L2 Term (MSE)', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(df['alpha'], df['const_grad'], 'o-', linewidth=2, markersize=6,
            label='Const Grad', color='skyblue')
    ax.plot(df['alpha'], df['hetero_grad'], 's-', linewidth=2, markersize=6,
            label='Hetero Grad', color='coral')
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('Gradient Loss Component', fontsize=11)
    ax.set_title('Gradient Term (||∇u||²)', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Row 3, Col 3: Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Create summary text
    summary_text = "OPTIMAL CONFIGURATION\n" + "="*30 + "\n\n"
    summary_text += f"α (L2 weight):    {best_alpha:.1f}\n"
    summary_text += f"β (Grad weight):  {best_beta:.1f}\n\n"
    summary_text += "="*30 + "\n\n"
    summary_text += f"Const Sobolev:  {best_row['const_sobolev']:.4e}\n"
    summary_text += f"Hetero Sobolev: {best_row['hetero_sobolev']:.4e}\n\n"
    summary_text += f"Discrimination: {best_row['delta']:.4e}\n"
    summary_text += f"Abs Δ:          {best_row['abs_delta']:.4e}\n"
    summary_text += f"Rel Δ:          {best_row['rel_delta']:.4f}\n\n"
    summary_text += "="*30 + "\n\n"
    
    if best_alpha == 0.5:
        summary_text += "✓ Balanced L2/Gradient\n"
    elif best_alpha > 0.5:
        summary_text += "→ L2 term dominant\n"
    else:
        summary_text += "→ Gradient term dominant\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Sobolev Loss Weight Optimization: α + β = 1\n' +
                 'Finding Optimal Balance Between L2 and Gradient Terms',
                 fontsize=16, weight='bold', y=0.995)
    
    plt.savefig(output_dir / 'sobolev_weight_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: sobolev_weight_sweep.png")


if __name__ == "__main__":
    run_weight_sweep()
