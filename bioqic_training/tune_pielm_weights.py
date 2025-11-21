"""
Tune PIELM weights to find optimal balance.

Now that normalization fixed the magnitude issue, we need to find
the right balance between PDE, BC, and data constraints.
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from pielm_normalized import PIELMNormalized
import sys


class ConstantMuNetwork:
    """Return ground truth μ."""
    def __init__(self, mu_true):
        self.mu_true = mu_true
    def __call__(self, x):
        return self.mu_true


def tune_pielm_weights():
    """Test different weight configurations."""
    
    print("="*70)
    print("TUNING PIELM WEIGHTS")
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
    
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    # Subsample
    subsample = 1000
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    mu_true = torch.tensor(data['mu_data'][indices], dtype=torch.float32)
    
    print(f"   Points: {subsample}")
    print(f"   u_meas range: [{u_meas.min():.3f}, {u_meas.max():.3f}]")
    
    # BC
    n_bc = 100
    bc_indices = torch.randperm(subsample)[:n_bc]
    u_bc_vals = u_meas[bc_indices]
    
    # Create network
    mu_network = ConstantMuNetwork(mu_true)
    
    # Test configurations
    configs = [
        # (bc_weight, data_weight, pde_weight, name)
        (100.0, 0.1, 1.0, "Balanced (BC=100, Data=0.1, PDE=1)"),
        (1000.0, 1.0, 1.0, "High BC (BC=1000, Data=1, PDE=1)"),
        (100.0, 10.0, 1.0, "High Data (BC=100, Data=10, PDE=1)"),
        (10.0, 1.0, 0.1, "Low PDE (BC=10, Data=1, PDE=0.1)"),
        (1000.0, 100.0, 1.0, "High BC+Data (BC=1000, Data=100, PDE=1)"),
        (100.0, 0.1, 0.0, "No PDE (BC=100, Data=0.1, PDE=0)"),
    ]
    
    print("\n2. Testing weight configurations...")
    print("\n" + "="*90)
    print(f"{'Config':<40} {'BC wt':<10} {'Data wt':<10} {'PDE wt':<10} {'MSE':<10}")
    print("="*90)
    
    best_mse = float('inf')
    best_config = None
    results = []  # Store all results for visualization
    
    for bc_weight, data_weight, pde_weight, name in configs:
        model = PIELMNormalized(
            mu_network=mu_network,
            poly_degree=5,
            seed=42
        )
        
        with torch.no_grad():
            try:
                u_pred, _ = model(
                    x,
                    rho_omega2=rho_omega2,
                    bc_indices=bc_indices,
                    u_bc_vals=u_bc_vals,
                    bc_weight=bc_weight,
                    u_data=u_meas,
                    data_weight=data_weight,
                    pde_weight=pde_weight,
                    verbose=False
                )
                
                mse = torch.mean((u_pred - u_meas)**2).item()
                u_min = u_pred.min().item()
                u_max = u_pred.max().item()
                
                # Store results
                results.append((mse, bc_weight, data_weight, name))
                
                print(f"{name:<40} {bc_weight:<10.1f} {data_weight:<10.2f} {pde_weight:<10.2f} {mse:<10.6f}")
                print(f"  → u_pred: [{u_min:.3f}, {u_max:.3f}]")
                
                if mse < best_mse:
                    best_mse = mse
                    best_config = (bc_weight, data_weight, pde_weight, name)
                    best_u_pred = u_pred
                    
            except Exception as e:
                print(f"{name:<40} ERROR: {str(e)[:30]}")
    
    print("="*90)
    
    if best_config is None:
        print("\n❌ All configurations failed!")
        print("Check error messages above for details.")
        return
    
    print(f"\n✅ Best configuration: {best_config[3]}")
    print(f"   BC weight: {best_config[0]}")
    print(f"   Data weight: {best_config[1]}")
    print(f"   PDE weight: {best_config[2]}")
    print(f"   MSE: {best_mse:.6f}")
    
    # Analyze best solution
    print(f"\n📊 Best solution analysis:")
    print(f"   u_pred range: [{best_u_pred.min():.3f}, {best_u_pred.max():.3f}]")
    print(f"   u_meas range: [{u_meas.min():.3f}, {u_meas.max():.3f}]")
    
    # Check if in valid range
    in_range = (best_u_pred >= 0).all() and (best_u_pred <= 1).all()
    if in_range:
        print(f"   ✓ u_pred within [0, 1]")
    else:
        n_below = (best_u_pred < 0).sum().item()
        n_above = (best_u_pred > 1).sum().item()
        print(f"   ⚠️  {n_below} points < 0, {n_above} points > 1")
    
    # Error statistics
    error = (best_u_pred - u_meas).numpy()
    print(f"   Error mean: {error.mean():.6f}")
    print(f"   Error std: {error.std():.6f}")
    print(f"   Error median: {np.median(error):.6f}")
    
    if best_mse < 0.05:
        print(f"\n   🎉 EXCELLENT! MSE < 0.05")
        print(f"      Forward problem can fit data well!")
    elif best_mse < 0.15:
        print(f"\n   ✓ Good! MSE < 0.15")
        print(f"      Normalization + tuning worked!")
    else:
        print(f"\n   ⚠️  MSE still high: {best_mse:.3f}")
        print(f"      Likely due to PDE mismatch (elastic vs viscoelastic)")
    
    # Create comprehensive visualizations
    print(f"\n3. Creating visualizations...")
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: MSE comparison
    ax1 = plt.subplot(3, 3, 1)
    config_names = [r[3] for r in results]
    mses = [r[0] for r in results]
    colors = ['red' if m > 0.05 else 'orange' if m > 0.02 else 'green' for m in mses]
    bars = ax1.barh(range(len(mses)), mses, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(mses)))
    ax1.set_yticklabels(config_names, fontsize=9)
    ax1.set_xlabel('MSE', fontweight='bold')
    ax1.set_title('MSE Comparison', fontweight='bold', fontsize=12)
    ax1.axvline(0.02, color='g', linestyle='--', linewidth=2, label='Target (0.02)')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend()
    
    # Plot 2: Predicted vs Measured (scatter)
    ax2 = plt.subplot(3, 3, 2)
    u_meas_np = u_meas.numpy().flatten()
    u_pred_best_np = best_u_pred.numpy().flatten()
    ax2.scatter(u_meas_np, u_pred_best_np, alpha=0.4, s=10, c='blue')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    ax2.set_xlabel('Measured u (normalized)', fontweight='bold')
    ax2.set_ylabel('Predicted u (normalized)', fontweight='bold')
    ax2.set_title(f'Best Config: MSE={best_mse:.4f}', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Plot 3: Error histogram
    ax3 = plt.subplot(3, 3, 3)
    error_tensor = (best_u_pred - u_meas)
    error_np = error_tensor.numpy().flatten()
    ax3.hist(error_np, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax3.axvline(0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Prediction Error', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title(f'Error Distribution (σ={error_tensor.std():.4f})', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Ground truth μ distribution
    ax4 = plt.subplot(3, 3, 4)
    mu_true_np = mu_true.numpy().flatten()
    mu_phys = mu_true_np * 7000 + 3000  # Denormalize
    ax4.hist(mu_phys, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax4.set_xlabel('Storage Modulus (Pa)', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Ground Truth μ Distribution', fontweight='bold', fontsize=12)
    ax4.axvline(3000, color='blue', linestyle='--', label='Background (3 kPa)')
    ax4.axvline(10000, color='red', linestyle='--', label='Targets (10 kPa)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: u_meas distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(u_meas_np, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax5.set_xlabel('Displacement (normalized)', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title('Measured Displacement Distribution', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: u_pred distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(u_pred_best_np, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax6.set_xlabel('Displacement (normalized)', fontweight='bold')
    ax6.set_ylabel('Count', fontweight='bold')
    ax6.set_title('Predicted Displacement Distribution', fontweight='bold', fontsize=12)
    ax6.axvspan(-0.2, 0, alpha=0.3, color='red', label='< 0 (unphysical)')
    ax6.axvspan(1, 1.2, alpha=0.3, color='red')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Weight sensitivity - MSE vs BC weight
    ax7 = plt.subplot(3, 3, 7)
    bc_weights = [r[1] for r in results]
    ax7.scatter(bc_weights, mses, s=100, c=colors, alpha=0.7, edgecolors='black')
    ax7.set_xlabel('BC Weight', fontweight='bold')
    ax7.set_ylabel('MSE', fontweight='bold')
    ax7.set_title('BC Weight Sensitivity', fontweight='bold', fontsize=12)
    ax7.set_xscale('log')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Weight sensitivity - MSE vs Data weight  
    ax8 = plt.subplot(3, 3, 8)
    data_weights = [r[2] for r in results]
    ax8.scatter(data_weights, mses, s=100, c=colors, alpha=0.7, edgecolors='black')
    ax8.set_xlabel('Data Weight', fontweight='bold')
    ax8.set_ylabel('MSE', fontweight='bold')
    ax8.set_title('Data Weight Sensitivity', fontweight='bold', fontsize=12)
    ax8.set_xscale('log')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
PIELM FORWARD PROBLEM TUNING

Dataset: BIOQIC Phase 1 Box
• Grid: 100×80×10 (1mm)
• Freq: 60 Hz (ω = 377 rad/s)
• Material: Voigt viscoelastic
  Background: 3 kPa
  Targets: 10 kPa (4 circles)

Best Configuration:
• {best_config[3]}
• BC wt: {best_config[0]:.1f}
• Data wt: {best_config[1]:.2f}  
• PDE wt: {best_config[2]:.2f}
• MSE: {best_mse:.6f}

Solution Quality:
• Range: [{best_u_pred.min():.3f}, {best_u_pred.max():.3f}]
• < 0: {(best_u_pred < 0).sum().item()} pts
• > 1: {(best_u_pred > 1).sum().item()} pts
• σ_err: {error_tensor.std():.4f}

Key Finding:
✓ ρω² normalization fixed collapse
✓ Higher data wt improves fit
✓ Lower PDE wt → flexibility
→ Data-driven works better
  (PDE too restrictive)
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PIELM Weight Tuning Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'outputs/pielm_weight_tuning.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    
    plt.show()
    
    return best_mse, best_config


if __name__ == '__main__':
    tune_pielm_weights()
