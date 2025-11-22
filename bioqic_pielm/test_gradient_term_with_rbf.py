"""
Test gradient term effectiveness using RBF-based gradients.

This will show if better gradient estimation (RBF vs finite diff) improves the
gradient term's contribution to the forward problem accuracy.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RBFInterpolator
import pandas as pd

from data_loader import BIOQICDataLoader


class ForwardMREModelWithGradient:
    """Simplified forward model for testing gradient term with RBF gradients."""

    def __init__(self, n_neurons, input_dim, omega_basis, seed=42, basis_type='sin'):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.omega_basis = omega_basis
        self.basis_type = basis_type

        # Random basis weights and biases
        self.B = torch.randn(input_dim, n_neurons) * omega_basis
        self.phi_bias = torch.randn(1, n_neurons) * omega_basis

    def get_basis_and_derivatives(self, x, include_gradient=False):
        """Compute basis functions and derivatives analytically."""
        N = x.shape[0]

        # Compute basis: sin(w·x + b)
        Z = x @ self.B + self.phi_bias  # (N, M)
        phi = torch.sin(Z)  # (N, M)

        # Analytical Laplacian: ∇²φ = -||w||² * φ
        freq_sq = torch.sum(self.B ** 2, dim=0, keepdim=True)  # (1, M)
        phi_lap = -phi * freq_sq  # (N, M)

        if include_gradient:
            # Analytical gradient: ∇φ = cos(w·x + b) * w
            cos_Z = torch.cos(Z)  # (N, M)
            phi_grad = cos_Z.unsqueeze(2) * self.B.T.unsqueeze(0)  # (N, M, dim)
            return phi, phi_lap, phi_grad
        else:
            return phi, phi_lap, None

    def compute_gradient_rbf(self, coords_np, mu_np, rbf_neighbors=1000):
        """Compute ∇μ using RBF interpolation."""
        # Subsample for RBF fitting
        if len(coords_np) > rbf_neighbors:
            idx = np.random.choice(len(coords_np), rbf_neighbors, replace=False)
            coords_fit = coords_np[idx]
            mu_fit = mu_np[idx]
        else:
            coords_fit = coords_np
            mu_fit = mu_np

        # Fit RBF
        rbf = RBFInterpolator(coords_fit, mu_fit, kernel='thin_plate_spline', epsilon=1.0)

        # Compute gradients using finite differences on RBF
        h = 1e-4  # 0.1mm
        grad_mu = np.zeros((len(coords_np), 3))

        for d in range(3):
            coords_plus = coords_np.copy()
            coords_minus = coords_np.copy()
            coords_plus[:, d] += h
            coords_minus[:, d] -= h

            mu_plus = rbf(coords_plus)
            mu_minus = rbf(coords_minus)

            grad_mu[:, d] = (mu_plus - mu_minus) / (2 * h)

        return grad_mu

    def build_system(self, x, mu_field, phi, phi_lap, phi_grad, rho_omega2,
                     bc_indices, u_bc_vals, bc_weight, grad_mu=None):
        """Build least-squares system with optional pre-computed gradient."""

        # PDE rows
        if grad_mu is not None and phi_grad is not None:
            # Full form with gradient term: (μ·∇²φ + ∇μ·∇φ + ρω²·φ) / ρω² = 0
            # grad_mu: (N, dim), phi_grad: (N, M, dim)
            grad_term = torch.sum(grad_mu.unsqueeze(1) * phi_grad, dim=2)  # (N, M)
            H_pde = (mu_field / rho_omega2) * phi_lap + (grad_term / rho_omega2) + phi
        else:
            # Simplified form: (μ·∇²φ + ρω²·φ) / ρω² = 0
            H_pde = (mu_field / rho_omega2) * phi_lap + phi

        b_pde = torch.zeros(phi.shape[0], 1, device=phi.device)

        # BC rows
        H_bc = phi[bc_indices, :] * bc_weight
        b_bc = u_bc_vals.reshape(-1, 1) * bc_weight

        # Stack
        H = torch.vstack([H_pde, H_bc])
        b = torch.vstack([b_pde, b_bc])

        return H, b

    def solve(self, x, mu_field, bc_indices, u_bc_vals, rho_omega2, bc_weight,
              grad_mu=None):
        """Solve forward problem with optional pre-computed gradient."""

        include_gradient = grad_mu is not None

        # Get basis and derivatives
        phi, phi_lap, phi_grad = self.get_basis_and_derivatives(x, include_gradient=include_gradient)

        # Build system
        H, b = self.build_system(x, mu_field, phi, phi_lap, phi_grad, rho_omega2,
                                 bc_indices, u_bc_vals, bc_weight, grad_mu)

        # Solve
        C = torch.linalg.lstsq(H, b).solution

        # Predict
        u_pred = phi @ C

        return u_pred


def compute_region_metrics(u_pred, u_meas, mu_field, blob_threshold=8000.0):
    """Compute R² for blob and background regions."""
    mu_np = mu_field.cpu().numpy().flatten() if torch.is_tensor(mu_field) else mu_field.flatten()

    is_blob = mu_np > blob_threshold
    is_background = ~is_blob

    results = {}

    if is_blob.sum() > 0:
        error_blob = u_pred[is_blob] - u_meas[is_blob]
        mse_blob = torch.mean(error_blob ** 2).item()
        var_blob = torch.var(u_meas[is_blob]).item()
        r2_blob = 1 - mse_blob / var_blob if var_blob > 0 else 0
        results['blob'] = {'r2': r2_blob, 'mse': mse_blob}
    else:
        results['blob'] = None

    if is_background.sum() > 0:
        error_bg = u_pred[is_background] - u_meas[is_background]
        mse_bg = torch.mean(error_bg ** 2).item()
        var_bg = torch.var(u_meas[is_background]).item()
        r2_bg = 1 - mse_bg / var_bg if var_bg > 0 else 0
        results['background'] = {'r2': r2_bg, 'mse': mse_bg}
    else:
        results['background'] = None

    return results


def main():
    print("="*80)
    print("GRADIENT TERM TEST WITH RBF GRADIENTS")
    print("="*80)

    device = torch.device('cpu')

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'rbf_gradient_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    bc_weight = 10
    n_points = 5000
    neurons_list = [1000, 5000, 10000]
    omega_basis = 170.0

    print(f"\nConfiguration:")
    print(f"  Sampling: uniform (5,000 points)")
    print(f"  BC weight: {bc_weight}")
    print(f"  Neurons: {neurons_list}")
    print(f"  Omega basis: {omega_basis}")

    # Physics parameters
    freq = 60
    omega = 2 * np.pi * freq
    rho = 1000
    rho_omega2 = rho * omega ** 2

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    loader = BIOQICDataLoader(
        data_dir=str(data_dir),
        displacement_mode='z_component',
        subsample=n_points,
        seed=42,
        adaptive_sampling=False
    )
    data = loader.load()

    coords = data['coords']
    u_raw = data['u_raw']
    mu_raw = data['mu_raw']

    x = torch.from_numpy(coords).float().to(device)
    u_meas = torch.from_numpy(u_raw).float().to(device)
    mu = torch.from_numpy(mu_raw).float().to(device)

    print(f"  Points: {len(x)}")
    print(f"  μ range: [{mu.min():.0f}, {mu.max():.0f}] Pa")

    # Compute RBF gradients ONCE (expensive operation)
    print("\n" + "="*80)
    print("COMPUTING RBF GRADIENTS (ONE-TIME)")
    print("="*80)

    model_temp = ForwardMREModelWithGradient(
        n_neurons=1000, input_dim=3, omega_basis=omega_basis, seed=42, basis_type='sin'
    )

    print("  Fitting RBF to μ(x) field...")
    grad_mu_rbf_np = model_temp.compute_gradient_rbf(coords, mu_raw.flatten(), rbf_neighbors=1000)
    grad_mu_rbf = torch.from_numpy(grad_mu_rbf_np).float().to(device)

    grad_mag = np.linalg.norm(grad_mu_rbf_np, axis=1)
    print(f"  RBF gradient statistics:")
    print(f"    Mean |∇μ|: {grad_mag.mean():.1f} Pa/m")
    print(f"    Median |∇μ|: {np.median(grad_mag):.1f} Pa/m")
    print(f"    % Zero: {(grad_mag < 1e-6).sum() / len(grad_mag) * 100:.1f}%")

    # Boundary conditions
    print("\n" + "="*80)
    print("SETTING UP BOUNDARY CONDITIONS")
    print("="*80)

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

    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")

    # Run tests
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)

    results = []

    for n_neurons in neurons_list:
        print(f"\n{'='*80}")
        print(f"NEURONS = {n_neurons}")
        print(f"{'='*80}")

        # Create model
        model = ForwardMREModelWithGradient(
            n_neurons=n_neurons,
            input_dim=3,
            omega_basis=omega_basis,
            seed=42,
            basis_type='sin'
        )

        # Test 1: Without gradient term (baseline)
        print(f"\n[1/2] Without gradient term...")
        try:
            u_pred_no_grad = model.solve(
                x, mu, bc_indices, u_bc_vals, rho_omega2, bc_weight,
                grad_mu=None
            )

            error = u_pred_no_grad - u_meas
            mse = torch.mean(error ** 2).item()
            var_u = torch.var(u_meas).item()
            r2 = 1 - mse / var_u

            region_metrics = compute_region_metrics(u_pred_no_grad, u_meas, mu)
            blob_r2_no = region_metrics['blob']['r2'] if region_metrics['blob'] else None

            print(f"  Overall R²: {r2:.4f}")
            print(f"  Blob R²: {blob_r2_no:.4f}" if blob_r2_no else "  Blob R²: N/A")

            results.append({
                'neurons': n_neurons,
                'gradient_method': 'None',
                'r2': r2,
                'blob_r2': blob_r2_no,
                'mse': mse,
                'background_r2': region_metrics['background']['r2'] if region_metrics['background'] else None
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Test 2: With RBF gradient term
        print(f"\n[2/2] With RBF gradient term...")
        try:
            u_pred_rbf = model.solve(
                x, mu, bc_indices, u_bc_vals, rho_omega2, bc_weight,
                grad_mu=grad_mu_rbf
            )

            error = u_pred_rbf - u_meas
            mse = torch.mean(error ** 2).item()
            var_u = torch.var(u_meas).item()
            r2 = 1 - mse / var_u

            region_metrics = compute_region_metrics(u_pred_rbf, u_meas, mu)
            blob_r2_rbf = region_metrics['blob']['r2'] if region_metrics['blob'] else None

            print(f"  Overall R²: {r2:.4f}")
            print(f"  Blob R²: {blob_r2_rbf:.4f}" if blob_r2_rbf else "  Blob R²: N/A")

            if blob_r2_no and blob_r2_rbf:
                improvement = blob_r2_rbf - blob_r2_no
                print(f"  Blob R² change: {improvement:+.4f}")

            results.append({
                'neurons': n_neurons,
                'gradient_method': 'RBF',
                'r2': r2,
                'blob_r2': blob_r2_rbf,
                'mse': mse,
                'background_r2': region_metrics['background']['r2'] if region_metrics['background'] else None
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'rbf_gradient_comparison.csv', index=False)
    print(f"  Saved: rbf_gradient_comparison.csv")

    # Create visualizations
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Blob R² comparison
    ax = axes[0]
    neurons = neurons_list
    x_pos = np.arange(len(neurons))
    width = 0.35

    no_grad_scores = [results_df[(results_df['neurons'] == n) & (results_df['gradient_method'] == 'None')]['blob_r2'].values[0]
                     for n in neurons]
    rbf_grad_scores = [results_df[(results_df['neurons'] == n) & (results_df['gradient_method'] == 'RBF')]['blob_r2'].values[0]
                      for n in neurons]

    ax.bar(x_pos - width/2, no_grad_scores, width, label='No Gradient Term', color='lightcoral')
    ax.bar(x_pos + width/2, rbf_grad_scores, width, label='RBF Gradient Term', color='green', alpha=0.7)

    ax.set_xlabel('Number of Neurons', fontsize=12)
    ax.set_ylabel('Blob R²', fontsize=12)
    ax.set_title('RBF Gradient Term Effect on Blob R²', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(neurons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 0.9])

    # Plot 2: Improvement
    ax = axes[1]
    improvements = [rbf - no for rbf, no in zip(rbf_grad_scores, no_grad_scores)]
    colors = ['green' if x > 0 else 'red' for x in improvements]

    ax.bar(x_pos, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0.01, color='green', linestyle=':', linewidth=1, label='Threshold (+1%)')
    ax.axhline(y=-0.01, color='red', linestyle=':', linewidth=1, label='Threshold (-1%)')

    ax.set_xlabel('Number of Neurons', fontsize=12)
    ax.set_ylabel('Blob R² Change', fontsize=12)
    ax.set_title('RBF Gradient Term Impact', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(neurons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rbf_gradient_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rbf_gradient_improvement.png")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    avg_improvement = np.mean(improvements)
    print(f"\nAverage Blob R² improvement with RBF gradients: {avg_improvement:+.4f}")

    if avg_improvement > 0.01:
        print("\n[+] RBF GRADIENT TERM HELPS!")
        print("    Recommendation: Include gradient term with RBF interpolation")
    elif avg_improvement < -0.01:
        print("\n[-] RBF GRADIENT TERM HURTS")
        print("    Recommendation: Stick with simplified form")
    else:
        print("\n[=] RBF GRADIENT TERM HAS NEGLIGIBLE EFFECT")
        print("    Recommendation: Current approximation is sufficient")
        print("    (RBF overhead not justified)")

    print(f"\nAll results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
