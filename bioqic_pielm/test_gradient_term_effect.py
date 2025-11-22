"""
Test Gradient Term Effect on Constant vs Heterogeneous Mu
==========================================================

Tests whether including the gradient term (∇μ·∇φ) in the weak form helps
discriminate between constant and heterogeneous stiffness fields.

Current implementation: ∇·(μ∇u) = μ·∇²u (assumes constant μ, ∇μ = 0)
Full form: ∇·(μ∇u) = μ·∇²u + ∇μ·∇u

For constant μ: ∇μ = 0, so both forms are equivalent.
For heterogeneous μ: ∇μ ≠ 0, gradient term should matter.

Based on best configuration from grid search:
- Uniform sampling (not adaptive)
- 10000 neurons
- 5000 sampling points
- BC weight = 10
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import BIOQICDataLoader


class ForwardMREModelWithGradient:
    """Forward model with option to include/exclude gradient term."""

    def __init__(self, n_neurons, input_dim, omega_basis, seed=42, basis_type='sin'):
        torch.manual_seed(seed)
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.omega_basis = omega_basis
        self.basis_type = basis_type

        # Random basis weights and biases (frozen)
        self.B = torch.randn(input_dim, n_neurons) * omega_basis
        self.phi_bias = torch.randn(1, n_neurons) * omega_basis

    def get_basis_and_derivatives(self, x, include_gradient=False):
        """Compute basis functions, Laplacians, and optionally gradients using analytical derivatives.

        For sin basis:
            φ(x) = sin(w·x + b)
            ∇φ = cos(w·x + b) * w
            ∇²φ = -||w||² * φ
        """
        N = x.shape[0]

        # Compute basis: sin(w·x + b)
        Z = x @ self.B + self.phi_bias  # (N, M)
        phi = torch.sin(Z)  # (N, M)

        # Analytical Laplacian: ∇²φ = -||w||² * φ
        freq_sq = torch.sum(self.B ** 2, dim=0, keepdim=True)  # (1, M)
        phi_lap = -phi * freq_sq  # (N, M)

        if include_gradient:
            # Analytical gradient: ∇φ = cos(w·x + b) * w
            # For each basis function j: ∇φ_j = cos(Z_j) * w_j
            cos_Z = torch.cos(Z)  # (N, M)

            # phi_grad[i, j, d] = ∂φ_j/∂x_d at point i = cos(Z[i,j]) * B[d,j]
            # Shape: (N, M, dim)
            phi_grad = cos_Z.unsqueeze(2) * self.B.T.unsqueeze(0)  # (N, M, dim)

            return phi, phi_lap, phi_grad
        else:
            return phi, phi_lap, None

    def compute_mu_gradient(self, x, mu_field):
        """Compute ∇μ using finite differences.

        For each point, find nearest neighbors along each axis and compute gradient.
        This is a simple implementation using vectorized operations.
        """
        N = x.shape[0]
        grad_mu = torch.zeros(N, self.input_dim, device=x.device, dtype=mu_field.dtype)

        # Small perturbation for finite differences
        h = 1e-3  # 1mm in SI units

        # For each spatial dimension
        for d in range(self.input_dim):
            # For each point, find the nearest neighbor in the +/- direction
            # and estimate gradient

            # Create shifted coordinates
            for i in range(N):
                # Current point
                xi = x[i:i+1, :]

                # Find nearest points in + and - directions along dimension d
                # Points in + direction: x_d > xi_d
                mask_plus = x[:, d] > xi[0, d]
                # Points in - direction: x_d < xi_d
                mask_minus = x[:, d] < xi[0, d]

                if mask_plus.sum() > 0 and mask_minus.sum() > 0:
                    # Find nearest in each direction
                    dist_plus = torch.abs(x[mask_plus, d] - xi[0, d])
                    dist_minus = torch.abs(x[mask_minus, d] - xi[0, d])

                    idx_plus = torch.where(mask_plus)[0][torch.argmin(dist_plus)]
                    idx_minus = torch.where(mask_minus)[0][torch.argmin(dist_minus)]

                    # Finite difference
                    dx = x[idx_plus, d] - x[idx_minus, d]
                    if dx > 1e-8:
                        grad_mu[i, d] = (mu_field[idx_plus, 0] - mu_field[idx_minus, 0]) / dx
                elif mask_plus.sum() > 0:
                    # Forward difference
                    dist_plus = torch.abs(x[mask_plus, d] - xi[0, d])
                    idx_plus = torch.where(mask_plus)[0][torch.argmin(dist_plus)]
                    dx = x[idx_plus, d] - xi[0, d]
                    if dx > 1e-8:
                        grad_mu[i, d] = (mu_field[idx_plus, 0] - mu_field[i, 0]) / dx
                elif mask_minus.sum() > 0:
                    # Backward difference
                    dist_minus = torch.abs(x[mask_minus, d] - xi[0, d])
                    idx_minus = torch.where(mask_minus)[0][torch.argmin(dist_minus)]
                    dx = xi[0, d] - x[idx_minus, d]
                    if dx > 1e-8:
                        grad_mu[i, d] = (mu_field[i, 0] - mu_field[idx_minus, 0]) / dx

        return grad_mu  # (N, dim)

    def build_system(self, x, mu_field, phi, phi_lap, phi_grad, rho_omega2,
                     bc_indices, u_bc_vals, bc_weight, include_grad_term=False):
        """Build least-squares system with or without gradient term."""

        # PDE rows
        if include_grad_term and phi_grad is not None:
            # Compute ∇μ
            print("  Computing mu gradient...")
            grad_mu = self.compute_mu_gradient(x, mu_field)  # (N, dim)

            # Compute ∇μ·∇φ for all basis functions
            # phi_grad: (N, M, dim), grad_mu: (N, dim)
            # grad_term: (N, M) = sum over dim of grad_mu * phi_grad
            grad_term = torch.sum(grad_mu.unsqueeze(1) * phi_grad, dim=2)  # (N, M)

            # Full form: (μ·∇²φ + ∇μ·∇φ + ρω²·φ) / ρω² = 0
            H_pde = (mu_field / rho_omega2) * phi_lap + (grad_term / rho_omega2) + phi
        else:
            # Simplified form: (μ·∇²φ + ρω²·φ) / ρω² = 0
            H_pde = (mu_field / rho_omega2) * phi_lap + phi

        b_pde = torch.zeros(phi.shape[0], 1, device=phi.device)

        # BC rows
        H_bc = phi[bc_indices, :] * bc_weight
        b_bc = u_bc_vals * bc_weight

        H = torch.cat([H_pde, H_bc], dim=0)
        b = torch.cat([b_pde, b_bc], dim=0)

        return H, b

    def solve(self, x, mu_field, bc_indices, u_bc_vals, rho_omega2, bc_weight,
              include_grad_term=False):
        """Solve forward problem with or without gradient term."""

        # Move to same device as x
        self.B = self.B.to(x.device)
        self.phi_bias = self.phi_bias.to(x.device)

        # Get basis and derivatives
        phi, phi_lap, phi_grad = self.get_basis_and_derivatives(x, include_gradient=include_grad_term)

        # Build system
        H, b = self.build_system(x, mu_field, phi, phi_lap, phi_grad, rho_omega2,
                                 bc_indices, u_bc_vals, bc_weight, include_grad_term)

        # Solve least-squares using torch.linalg.lstsq
        C = torch.linalg.lstsq(H, b).solution

        # Predict displacement
        u_pred = phi @ C

        return u_pred


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


def main():
    print("\n" + "="*80)
    print("GRADIENT TERM EFFECT TEST")
    print("="*80)
    print("\nComparing two formulations:")
    print("  1. Simplified: μ·∇²u + ρω²·u = 0 (assumes ∇μ = 0)")
    print("  2. Full form:  μ·∇²u + ∇μ·∇u + ρω²·u = 0 (includes gradient term)")
    print("\nHypothesis: For heterogeneous μ, full form should be more accurate.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'gradient_term_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    bc_weight = 10
    n_points = 5000
    neurons_list = [1000, 5000, 10000]  # Test multiple neuron counts
    omega_basis = 170.0
    use_adaptive = False  # Uniform sampling was best

    print(f"\nConfiguration:")
    print(f"  Sampling: uniform")
    print(f"  BC weight: {bc_weight}")
    print(f"  Sampling points: {n_points}")
    print(f"  Neurons: {neurons_list}")
    print(f"  Omega basis: {omega_basis}")

    # Physics parameters
    freq = 60
    omega = 2 * np.pi * freq
    rho = 1000
    rho_omega2 = rho * omega ** 2

    # Load data
    print(f"\nLoading data from: {data_dir}")
    loader = BIOQICDataLoader(
        data_dir=str(data_dir),
        displacement_mode='z_component',
        subsample=n_points,
        seed=42,
        adaptive_sampling=use_adaptive
    )
    data = loader.load()

    coords = data['coords']
    u_raw = data['u_raw']
    mu_raw = data['mu_raw']

    x = torch.from_numpy(coords).float().to(device)
    u_meas = torch.from_numpy(u_raw).float().to(device)
    mu_heterogeneous = torch.from_numpy(mu_raw).float().to(device)
    mu_constant = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)

    print(f"\nData loaded:")
    print(f"  Points: {len(x)}")
    print(f"  Constant μ: {mu_constant[0, 0].item():.0f} Pa")
    print(f"  Heterogeneous μ: [{mu_heterogeneous.min():.0f}, {mu_heterogeneous.max():.0f}] Pa")

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

    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")

    # Test configurations: (mu_type, mu_field, include_grad_term, name)
    test_configs = [
        ('constant_5000', mu_constant, False, 'const_no_grad'),
        ('constant_5000', mu_constant, True, 'const_with_grad'),
        ('heterogeneous', mu_heterogeneous, False, 'hetero_no_grad'),
        ('heterogeneous', mu_heterogeneous, True, 'hetero_with_grad'),
    ]

    results = []
    total_tests = len(neurons_list) * len(test_configs)
    test_count = 0

    print("\n" + "="*80)
    print(f"RUNNING {total_tests} TESTS")
    print("="*80)

    for n_neurons in neurons_list:
        print(f"\n{'='*80}")
        print(f"NEURONS = {n_neurons}")
        print(f"{'='*80}")

        # Create model for this neuron count
        print(f"\nCreating model with {n_neurons} neurons...")
        model = ForwardMREModelWithGradient(
            n_neurons=n_neurons,
            input_dim=3,
            omega_basis=omega_basis,
            seed=42,
            basis_type='sin'
        )

        for mu_type, mu_field, include_grad, name in test_configs:
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] {'-'*60}")
            print(f"Test: {name}, neurons={n_neurons}")
            print(f"  Mu type: {mu_type}")
            print(f"  Include gradient term: {include_grad}")
            print(f"{'-'*80}")

            try:
                # Solve
                u_pred = model.solve(
                    x, mu_field, bc_indices, u_bc_vals,
                    rho_omega2, bc_weight,
                    include_grad_term=include_grad
                )

                # Compute metrics
                error = u_pred - u_meas
                mse = torch.mean(error ** 2).item()
                var_u = torch.var(u_meas).item()
                r2 = 1 - mse / var_u if var_u > 0 else 0

                region_metrics = compute_region_metrics(u_pred, u_meas, mu_field)

                blob_r2 = region_metrics['blob']['r2'] if region_metrics['blob'] else None
                blob_mse = region_metrics['blob']['mse'] if region_metrics['blob'] else None

                blob_r2_str = f"{blob_r2:.4f}" if blob_r2 is not None else "N/A"
                print(f"  Overall R²: {r2:.4f}")
                print(f"  Blob R²: {blob_r2_str}")
                print(f"  MSE: {mse:.6e}")

                results.append({
                    'neurons': n_neurons,
                    'test_name': name,
                    'mu_type': mu_type,
                    'include_grad_term': include_grad,
                    'mse': mse,
                    'r2': r2,
                    'blob_r2': blob_r2,
                    'blob_mse': blob_mse,
                    'background_r2': region_metrics['background']['r2'] if region_metrics['background'] else None,
                    'u_pred': u_pred.cpu().numpy()
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'neurons': n_neurons,
                    'test_name': name,
                    'mu_type': mu_type,
                    'include_grad_term': include_grad,
                    'mse': None,
                    'r2': None,
                    'blob_r2': None,
                    'blob_mse': None,
                    'background_r2': None,
                    'u_pred': None
                })

    # Save results
    results_df = pd.DataFrame(results)
    results_df_csv = results_df.drop(columns=['u_pred'])
    results_df_csv.to_csv(output_dir / 'gradient_term_comparison.csv', index=False)

    # Analyze discrimination for each neuron count
    print("\n" + "="*80)
    print("DISCRIMINATION ANALYSIS BY NEURON COUNT")
    print("="*80)

    summary_data = []

    for n_neurons in neurons_list:
        print(f"\n{'='*80}")
        print(f"NEURONS = {n_neurons}")
        print(f"{'='*80}")

        df_n = results_df[results_df['neurons'] == n_neurons]

        # Extract results
        const_no = df_n[df_n['test_name'] == 'const_no_grad'].iloc[0]
        const_with = df_n[df_n['test_name'] == 'const_with_grad'].iloc[0]
        hetero_no = df_n[df_n['test_name'] == 'hetero_no_grad'].iloc[0]
        hetero_with = df_n[df_n['test_name'] == 'hetero_with_grad'].iloc[0]

        print("\n1. WITHOUT GRADIENT TERM (current implementation):")
        if const_no['r2'] is not None:
            blob_r2_str = f"{const_no['blob_r2']:.4f}" if const_no['blob_r2'] is not None else 'N/A'
            print(f"   Constant mu:      Overall R² = {const_no['r2']:.4f}, Blob R² = {blob_r2_str}")
        if hetero_no['r2'] is not None:
            blob_r2_str = f"{hetero_no['blob_r2']:.4f}" if hetero_no['blob_r2'] is not None else 'N/A'
            print(f"   Heterogeneous mu: Overall R² = {hetero_no['r2']:.4f}, Blob R² = {blob_r2_str}")

        blob_diff_no = None
        if const_no['blob_r2'] and hetero_no['blob_r2']:
            blob_diff_no = abs(hetero_no['blob_r2'] - const_no['blob_r2'])
            print(f"   Blob R² difference: {blob_diff_no:.4f}")

        print("\n2. WITH GRADIENT TERM (full weak form):")
        if const_with['r2'] is not None:
            blob_r2_str = f"{const_with['blob_r2']:.4f}" if const_with['blob_r2'] is not None else 'N/A'
            print(f"   Constant mu:      Overall R² = {const_with['r2']:.4f}, Blob R² = {blob_r2_str}")
        if hetero_with['r2'] is not None:
            blob_r2_str = f"{hetero_with['blob_r2']:.4f}" if hetero_with['blob_r2'] is not None else 'N/A'
            print(f"   Heterogeneous mu: Overall R² = {hetero_with['r2']:.4f}, Blob R² = {blob_r2_str}")

        blob_diff_with = None
        if const_with['blob_r2'] and hetero_with['blob_r2']:
            blob_diff_with = abs(hetero_with['blob_r2'] - const_with['blob_r2'])
            print(f"   Blob R² difference: {blob_diff_with:.4f}")

        # Compare gradient term impact
        print("\n3. GRADIENT TERM IMPACT:")

        const_improvement = None
        hetero_improvement = None
        disc_improvement = None

        if const_with['blob_r2'] and const_no['blob_r2']:
            const_improvement = const_with['blob_r2'] - const_no['blob_r2']
            print(f"   Constant mu: {const_improvement:+.4f} ({'improvement' if const_improvement > 0 else 'degradation'})")

        if hetero_with['blob_r2'] and hetero_no['blob_r2']:
            hetero_improvement = hetero_with['blob_r2'] - hetero_no['blob_r2']
            print(f"   Heterogeneous mu: {hetero_improvement:+.4f} ({'improvement' if hetero_improvement > 0 else 'degradation'})")

        if blob_diff_with is not None and blob_diff_no is not None:
            disc_improvement = blob_diff_with - blob_diff_no
            print(f"   Discrimination: {disc_improvement:+.4f}")

        # Store summary
        summary_data.append({
            'neurons': n_neurons,
            'const_no_blob_r2': const_no['blob_r2'],
            'const_with_blob_r2': const_with['blob_r2'],
            'hetero_no_blob_r2': hetero_no['blob_r2'],
            'hetero_with_blob_r2': hetero_with['blob_r2'],
            'const_improvement': const_improvement,
            'hetero_improvement': hetero_improvement,
            'discrimination_no_grad': blob_diff_no,
            'discrimination_with_grad': blob_diff_with,
            'discrimination_improvement': disc_improvement
        })

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'gradient_term_summary.csv', index=False)

    # Overall conclusion
    print("\n" + "="*80)
    print("OVERALL CONCLUSION")
    print("="*80)

    # Find best configuration
    hetero_results = results_df[results_df['mu_type'] == 'heterogeneous']
    hetero_results = hetero_results[hetero_results['blob_r2'].notna()]

    if len(hetero_results) > 0:
        best = hetero_results.loc[hetero_results['blob_r2'].idxmax()]
        print(f"\nBest heterogeneous Blob R²:")
        print(f"  Neurons: {best['neurons']}")
        print(f"  Gradient term: {'Yes' if best['include_grad_term'] else 'No'}")
        print(f"  Blob R²: {best['blob_r2']:.4f}")
        print(f"  Overall R²: {best['r2']:.4f}")

    # Check if gradient term helps consistently
    avg_hetero_improvement = summary_df['hetero_improvement'].mean()
    avg_disc_improvement = summary_df['discrimination_improvement'].mean()

    print(f"\nAverage gradient term impact (across {len(neurons_list)} neuron counts):")
    print(f"  Heterogeneous Blob R² change: {avg_hetero_improvement:+.4f}")
    print(f"  Discrimination change: {avg_disc_improvement:+.4f}")

    if avg_hetero_improvement > 0.01:
        print("\n[+] GRADIENT TERM HELPS: Include in forward model")
    elif avg_hetero_improvement < -0.01:
        print("\n[-] GRADIENT TERM HURTS: Abandon this approach")
    else:
        print("\n[=] GRADIENT TERM HAS NEGLIGIBLE EFFECT: Current approximation is sufficient")

    print(f"\n\nResults saved to:")
    print(f"  {output_dir / 'gradient_term_comparison.csv'}")
    print(f"  {output_dir / 'gradient_term_summary.csv'}")

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # 1. Blob R² comparison across neuron counts
        fig, ax = plt.subplots(figsize=(12, 6))

        neurons = summary_df['neurons'].values
        x_pos = np.arange(len(neurons))
        width = 0.2

        bars1 = ax.bar(x_pos - 1.5*width, summary_df['const_no_blob_r2'], width,
                       label='Constant (no grad)', color='lightblue')
        bars2 = ax.bar(x_pos - 0.5*width, summary_df['const_with_blob_r2'], width,
                       label='Constant (with grad)', color='blue')
        bars3 = ax.bar(x_pos + 0.5*width, summary_df['hetero_no_blob_r2'], width,
                       label='Heterogeneous (no grad)', color='lightcoral')
        bars4 = ax.bar(x_pos + 1.5*width, summary_df['hetero_with_blob_r2'], width,
                       label='Heterogeneous (with grad)', color='red')

        ax.set_xlabel('Number of Neurons', fontsize=12)
        ax.set_ylabel('Blob R²', fontsize=12)
        ax.set_title('Gradient Term Effect on Blob R² Across Neuron Counts', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(neurons)
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'blob_r2_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [1/5] Blob R² comparison plot saved")

        # 2. Gradient term improvement plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Heterogeneous improvement
        colors_hetero = ['green' if x > 0 else 'red' for x in summary_df['hetero_improvement']]
        ax1.bar(x_pos, summary_df['hetero_improvement'], color=colors_hetero, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.axhline(y=0.01, color='green', linestyle=':', linewidth=1, label='Threshold (+0.01)')
        ax1.axhline(y=-0.01, color='red', linestyle=':', linewidth=1, label='Threshold (-0.01)')
        ax1.set_xlabel('Number of Neurons', fontsize=12)
        ax1.set_ylabel('Blob R² Change', fontsize=12)
        ax1.set_title('Gradient Term Effect on Heterogeneous Case', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(neurons)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Discrimination improvement
        colors_disc = ['green' if x > 0 else 'red' for x in summary_df['discrimination_improvement']]
        ax2.bar(x_pos, summary_df['discrimination_improvement'], color=colors_disc, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.axhline(y=0.01, color='green', linestyle=':', linewidth=1, label='Threshold (+0.01)')
        ax2.axhline(y=-0.01, color='red', linestyle=':', linewidth=1, label='Threshold (-0.01)')
        ax2.set_xlabel('Number of Neurons', fontsize=12)
        ax2.set_ylabel('Discrimination Change', fontsize=12)
        ax2.set_title('Gradient Term Effect on Discrimination', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(neurons)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'gradient_term_improvement.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [2/5] Gradient term improvement plot saved")

        # 3. Discrimination comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(neurons, summary_df['discrimination_no_grad'], marker='o', linewidth=2,
                markersize=8, label='Without gradient term', color='blue')
        ax.plot(neurons, summary_df['discrimination_with_grad'], marker='s', linewidth=2,
                markersize=8, label='With gradient term', color='red')

        ax.set_xlabel('Number of Neurons', fontsize=12)
        ax.set_ylabel('Discrimination (|Blob R² difference|)', fontsize=12)
        ax.set_title('Discrimination Ability: Constant vs Heterogeneous', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'discrimination_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [3/5] Discrimination comparison plot saved")

        # 4. Overall R² vs Blob R² scatter
        fig, ax = plt.subplots(figsize=(10, 8))

        for idx, row in results_df.iterrows():
            if row['r2'] is not None and row['blob_r2'] is not None:
                marker = 'o' if row['mu_type'] == 'constant_5000' else '^'
                color = 'blue' if not row['include_grad_term'] else 'red'
                size = 50 + (row['neurons'] / 200)
                label_parts = []
                if idx == 0 or (results_df.iloc[idx-1]['mu_type'] != row['mu_type'] or
                                results_df.iloc[idx-1]['include_grad_term'] != row['include_grad_term']):
                    mu_label = 'Const' if row['mu_type'] == 'constant_5000' else 'Hetero'
                    grad_label = 'with grad' if row['include_grad_term'] else 'no grad'
                    label = f"{mu_label} ({grad_label})"
                else:
                    label = None

                ax.scatter(row['r2'], row['blob_r2'], marker=marker, s=size,
                          color=color, alpha=0.6, edgecolors='black', linewidth=0.5,
                          label=label)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Overall R²', fontsize=12)
        ax.set_ylabel('Blob R²', fontsize=12)
        ax.set_title('Overall R² vs Blob R² (size = neuron count)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.95, 1.0])
        ax.set_ylim([0.7, 1.0])

        plt.tight_layout()
        plt.savefig(output_dir / 'overall_vs_blob_r2.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [4/5] Overall vs Blob R² scatter plot saved")

        # 5. Summary heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create data matrix: rows = neurons, cols = [const_no, const_with, hetero_no, hetero_with]
        data_matrix = []
        for n in neurons_list:
            row_data = summary_df[summary_df['neurons'] == n]
            data_matrix.append([
                row_data['const_no_blob_r2'].values[0],
                row_data['const_with_blob_r2'].values[0],
                row_data['hetero_no_blob_r2'].values[0],
                row_data['hetero_with_blob_r2'].values[0]
            ])

        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)

        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(len(neurons_list)))
        ax.set_xticklabels(['Const\n(no grad)', 'Const\n(with grad)',
                           'Hetero\n(no grad)', 'Hetero\n(with grad)'])
        ax.set_yticklabels([f'{n} neurons' for n in neurons_list])

        # Add text annotations
        for i in range(len(neurons_list)):
            for j in range(4):
                text = ax.text(j, i, f'{data_matrix[i][j]:.4f}',
                              ha="center", va="center", color="black", fontsize=10)

        ax.set_title('Blob R² Heatmap: Gradient Term Effect', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Blob R²', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_dir / 'blob_r2_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [5/5] Blob R² heatmap saved")

        print(f"\nAll visualizations saved to: {output_dir}")

    except Exception as e:
        print(f"  ERROR creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    print("="*80)


if __name__ == "__main__":
    main()
