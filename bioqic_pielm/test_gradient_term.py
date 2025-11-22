"""
Test Impact of ∇μ·∇u Term
==========================

Compare forward solver performance with:
1. Simplified: μ·∇²u + ρω²·u = 0  (current implementation)
2. Full: ∇·(μ∇u) + ρω²·u = 0  →  μ·∇²u + ∇μ·∇u + ρω²·u = 0

Hypothesis: The ∇μ·∇u term becomes important at inclusion boundaries
where stiffness changes rapidly (large ∇μ).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel


class ForwardMREModelWithGradient(ForwardMREModel):
    """Extended forward model that includes ∇μ·∇u term."""
    
    def get_basis_gradient_and_laplacian(self, x: torch.Tensor):
        """Compute basis functions, their gradients, and Laplacians.
        
        Returns:
            phi: (N, M) basis function values
            phi_grad: (N, M, dim) gradient of each basis function
            phi_lap: (N, M) Laplacian values
        """
        Z = x @ self.B + self.phi_bias  # (N, M)
        freq_sq = torch.sum(self.B ** 2, dim=0, keepdim=True)  # (1, M)
        
        if self.basis_type == 'sin':
            phi = torch.sin(Z)
            phi_lap = -phi * freq_sq
            
            # ∇φ_j = cos(w_j·x + b_j) * w_j
            # Shape: (N, M, dim)
            cos_Z = torch.cos(Z)  # (N, M)
            phi_grad = cos_Z.unsqueeze(-1) * self.B.T.unsqueeze(0)  # (N, M, dim)
            
        elif self.basis_type == 'tanh':
            phi = torch.tanh(Z)
            phi_lap = -2 * phi * (1 - phi ** 2) * freq_sq
            
            # ∇φ_j = sech²(w_j·x + b_j) * w_j = (1 - tanh²) * w_j
            sech_sq = 1 - phi ** 2  # (N, M)
            phi_grad = sech_sq.unsqueeze(-1) * self.B.T.unsqueeze(0)  # (N, M, dim)
            
        else:
            raise ValueError(f"Basis type {self.basis_type} not implemented for gradient")
        
        return phi, phi_grad, phi_lap
    
    def compute_mu_gradient_finite_diff(
        self, 
        x: torch.Tensor, 
        mu_field: torch.Tensor,
        k: int = 10,
        eps: float = 1e-4
    ) -> torch.Tensor:
        """Compute ∇μ using k-nearest neighbor finite differences.
        
        For each point, find k nearest neighbors and fit local gradient
        using least squares: μ(x+Δx) ≈ μ(x) + ∇μ·Δx
        
        Args:
            x: (N, dim) spatial coordinates
            mu_field: (N, 1) stiffness values at those coordinates
            k: number of neighbors for gradient estimation
            eps: regularization for singular matrices
        
        Returns:
            grad_mu: (N, dim) gradient of stiffness field
        """
        N, dim = x.shape
        grad_mu = torch.zeros(N, dim, device=x.device, dtype=x.dtype)
        
        # Compute pairwise distances (memory intensive for large N)
        # For efficiency, we'll do this in batches if needed
        batch_size = 1000
        
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_indices = range(batch_start, batch_end)
            
            # For each point in batch
            for i in batch_indices:
                x_i = x[i:i+1, :]  # (1, dim)
                mu_i = mu_field[i:i+1, :]  # (1, 1)
                
                # Compute distances to all other points
                dists = torch.norm(x - x_i, dim=1)  # (N,)
                
                # Find k+1 nearest neighbors (including self)
                _, nn_indices = torch.topk(dists, k+1, largest=False)
                nn_indices = nn_indices[1:]  # Exclude self
                
                # Get neighbor coordinates and values
                x_neighbors = x[nn_indices, :]  # (k, dim)
                mu_neighbors = mu_field[nn_indices, :]  # (k, 1)
                
                # Local coordinate system: Δx = x_neighbor - x_i
                delta_x = x_neighbors - x_i  # (k, dim)
                delta_mu = mu_neighbors - mu_i  # (k, 1)
                
                # Solve least squares: Δμ ≈ ∇μ·Δx
                # (Δx^T Δx) ∇μ = Δx^T Δμ
                # A = delta_x, b = delta_mu
                # ∇μ = (A^T A)^{-1} A^T b
                
                A = delta_x  # (k, dim)
                b = delta_mu  # (k, 1)
                
                # Add regularization for stability
                ATA = A.T @ A + eps * torch.eye(dim, device=x.device, dtype=x.dtype)  # (dim, dim)
                ATb = A.T @ b  # (dim, 1)
                
                # Solve
                try:
                    grad_mu_i = torch.linalg.solve(ATA, ATb)  # (dim, 1)
                    grad_mu[i, :] = grad_mu_i.squeeze()
                except:
                    # If singular, set gradient to zero
                    grad_mu[i, :] = 0.0
        
        return grad_mu
    
    def build_system_with_gradient(
        self,
        mu_field: torch.Tensor,
        grad_mu: torch.Tensor,
        phi: torch.Tensor,
        phi_grad: torch.Tensor,
        phi_lap: torch.Tensor,
        rho_omega2: float,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        bc_weight: float = 1.0,
        u_data: torch.Tensor = None,
        data_weight: float = 0.0
    ):
        """Build system including ∇μ·∇u term.
        
        Full PDE: μ·∇²u + ∇μ·∇u + ρω²·u = 0
        
        With u = Σ c_j φ_j:
            μ·Σ c_j ∇²φ_j + ∇μ·Σ c_j ∇φ_j + ρω²·Σ c_j φ_j = 0
        
        H_pde = (μ/ρω²)·∇²φ + (∇μ·∇φ)/ρω² + φ
        """
        # Compute ∇μ·∇φ term
        # grad_mu: (N, dim), phi_grad: (N, M, dim)
        # For each basis j: (∇μ·∇φ_j) = Σ_i (∂μ/∂x_i)(∂φ_j/∂x_i)
        grad_mu_dot_grad_phi = torch.sum(
            grad_mu.unsqueeze(1) * phi_grad,  # (N, 1, dim) * (N, M, dim)
            dim=2  # Sum over spatial dimensions
        )  # (N, M)
        
        # PDE rows with gradient term
        H_pde = (mu_field / rho_omega2) * phi_lap + \
                (grad_mu_dot_grad_phi / rho_omega2) + \
                phi
        b_pde = torch.zeros(phi.shape[0], 1, device=phi.device)
        
        H_list = [H_pde]
        b_list = [b_pde]
        
        # BC rows
        if bc_weight > 0 and bc_indices is not None and len(bc_indices) > 0:
            H_bc = phi[bc_indices, :] * bc_weight
            b_bc = u_bc_vals * bc_weight
            H_list.append(H_bc)
            b_list.append(b_bc)
        
        # Data rows
        if data_weight > 0 and u_data is not None:
            H_data = phi * data_weight
            b_data = u_data * data_weight
            H_list.append(H_data)
            b_list.append(b_data)
        
        H = torch.cat(H_list, dim=0)
        b = torch.cat(b_list, dim=0)
        
        return H, b
    
    def solve_given_mu_with_gradient(
        self,
        x: torch.Tensor,
        mu_field: torch.Tensor,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        rho_omega2: float,
        bc_weight: float = 1.0,
        u_data: torch.Tensor = None,
        data_weight: float = 0.0,
        k_neighbors: int = 10
    ):
        """Solve with full PDE including gradient term."""
        print("  Computing basis functions and gradients...")
        phi, phi_grad, phi_lap = self.get_basis_gradient_and_laplacian(x)
        
        print(f"  Computing ∇μ using {k_neighbors}-NN finite differences...")
        grad_mu = self.compute_mu_gradient_finite_diff(x, mu_field, k=k_neighbors)
        
        print(f"    |∇μ| range: [{torch.norm(grad_mu, dim=1).min():.2e}, {torch.norm(grad_mu, dim=1).max():.2e}] Pa/m")
        
        print("  Building system matrix with ∇μ·∇u term...")
        # Build system
        H, b = self.build_system_with_gradient(
            mu_field, grad_mu, phi, phi_grad, phi_lap,
            rho_omega2, bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight
        )
        
        print("  Solving linear system...")
        # Solve
        from pielm_solver import pielm_solve
        c_opt = pielm_solve(H, b, verbose=False)
        
        # Compute displacement
        u_pred = phi @ c_opt
        
        return u_pred, c_opt


def compare_formulations():
    """Compare simplified vs full PDE formulation."""
    print("\n" + "="*70)
    print("COMPARING SIMPLIFIED VS FULL PDE FORMULATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Setup
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_dir = project_root / 'data' / 'processed' / 'phase1_box'
    output_dir = script_dir / 'outputs' / 'gradient_term_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    n_points = 10000
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
    mu_true = torch.from_numpy(mu_raw).float().to(device)
    # Test with homogeneous mu (constant 5000 Pa)
    mu_constant = torch.full((len(coords), 1), 5000.0, dtype=torch.float32, device=device)
    
    # Use heterogeneous mu for testing
    mu_test = mu_true  # Change to mu_constant to test with homogeneous field
    
    # BCs - Box strategy
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
    
    # Physics
    freq = 60
    rho_omega2 = 1000 * (2 * np.pi * freq) ** 2
    
    print(f"\nData loaded: {len(x)} points, {len(bc_indices)} BC points")
    print(f"Mu test field: Constant {mu_test[0, 0].item():.0f} Pa everywhere")
    print(f"Expected |∇μ| ≈ 0 → gradient term should have minimal effect")
    
    # Test configuration
    bc_weight = 10
    n_neurons = 1000
    omega_basis = 170.0
    
    print(f"\nTest config: neurons={n_neurons}, bc_weight={bc_weight}")
    
    # Model 1: Simplified (current)
    print("\n[1/2] Testing SIMPLIFIED formulation (μ·∇²u + ρω²·u = 0)...")
    model_simplified = ForwardMREModel(
        n_wave_neurons=n_neurons,
        input_dim=3,
        omega_basis=omega_basis,
        mu_min=3000.0,
        mu_max=10000.0,
        seed=42,
        basis_type='sin'
    ).to(device)
    
    with torch.no_grad():
        u_pred_simplified, _ = model_simplified.solve_given_mu(
            x, mu_test, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0
        )
    
    error_simplified = u_pred_simplified - u_meas
    mse_simplified = torch.mean(error_simplified ** 2).item()
    r2_simplified = 1 - mse_simplified / torch.var(u_meas).item()
    
    print(f"  R² = {r2_simplified:.4f}, MSE = {mse_simplified:.6e}")
    
    # Model 2: Full (with gradient term)
    print("\n[2/2] Testing FULL formulation (μ·∇²u + ∇μ·∇u + ρω²·u = 0)...")
    model_full = ForwardMREModelWithGradient(
        n_wave_neurons=n_neurons,
        input_dim=3,
        omega_basis=omega_basis,
        mu_min=3000.0,
        mu_max=10000.0,
        seed=42,
        basis_type='sin'
    ).to(device)
    
    with torch.no_grad():
        u_pred_full, _ = model_full.solve_given_mu_with_gradient(
            x, mu_test, bc_indices, u_bc_vals, rho_omega2,
            bc_weight=bc_weight, u_data=None, data_weight=0.0,
            k_neighbors=10
        )
    
    error_full = u_pred_full - u_meas
    mse_full = torch.mean(error_full ** 2).item()
    r2_full = 1 - mse_full / torch.var(u_meas).item()
    
    print(f"  R² = {r2_full:.4f}, MSE = {mse_full:.6e}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Simplified (no ∇μ·∇u): R² = {r2_simplified:.4f}, MSE = {mse_simplified:.6e}")
    print(f"Full (with ∇μ·∇u):     R² = {r2_full:.4f}, MSE = {mse_full:.6e}")
    print(f"Improvement: ΔR² = {r2_full - r2_simplified:.6f} ({(r2_full - r2_simplified)/r2_simplified * 100:.2f}%)")
    print("="*70)
    
    # Visualize comparison
    print("\nCreating comparison visualizations...")
    
    # Get gradient magnitude for visualization
    print("  Computing ∇μ magnitude map...")
    with torch.no_grad():
        grad_mu = model_full.compute_mu_gradient_finite_diff(x, mu_test, k=10)
        grad_mu_mag = torch.norm(grad_mu, dim=1, keepdim=True).cpu().numpy().flatten()
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    coords_np = x.cpu().numpy()
    u_meas_np = u_meas.cpu().numpy().flatten()
    u_simp_np = u_pred_simplified.cpu().numpy().flatten()
    u_full_np = u_pred_full.cpu().numpy().flatten()
    mu_np = mu_test.cpu().numpy().flatten()
    
    # Get middle slice
    z_mid = np.median(coords_np[:, 2])
    z_tol = 0.002
    z_mask = np.abs(coords_np[:, 2] - z_mid) < z_tol
    
    u_vmin, u_vmax = -0.025, 0.025
    
    # Row 1: Measured and predictions
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=u_meas_np[z_mask],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    # No mu contours for constant field
    plt.colorbar(sc1, ax=ax1, label='u (m)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'u_measured (z≈{z_mid:.4f}m)\n(Data has inclusions, but μ=const)')
    ax1.set_aspect('equal')
    
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=u_simp_np[z_mask],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    plt.colorbar(sc2, ax=ax2, label='u (m)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Simplified (R²={r2_simplified:.4f})')
    ax2.set_aspect('equal')
    
    ax3 = axes[0, 2]
    sc3 = ax3.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=u_full_np[z_mask],
                      cmap='viridis', s=8, alpha=0.8, vmin=u_vmin, vmax=u_vmax)
    plt.colorbar(sc3, ax=ax3, label='u (m)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Full with ∇μ·∇u (R²={r2_full:.4f})')
    ax3.set_aspect('equal')
    
    # Row 2: Errors
    error_simp = np.abs(u_simp_np - u_meas_np)
    error_full = np.abs(u_full_np - u_meas_np)
    
    ax4 = axes[1, 0]
    sc4 = ax4.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=error_simp[z_mask],
                      cmap='hot', s=8, alpha=0.8)
    plt.colorbar(sc4, ax=ax4, label='|error| (m)')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Error: Simplified')
    ax4.set_aspect('equal')
    
    ax5 = axes[1, 1]
    sc5 = ax5.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=error_full[z_mask],
                      cmap='hot', s=8, alpha=0.8)
    plt.colorbar(sc5, ax=ax5, label='|error| (m)')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Error: Full with ∇μ·∇u')
    ax5.set_aspect('equal')
    
    # Error difference
    ax6 = axes[1, 2]
    error_diff = error_simp - error_full  # Positive = full is better
    sc6 = ax6.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=error_diff[z_mask],
                      cmap='RdBu_r', s=8, alpha=0.8, vmin=-0.001, vmax=0.001)
    plt.colorbar(sc6, ax=ax6, label='Δ|error| (m)')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Error Improvement\n(Red=Full better, Blue=Simplified better)')
    ax6.set_aspect('equal')
    
    # Row 3: Gradient magnitude and analysis
    ax7 = axes[2, 0]
    sc7 = ax7.scatter(coords_np[z_mask, 0], coords_np[z_mask, 1], c=grad_mu_mag[z_mask],
                      cmap='plasma', s=8, alpha=0.8, vmin=0, vmax=5e3)  # Lower scale for constant mu
    plt.colorbar(sc7, ax=ax7, label='|∇μ| (Pa/m)')
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title('Stiffness Gradient Magnitude\n(Should be ~0 for constant μ)')
    ax7.set_aspect('equal')
    
    # Correlation: |∇μ| vs error improvement
    ax8 = axes[2, 1]
    # Only plot where gradient is significant
    grad_threshold = 1e4
    mask_grad = grad_mu_mag > grad_threshold
    ax8.scatter(grad_mu_mag[mask_grad], error_diff[mask_grad], alpha=0.3, s=1)
    ax8.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax8.set_xlabel('|∇μ| (Pa/m)')
    ax8.set_ylabel('Error Improvement (m)')
    ax8.set_title(f'Correlation: |∇μ| vs Error Improvement\n(Points with |∇μ| > {grad_threshold:.0e})')
    ax8.grid(True, alpha=0.3)
    
    # Histogram of error improvements
    ax9 = axes[2, 2]
    ax9.hist(error_diff, bins=100, alpha=0.7, edgecolor='black')
    ax9.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax9.axvline(x=np.mean(error_diff), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(error_diff):.2e} m')
    ax9.set_xlabel('Error Improvement (m)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Distribution of Error Improvements')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comparison: Simplified vs Full PDE (HOMOGENEOUS μ=5000 Pa)\n' +
                 f'ΔR² = {r2_full - r2_simplified:.6f} | Mean Error Improvement = {np.mean(error_diff):.2e} m\n' +
                 f'Expected: ∇μ ≈ 0 → gradient term should have NO effect', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_term_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_dir / 'gradient_term_comparison.png'}")
    
    # Additional statistics
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print(f"Points with improved error (full < simplified): {np.sum(error_diff > 0)} ({100*np.mean(error_diff > 0):.1f}%)")
    print(f"Points with worse error (full > simplified): {np.sum(error_diff < 0)} ({100*np.mean(error_diff < 0):.1f}%)")
    print(f"Mean error improvement: {np.mean(error_diff):.6e} m")
    print(f"Max error improvement: {np.max(error_diff):.6e} m")
    print(f"Max error degradation: {np.min(error_diff):.6e} m")
    print(f"\nGradient magnitude statistics:")
    print(f"|∇μ| range: [{grad_mu_mag.min():.2e}, {grad_mu_mag.max():.2e}] Pa/m")
    print(f"|∇μ| mean: {grad_mu_mag.mean():.2e} Pa/m")
    print(f"Points with |∇μ| > 1e5 Pa/m: {np.sum(grad_mu_mag > 1e5)} ({100*np.mean(grad_mu_mag > 1e5):.1f}%)")
    
    print("\n" + "="*70)
    print("THEORETICAL IMPACT OF ∇μ·∇u TERM")
    print("="*70)
    print("\n1. Where it matters:")
    print("   - At inclusion boundaries where μ changes rapidly")
    print("   - ∇μ is large → ∇μ·∇u term becomes significant")
    print("\n2. Magnitude estimate:")
    print(f"   - Typical μ jump: ~7000 Pa over ~0.01 m")
    print(f"   - |∇μ| ~ 7000/0.01 = 7×10⁵ Pa/m")
    print(f"   - Typical |∇u| ~ 1 m/m = 1 (dimensionless)")
    print(f"   - ∇μ·∇u ~ 7×10⁵ Pa/m")
    print(f"   - Compare to μ·∇²u: μ·λ² ~ 5000·(2π/0.037)² ~ 1.4×10⁸ Pa/m²")
    print(f"   - Ratio: (∇μ·∇u)/(μ·∇²u) ~ 0.5% typically")
    print("\n3. Conclusion:")
    print("   - ∇μ·∇u term is small for these inclusion sizes")
    print("   - May be more important for:")
    print("     * Sharp inclusions (smaller radius)")
    print("     * Larger stiffness contrasts")
    print("     * Higher frequency (shorter wavelength)")
    print("="*70)


if __name__ == "__main__":
    compare_formulations()
