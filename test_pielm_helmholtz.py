"""
Test PIELM Helmholtz Solver with Bernstein Basis

Test with known homogeneous solution to verify proper functioning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pielm_helmholtz_bernstein import PIELMHelmholtzSolver
from bernstein_basis import verify_bernstein_derivatives


def generate_synthetic_homogeneous_data(n_data=1000, n_colloc=150):
    """
    Generate synthetic MRE data with homogeneous stiffness.
    
    Reduced collocation points to avoid ill-conditioning.
    Rule of thumb: N_colloc should be ~0.2-0.5 * N_data
    """
    # Domain
    x_min, x_max = 0.0, 0.08
    y_min, y_max = 0.0, 0.1
    z_min, z_max = 0.0, 0.01
    
    # Physics parameters
    mu_true = 5000.0  # Pa
    rho = 1000.0      # kg/m³
    freq = 60.0       # Hz
    omega = 2 * np.pi * freq
    
    # Wave number: k² = ρω²/μ
    k_squared = rho * omega**2 / mu_true
    k = np.sqrt(k_squared)
    
    print("Synthetic Data Generation")
    print("="*70)
    print(f"Domain: [{x_min}, {x_max}] × [{y_min}, {y_max}] × [{z_min}, {z_max}] m")
    print(f"Stiffness: μ = {mu_true} Pa (homogeneous)")
    print(f"Frequency: {freq} Hz")
    print(f"Wave number: k = {k:.3f} rad/m")
    print(f"Wavelength: λ = {2*np.pi/k:.4f} m")
    
    # Generate random points for data
    np.random.seed(42)
    X_data = np.random.rand(n_data, 3)
    X_data[:, 0] *= (x_max - x_min)
    X_data[:, 1] *= (y_max - y_min)
    X_data[:, 2] *= (z_max - z_min)
    
    # Analytical solution: plane wave u = A·exp(i·k·x)
    A = 1e-6  # Amplitude: 1 μm
    u_data = A * np.exp(1j * k * X_data[:, 0])
    
    # Add small noise
    noise_level = 0.02  # 2% noise
    u_data += noise_level * A * (np.random.randn(n_data) + 1j * np.random.randn(n_data))
    
    # Generate collocation points
    X_colloc = np.random.rand(n_colloc, 3)
    X_colloc[:, 0] *= (x_max - x_min)
    X_colloc[:, 1] *= (y_max - y_min)
    X_colloc[:, 2] *= (z_max - z_min)
    
    # True μ field
    mu_true_field = np.full(X_colloc.shape[0], mu_true)
    
    # Verify PDE is satisfied: μ∇²u + ρω²u = 0
    # For u = A·exp(i·k·x): ∇²u = -k²·u
    # So: μ·(-k²·u) + ρω²·u = -μk²·u + ρω²·u = -ρω²·u + ρω²·u = 0 ✓
    
    print(f"\nGenerated {n_data} data points")
    print(f"Generated {n_colloc} collocation points")
    print(f"Displacement range: {np.abs(u_data).min():.3e} to {np.abs(u_data).max():.3e} m")
    print("="*70)
    
    domain = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    return X_data, u_data, X_colloc, mu_true_field, domain, omega, rho


def test_basis_functions():
    """Test Bernstein basis function derivatives against finite differences."""
    print("\n" + "="*70)
    print("Test 1: Bernstein Basis Derivatives")
    print("="*70)
    
    domain = ((0, 0.08), (0, 0.1), (0, 0.01))
    results = verify_bernstein_derivatives(degrees=(5, 5, 4), domain=domain, h=1e-5)
    
    print(f"\nGradient error: {results['gradient_error']:.6e}")
    print(f"Laplacian error: {results['laplacian_error']:.6e}")
    
    if results['gradient_error'] < 1e-4:
        print("\n✓ Gradient test PASSED")
    else:
        print("\n✗ Gradient test FAILED")
    
    print("="*70)


def test_homogeneous_recovery():
    """Test recovery of homogeneous stiffness field."""
    print("\n" + "="*70)
    print("Test 2: Homogeneous Stiffness Recovery")
    print("="*70)
    
    # Generate data
    X_data, u_data, X_colloc, mu_true_field, domain, omega, rho = \
        generate_synthetic_homogeneous_data(n_data=1000, n_colloc=400)
    
    # Initialize solver with Bernstein basis - REDUCED DEGREES
    solver = PIELMHelmholtzSolver(
        degrees_u=(6, 7, 5),  # (6+1)*(7+1)*(5+1) = 336 features (reduced)
        degrees_mu=(5, 6, 4),  # (5+1)*(6+1)*(4+1) = 210 features
        domain=domain,
        omega=omega,
        rho=rho
    )
    
    # Train with more aggressive physics scheduling
    lambda_physics_schedule = [1.0, 5.0, 10.0] + [10.0] * 17
    lambda_reg_schedule = [5.0, 1.0, 0.5] + [0.1] * 17
    
    solver.train(
        X_data=X_data,
        u_data=u_data,
        X_colloc=X_colloc,
        n_iterations=20,
        lambda_data=1.0,
        lambda_physics_schedule=lambda_physics_schedule,
        lambda_reg_schedule=lambda_reg_schedule,
        ridge=1e-8,  # Slightly increased for safety
        verbose=True
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation on Test Set")
    print("="*70)
    
    # Generate test points
    X_test = np.random.rand(500, 3)
    X_test[:, 0] *= domain[0][1]
    X_test[:, 1] *= domain[1][1]
    X_test[:, 2] *= domain[2][1]
    
    # True solution
    mu_true = 5000.0
    k = np.sqrt(rho * omega**2 / mu_true)
    u_test_true = 1e-6 * np.exp(1j * k * X_test[:, 0])
    mu_test_true = np.full(X_test.shape[0], mu_true)
    
    # Predict
    u_pred = solver.predict_u(X_test)
    mu_pred = solver.predict_mu(X_test)
    
    # Errors
    u_error = np.sqrt(np.mean(np.abs(u_pred - u_test_true)**2))
    u_rel_error = u_error / (np.std(np.abs(u_test_true)) + 1e-10)
    
    mu_error = np.mean(np.abs(mu_pred - mu_test_true))
    mu_rel_error = mu_error / mu_true
    
    print(f"\nDisplacement:")
    print(f"  MSE: {u_error**2:.6e}")
    print(f"  Relative error: {u_rel_error*100:.2f}%")
    print(f"  True range: {np.abs(u_test_true).min():.3e} to {np.abs(u_test_true).max():.3e}")
    print(f"  Pred range: {np.abs(u_pred).min():.3e} to {np.abs(u_pred).max():.3e}")
    
    print(f"\nStiffness:")
    print(f"  MAE: {mu_error:.2f} Pa")
    print(f"  Relative error: {mu_rel_error*100:.2f}%")
    print(f"  True: {mu_true:.1f} Pa (constant)")
    print(f"  Predicted: {mu_pred.mean():.1f} ± {mu_pred.std():.1f} Pa")
    print(f"  Range: [{mu_pred.min():.1f}, {mu_pred.max():.1f}] Pa")
    
    # Check PDE residual
    lap_u_pred = solver.compute_laplacian_u(X_test)
    residual = mu_pred * lap_u_pred + rho * omega**2 * u_pred
    residual_norm = np.sqrt(np.mean(np.abs(residual)**2))
    
    print(f"\nPDE Residual:")
    print(f"  ||μ∇²u + ρω²u||: {residual_norm:.6e}")
    print(f"  Relative to ||ρω²u||: {residual_norm / (rho * omega**2 * np.abs(u_pred).mean()):.6e}")
    
    # Success criteria
    print("\n" + "="*70)
    print("Test Results:")
    print("="*70)
    
    passed = True
    
    if u_rel_error < 0.5:  # 50% relative error
        print("✓ Displacement recovery: PASSED")
    else:
        print("✗ Displacement recovery: FAILED")
        passed = False
    
    if mu_rel_error < 0.2:  # 20% error on μ
        print("✓ Stiffness recovery: PASSED")
    else:
        print("✗ Stiffness recovery: FAILED")
        passed = False
    
    if passed:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
    
    print("="*70)
    
    # Visualization
    visualize_results(X_test, u_test_true, u_pred, mu_test_true, mu_pred, solver)
    
    return solver


def visualize_results(X_test, u_true, u_pred, mu_true, mu_pred, solver):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Displacement magnitude comparison
    ax = axes[0, 0]
    ax.scatter(np.abs(u_true), np.abs(u_pred), alpha=0.5, s=10)
    lim = [0, max(np.abs(u_true).max(), np.abs(u_pred).max())]
    ax.plot(lim, lim, 'r--', label='Perfect')
    ax.set_xlabel('|u_true| (m)')
    ax.set_ylabel('|u_pred| (m)')
    ax.set_title('Displacement Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Displacement phase comparison
    ax = axes[0, 1]
    phase_true = np.angle(u_true)
    phase_pred = np.angle(u_pred)
    ax.scatter(phase_true, phase_pred, alpha=0.5, s=10)
    lim = [-np.pi, np.pi]
    ax.plot(lim, lim, 'r--', label='Perfect')
    ax.set_xlabel('Phase(u_true) (rad)')
    ax.set_ylabel('Phase(u_pred) (rad)')
    ax.set_title('Displacement Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Stiffness comparison
    ax = axes[0, 2]
    ax.scatter(mu_true, mu_pred, alpha=0.5, s=10)
    lim = [min(mu_true.min(), mu_pred.min()), max(mu_true.max(), mu_pred.max())]
    ax.plot(lim, lim, 'r--', label='Perfect')
    ax.set_xlabel('μ_true (Pa)')
    ax.set_ylabel('μ_pred (Pa)')
    ax.set_title('Stiffness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Training history - losses
    ax = axes[1, 0]
    if len(solver.history['loss_data']) > 0:
        iters = np.arange(0, len(solver.history['loss_data'])) * 5
        ax.semilogy(iters, solver.history['loss_data'], 'b-o', label='Data loss', markersize=4)
        ax.semilogy(iters, solver.history['loss_physics'], 'r-s', label='Physics loss', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Laplacian magnitude evolution
    ax = axes[1, 1]
    if len(solver.history['laplacian_magnitude']) > 0:
        iters = np.arange(0, len(solver.history['laplacian_magnitude'])) * 5
        ax.semilogy(iters, solver.history['laplacian_magnitude'], 'g-^', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean |∇²u|')
        ax.set_title('Laplacian Magnitude')
        ax.grid(True, alpha=0.3)
    
    # 6. μ range evolution
    ax = axes[1, 2]
    if len(solver.history['mu_range']) > 0:
        iters = np.arange(0, len(solver.history['mu_range'])) * 5
        mu_ranges = np.array(solver.history['mu_range'])
        ax.plot(iters, mu_ranges[:, 0], 'b-o', label='Min μ', markersize=4)
        ax.plot(iters, mu_ranges[:, 1], 'r-s', label='Max μ', markersize=4)
        ax.axhline(y=5000, color='g', linestyle='--', label='True μ')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('μ (Pa)')
        ax.set_title('Stiffness Range')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_pielm_helmholtz_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: test_pielm_helmholtz_results.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIELM HELMHOLTZ SOLVER - COMPREHENSIVE TESTS")
    print("="*70)
    
    # Test 1: Basis function derivatives
    test_basis_functions()
    
    # Test 2: Homogeneous stiffness recovery
    solver = test_homogeneous_recovery()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
