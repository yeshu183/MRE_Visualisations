"""
Visualize polynomial vs wave basis comparison.

Shows:
1. MSE vs polynomial degree
2. MSE comparison: wave vs polynomial
3. Reconstruction quality comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model_polynomial import ForwardMREModelPolynomial
from forward_model_v3 import ForwardMREModelV3
import sys


def visualize_basis_comparison():
    """Create comprehensive visualization of basis function comparison."""
    
    print("="*70)
    print("Basis Function Comparison Visualization")
    print("="*70)
    
    # Load data
    print("\n1. Loading BIOQIC data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None
    )
    sys.stdout.flush()
    data = loader.load()
    
    # Subsample
    subsample = 2000
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    
    # BC
    bc_indices = torch.randperm(subsample)[:20]
    u_bc_vals = u_meas[bc_indices]
    
    # Create mu network
    mu_network = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.2, mu_max=1.2, seed=42
    )
    
    print("\n2. Testing polynomial basis (degrees 2-10)...")
    poly_degrees = list(range(2, 11))
    poly_mses = []
    poly_n_basis = []
    
    for degree in poly_degrees:
        model = ForwardMREModelPolynomial(
            mu_network=mu_network, poly_degree=degree, seed=42
        )
        
        with torch.no_grad():
            u_pred, _ = model(
                x, bc_indices, u_bc_vals, bc_weight=1.0,
                u_data=u_meas, data_weight=1000.0, verbose=False
            )
        
        mse = torch.mean((u_pred - u_meas)**2).item()
        poly_mses.append(mse)
        poly_n_basis.append(model.n_basis)
        print(f"   Degree {degree}: MSE = {mse:.6f}, n_basis = {model.n_basis}")
    
    print("\n3. Testing wave basis (omega 1-50)...")
    wave_omegas = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    wave_mses = []
    
    for omega in wave_omegas:
        model = ForwardMREModelV3(
            mu_network=mu_network, n_wave_neurons=200,
            omega_basis=omega, seed=42
        )
        
        with torch.no_grad():
            u_pred, _ = model(
                x, bc_indices, u_bc_vals, bc_weight=1.0,
                u_data=u_meas, data_weight=1000.0,
                use_pde=False, verbose=False
            )
        
        mse = torch.mean((u_pred - u_meas)**2).item()
        wave_mses.append(mse)
        print(f"   Omega {omega}: MSE = {mse:.6f}")
    
    # Best models for reconstruction comparison
    print("\n4. Generating reconstructions for comparison...")
    best_poly_degree = poly_degrees[np.argmin(poly_mses)]
    best_wave_omega = wave_omegas[np.argmin(wave_mses)]
    
    # Best polynomial
    model_poly = ForwardMREModelPolynomial(
        mu_network=mu_network, poly_degree=best_poly_degree, seed=42
    )
    with torch.no_grad():
        u_pred_poly, _ = model_poly(
            x, bc_indices, u_bc_vals, bc_weight=1.0,
            u_data=u_meas, data_weight=1000.0, verbose=False
        )
    
    # Best wave
    model_wave = ForwardMREModelV3(
        mu_network=mu_network, n_wave_neurons=200,
        omega_basis=best_wave_omega, seed=42
    )
    with torch.no_grad():
        u_pred_wave, _ = model_wave(
            x, bc_indices, u_bc_vals, bc_weight=1.0,
            u_data=u_meas, data_weight=1000.0,
            use_pde=False, verbose=False
        )
    
    # Create visualization
    print("\n5. Creating plots...")
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: MSE vs polynomial degree
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(poly_degrees, poly_mses, 'bo-', linewidth=2, markersize=8, label='Polynomial')
    ax1.axhline(y=0.01, color='g', linestyle='--', linewidth=2, label='Target (MSE=0.01)')
    ax1.set_xlabel('Polynomial Degree', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax1.set_title('Polynomial Basis: MSE vs Degree', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, max(poly_mses)*1.1])
    
    # Plot 2: Number of basis functions vs MSE
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(poly_n_basis, poly_mses, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.01, color='g', linestyle='--', linewidth=2, label='Target')
    ax2.set_xlabel('Number of Basis Functions', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax2.set_title('Polynomial: Basis Count vs MSE', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Wave basis comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(wave_omegas, wave_mses, 'ms-', linewidth=2, markersize=8, label='Wave (200 neurons)')
    ax3.axhline(y=min(poly_mses), color='b', linestyle='--', linewidth=2, 
                label=f'Best Poly (MSE={min(poly_mses):.4f})')
    ax3.set_xlabel('Omega (frequency scale)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax3.set_title('Wave Basis: MSE vs Frequency', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Scatter - measured vs polynomial prediction
    ax4 = plt.subplot(2, 3, 4)
    u_meas_np = u_meas.numpy()
    u_poly_np = u_pred_poly.numpy()
    ax4.scatter(u_meas_np, u_poly_np, alpha=0.3, s=10, c='blue')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    ax4.set_xlabel('Measured u', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted u (Polynomial)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Polynomial (deg={best_poly_degree}): MSE={min(poly_mses):.4f}', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_aspect('equal')
    
    # Plot 5: Scatter - measured vs wave prediction
    ax5 = plt.subplot(2, 3, 5)
    u_wave_np = u_pred_wave.numpy()
    ax5.scatter(u_meas_np, u_wave_np, alpha=0.3, s=10, c='magenta')
    ax5.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
    ax5.set_xlabel('Measured u', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Predicted u (Wave)', fontsize=12, fontweight='bold')
    ax5.set_title(f'Wave (ω={best_wave_omega}): MSE={min(wave_mses):.4f}', 
                  fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_aspect('equal')
    
    # Plot 6: Error distribution comparison
    ax6 = plt.subplot(2, 3, 6)
    error_poly = (u_pred_poly - u_meas).numpy()
    error_wave = (u_pred_wave - u_meas).numpy()
    ax6.hist(error_poly, bins=50, alpha=0.6, color='blue', label=f'Polynomial (σ={error_poly.std():.4f})')
    ax6.hist(error_wave, bins=50, alpha=0.6, color='magenta', label=f'Wave (σ={error_wave.std():.4f})')
    ax6.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax6.set_title('Error Distribution Comparison', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = 'outputs/basis_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n📊 Polynomial Basis:")
    print(f"   Best degree: {best_poly_degree}")
    print(f"   Best MSE: {min(poly_mses):.6f}")
    print(f"   Number of basis: {poly_n_basis[poly_degrees.index(best_poly_degree)]}")
    
    print(f"\n🌊 Wave Basis:")
    print(f"   Best omega: {best_wave_omega}")
    print(f"   Best MSE: {min(wave_mses):.6f}")
    print(f"   Number of basis: 200")
    
    improvement = min(wave_mses) / min(poly_mses)
    print(f"\n✅ Improvement: {improvement:.1f}× better with polynomial basis")
    
    if min(poly_mses) < 0.01:
        print(f"   🎉 SUCCESS! Polynomial achieves MSE < 0.01")
    elif min(poly_mses) < 0.02:
        print(f"   ✓ Very close! MSE = {min(poly_mses):.6f} (need < 0.01)")
    
    plt.show()


if __name__ == '__main__':
    visualize_basis_comparison()
