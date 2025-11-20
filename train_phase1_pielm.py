"""
Phase 1 Training Script: PIELM-MRE on BIOQIC Box Phantom
===========================================================

Trains Iterative PIELM on Phase 1 data:
- Dataset: BIOQIC four_target_phantom (4 inclusions, known ground truth)
- Goal: Recover heterogeneous stiffness distribution
- Method: Iterative alternating optimization with curriculum learning

Author: Yeshwanth Kesav
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import modules
from pielm_mre import IterativePIELMMRE
from physics_module import HelmholtzPhysics


def load_phase1_data():
    """
    Load preprocessed Phase 1 BIOQIC Box data.
    
    Returns:
        Dictionary with coordinates, displacement, ground truth, etc.
    """
    data_dir = Path('data/processed/phase1_box')
    
    print("="*60)
    print("Loading Phase 1 Data (BIOQIC Box Phantom)")
    print("="*60)
    
    # Load arrays
    coordinates = np.load(data_dir / 'coordinates.npy')
    coordinates_norm = np.load(data_dir / 'coordinates_normalized.npy')
    displacement = np.load(data_dir / 'displacement.npy')
    colloc_pts = np.load(data_dir / 'collocation_points.npy')
    stiffness_gt = np.load(data_dir / 'stiffness_ground_truth.npy')
    
    # Load preprocessing params
    params = np.load(data_dir / 'preprocessing_params.npy', allow_pickle=True).item()
    
    print(f"\n📊 Data Loaded:")
    print(f"   Coordinates: {coordinates.shape}")
    print(f"   Displacement: {displacement.shape} (complex)")
    print(f"   Ground truth stiffness: {stiffness_gt.shape}")
    print(f"   Collocation points: {colloc_pts.shape}")
    print(f"   Frequency: {params['frequency_hz']} Hz")
    print(f"   Voxel size: {params['voxel_size_m']*1000:.1f} mm")
    
    data = {
        'coordinates': coordinates,
        'coordinates_norm': coordinates_norm,
        'displacement': displacement,
        'collocation_points': colloc_pts,
        'stiffness_ground_truth': stiffness_gt,
        'params': params
    }
    
    return data


def prepare_training_data(data, use_component: int = 0, subsample: int = 1):
    """
    Prepare data for PIELM training.
    
    Args:
        data: Dictionary from load_phase1_data()
        use_component: Which displacement component to use (0=x, 1=y, 2=z)
        subsample: Subsampling factor for faster training (1=no subsample)
    
    Returns:
        X_data, u_data, X_colloc, mu_gt (all at sampled points)
    """
    print(f"\n{'='*60}")
    print("Preparing Training Data")
    print(f"{'='*60}")
    
    # Use normalized coordinates for better numerical stability
    X = data['coordinates_norm']
    u_full = data['displacement'][:, use_component]  # Select component
    mu_gt = data['stiffness_ground_truth'].squeeze()
    
    # Subsample for faster training (optional)
    if subsample > 1:
        indices = np.arange(0, X.shape[0], subsample)
        X = X[indices]
        u_full = u_full[indices]
        mu_gt = mu_gt[indices]
    
    # Split into data points (80%) and collocation points (20%)
    N_total = X.shape[0]
    N_data = int(0.8 * N_total)
    
    indices = np.random.permutation(N_total)
    data_indices = indices[:N_data]
    colloc_indices = indices[N_data:]
    
    X_data = X[data_indices]
    u_data = u_full[data_indices]
    
    X_colloc = X[colloc_indices]
    mu_gt_colloc = mu_gt[colloc_indices]
    
    print(f"\n📌 Training Split:")
    print(f"   Data points (80%): {X_data.shape[0]:,}")
    print(f"   Collocation points (20%): {X_colloc.shape[0]:,}")
    print(f"   Using displacement component: {['u_x', 'u_y', 'u_z'][use_component]}")
    print(f"   Displacement magnitude range: [{np.abs(u_data).min():.2e}, {np.abs(u_data).max():.2e}]")
    
    return X_data, u_data, X_colloc, mu_gt_colloc


def train_pielm(X_data, u_data, X_colloc, frequency=60.0, n_neurons=800):
    """
    Train PIELM-MRE solver.
    
    Args:
        X_data: Data points (N_data, 3)
        u_data: Measured displacement (N_data,) complex
        X_colloc: Collocation points (N_colloc, 3)
        frequency: MRE frequency (Hz)
        n_neurons: Number of hidden neurons
    
    Returns:
        Trained solver
    """
    print(f"\n{'='*60}")
    print("Initializing PIELM-MRE Solver")
    print(f"{'='*60}")
    
    # Initialize solver
    solver = IterativePIELMMRE(
        n_neurons=n_neurons,
        frequency=frequency,
        density=1000.0,
        seed=42
    )
    
    # Training configuration
    config = {
        'max_iterations': 30,
        'lambda_data': 0.1,  # REDUCED: Don't overfit to data
        'lambda_physics': 10.0,  # INCREASED: Force physics compliance
        'lambda_reg_schedule': [10.0, 1.0, 0.1],  # Higher regularization
        'ridge': 1e-6,  # Increased regularization
        'verbose': True
    }
    
    print(f"\n📋 Training Configuration:")
    for key, val in config.items():
        print(f"   {key}: {val}")
    
    # Train
    start_time = time.time()
    solver.train(
        X_data=X_data,
        u_measured=u_data,
        X_colloc=X_colloc,
        **config
    )
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Training completed in {elapsed:.2f} seconds")
    
    return solver


def evaluate_results(solver, X_test, u_test, mu_gt, params):
    """
    Evaluate trained solver against ground truth.
    
    Args:
        solver: Trained PIELM solver
        X_test: Test points
        u_test: Test displacement
        mu_gt: Ground truth stiffness
        params: Data parameters
    """
    print(f"\n{'='*60}")
    print("Evaluating Results")
    print(f"{'='*60}")
    
    # Predict displacement
    u_pred = solver.u_network.predict(X_test)
    
    # Predict stiffness
    mu_pred = solver.mu_network.predict(X_test)
    
    # Compute errors
    u_error = np.abs(u_pred - u_test)
    u_mse = np.mean(u_error**2)
    u_rel_error = np.sqrt(u_mse) / (np.mean(np.abs(u_test)) + 1e-10)
    
    mu_error_real = np.abs(mu_pred.real - mu_gt.real)
    mu_mae = np.mean(mu_error_real)
    mu_rel_error = mu_mae / (np.mean(np.abs(mu_gt.real)) + 1e-10)
    
    print(f"\n📊 Displacement Prediction:")
    print(f"   MSE: {u_mse:.2e}")
    print(f"   Relative Error: {u_rel_error*100:.2f}%")
    print(f"   Max Absolute Error: {u_error.max():.2e}")
    
    print(f"\n🎯 Stiffness Reconstruction:")
    print(f"   MAE (storage modulus): {mu_mae:.1f} Pa")
    print(f"   Relative Error: {mu_rel_error*100:.2f}%")
    print(f"   Predicted range: [{mu_pred.real.min():.1f}, {mu_pred.real.max():.1f}] Pa")
    print(f"   Ground truth range: [{mu_gt.real.min():.1f}, {mu_gt.real.max():.1f}] Pa")
    
    results = {
        'u_pred': u_pred,
        'u_test': u_test,
        'u_error': u_error,
        'u_mse': u_mse,
        'mu_pred': mu_pred,
        'mu_gt': mu_gt,
        'mu_mae': mu_mae
    }
    
    return results


def visualize_results(results, solver):
    """
    Create visualizations of training results.
    
    Args:
        results: Dictionary from evaluate_results()
        solver: Trained PIELM solver
    """
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")
    
    # 1. Training History
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    history = solver.history
    iterations = history['iteration']
    
    # Loss curves
    axes[0].semilogy(iterations, history['loss_total'], 'b-', lw=2, label='Total Loss')
    axes[0].semilogy(iterations, history['loss_data'], 'r--', lw=1.5, label='Data Loss')
    axes[0].semilogy(iterations, history['loss_physics_mu'], 'g:', lw=1.5, label='Physics Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Stiffness comparison
    mu_pred_real = results['mu_pred'].real
    mu_gt_real = results['mu_gt'].real
    
    axes[1].hist(mu_gt_real, bins=50, alpha=0.6, label='Ground Truth', color='blue', edgecolor='black')
    axes[1].hist(mu_pred_real, bins=50, alpha=0.6, label='Predicted', color='red', edgecolor='black')
    axes[1].set_xlabel('Storage Modulus (Pa)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Stiffness Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase1_training_results.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: phase1_training_results.png")
    plt.show()
    
    # 2. Error Analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Displacement error
    u_error = np.abs(results['u_error'])
    axes[0].hist(u_error, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0].set_xlabel('Absolute Error (m)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Displacement Prediction Error')
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)
    
    # Stiffness error
    mu_error = np.abs(results['mu_pred'].real - results['mu_gt'].real)
    axes[1].hist(mu_error, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1].set_xlabel('Absolute Error (Pa)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Stiffness Reconstruction Error')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase1_error_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: phase1_error_analysis.png")
    plt.show()


def main():
    """
    Main training pipeline for Phase 1.
    """
    print("\n" + "="*60)
    print("PHASE 1: PIELM-MRE Training on BIOQIC Box Phantom")
    print("="*60)
    
    # 1. Load data
    data = load_phase1_data()
    
    # 2. Prepare training data
    X_data, u_data, X_colloc, mu_gt = prepare_training_data(
        data,
        use_component=0,  # Use u_x component
        subsample=10  # Use every 10th point for faster training
    )
    
    # 3. Train PIELM
    solver = train_pielm(
        X_data=X_data,
        u_data=u_data,
        X_colloc=X_colloc,
        frequency=data['params']['frequency_hz'],
        n_neurons=500  # Start with moderate size
    )
    
    # 4. Evaluate on collocation points
    results = evaluate_results(
        solver=solver,
        X_test=X_colloc,
        u_test=u_data[: len(X_colloc)],  # Use subset for eval
        mu_gt=mu_gt,
        params=data['params']
    )
    
    # 5. Visualize
    visualize_results(results, solver)
    
    print(f"\n{'='*60}")
    print("✅ Phase 1 Training Complete!")
    print(f"{'='*60}")
    print("\n📌 Next Steps:")
    print("   1. Tune hyperparameters (n_neurons, lambda weights)")
    print("   2. Implement full derivative computation for μ network")
    print("   3. Move to Phase 2: BIOQIC Abdomen simulation")
    print("   4. Implement multi-frequency training")


if __name__ == "__main__":
    main()
