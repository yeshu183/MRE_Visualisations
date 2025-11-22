"""
BIOQIC-PIELM Training Script
============================

Main script for training MRE stiffness reconstruction on BIOQIC data.

Usage:
    python train.py                           # Default: baseline experiment
    python train.py --experiment actuator     # Physics-informed BCs
    python train.py --experiment data_only    # Data-driven (no PDE)
    python train.py --subsample 5000          # Quick test with fewer points

Experiments:
- baseline: Minimal BCs, z-component displacement
- actuator: Top Y-face BCs (physics-informed)
- data_only: Pure data fitting without PDE constraints
- strong_tv: Strong regularization for sharp features
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from data_loader import BIOQICDataLoader
from forward_model import ForwardMREModel
from trainer import MRETrainer


# Experiment configurations
EXPERIMENTS = {
    'baseline': {
        'description': 'Baseline with minimal BCs and effective physics',
        'displacement_mode': 'z_component',
        'bc_strategy': 'minimal',
        'rho_omega2': 400.0,  # Effective (tuned)
        'bc_weight': 200.0,
        'data_weight': 0.0,
        'tv_weight': 0.001,
        'n_wave_neurons': 60,
        'omega_basis': 15.0,
        'iterations': 3000,
        'lr': 0.005,
    },
    'actuator': {
        'description': 'Physics-informed BCs on top Y-face (actuator location)',
        'displacement_mode': 'z_component',
        'bc_strategy': 'actuator',
        'rho_omega2': 400.0,
        'bc_weight': 100.0,  # Lower since more BC points
        'data_weight': 0.0,
        'tv_weight': 0.001,
        'n_wave_neurons': 60,
        'omega_basis': 15.0,
        'iterations': 3000,
        'lr': 0.005,
    },
    'data_only': {
        'description': 'Data-driven approach (pure interpolation, no PDE)',
        'displacement_mode': 'z_component',
        'bc_strategy': 'minimal',
        'rho_omega2': 400.0,
        'bc_weight': 0.0,  # No BC enforcement
        'data_weight': 100.0,  # Strong data fitting
        'tv_weight': 0.002,
        'n_wave_neurons': 100,  # More basis for flexibility
        'omega_basis': 20.0,
        'iterations': 5000,
        'lr': 0.01,
    },
    'strong_tv': {
        'description': 'Strong TV regularization for piecewise-constant stiffness',
        'displacement_mode': 'z_component',
        'bc_strategy': 'minimal',
        'rho_omega2': 400.0,
        'bc_weight': 200.0,
        'data_weight': 0.0,
        'tv_weight': 0.01,  # Strong TV
        'n_wave_neurons': 60,
        'omega_basis': 15.0,
        'iterations': 5000,
        'lr': 0.003,
    },
    'physical': {
        'description': 'Physical rho_omega2 with coordinate rescaling',
        'displacement_mode': 'z_component',
        'bc_strategy': 'actuator',
        'rho_omega2': None,  # Computed from frequency
        'bc_weight': 100.0,
        'data_weight': 0.0,
        'tv_weight': 0.001,
        'n_wave_neurons': 80,
        'omega_basis': 10.0,  # Lower for physical scale
        'iterations': 5000,
        'lr': 0.003,
    },
    'physical_box': {
        'description': 'Correlation Loss: Box BC + Data Guidance',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,  # Optimal from forward analysis
        'data_weight': 10.0,
        'tv_weight': 0.01,
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'correlation',
    },
    'physical_relative_l2': {
        'description': 'Relative L2 Loss: Normalized for Small Displacements',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,  # Optimal from forward analysis
        'data_weight': 10.0,
        'tv_weight': 0.01,
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'relative_l2',
    },
    'physical_sobolev': {
        'description': 'Sobolev Loss: α=0.1, β=0.9 (90% Gradient, Optimal)',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,  # Strong BC constraint for unique solution
        'data_weight': 0.0,  # Pure physics-based inverse (no data constraints in forward solve)
        'tv_weight': 0.0,  # REDUCED from 0.01 (was crushing spatial variation)
        'use_cnn_mu': True,
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',  # Uses α=0.1, β=0.9 in trainer
    },
    'physical_mse': {
        'description': 'Standard MSE Loss: Baseline Comparison',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,  # Optimal from forward analysis
        'data_weight': 10.0,
        'tv_weight': 0.01,
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'mse',
    },
    'physical_sobolev_mlp': {
        'description': 'Sine Basis + Standard MLP (no CNN) - Baseline for RBF comparison',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,  # Pure physics-based inverse
        'tv_weight': 0.0,
        'use_cnn_mu': False,  # Use standard StiffnessNetwork (MLP)
        'basis_type': 'sin',  # Sine basis (current default)
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',
    },
    'physical_sobolev_rbf': {
        'description': 'RBF Basis + Standard MLP - Test locality hypothesis',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 2000,  # More neurons for RBF coverage
        'omega_basis': 170.0,  # σ = 5.88mm (same as sine for fair comparison)
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,  # Pure physics-based inverse
        'tv_weight': 0.0,
        'use_cnn_mu': False,  # Use standard StiffnessNetwork (MLP)
        'basis_type': 'rbf',  # KEY CHANGE: RBF instead of sine
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',
    },
    'physical_mse_mlp': {
        'description': 'Sine + MLP with MSE Loss - Fix Sobolev gradient issues',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'mse',  # MSE instead of Sobolev
    },
    'physical_sobolev_mlp_data': {
        'description': 'Sine + MLP + Data Weight - Stabilize inverse problem',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 10.0,  # Add data constraint to forward solve
        'tv_weight': 0.001,   # Weak TV regularization
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',
    },
    'physical_sobolev_mlp_lowlr': {
        'description': 'Sine + MLP + Low LR - Prevent gradient explosion',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.0001,  # 10× lower learning rate
        'loss_type': 'sobolev',
    },
    # RBF experiments with different loss functions
    'rbf_sobolev': {
        'description': 'RBF + Sobolev (α=0.1, β=0.9) - Optimal from analysis',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,  # REDUCED from 2000 (same as sine for fair comparison)
        'omega_basis': 170.0,    # σ = 5.88mm
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,      # Pure physics inverse
        'tv_weight': 0.0,        # No regularization
        'use_cnn_mu': False,
        'basis_type': 'rbf',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',
    },
    'rbf_correlation': {
        'description': 'RBF + Correlation Loss - Shape matching',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,  # REDUCED from 2000 for speed
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'rbf',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'correlation',
    },
    'rbf_relative_l2': {
        'description': 'RBF + Relative L2 - Normalized for small displacements',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,  # REDUCED from 2000 for speed
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'rbf',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'relative_l2',
    },
    'rbf_mse': {
        'description': 'RBF + MSE - Baseline comparison (poor discrimination)',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,  # REDUCED from 2000 for speed
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'rbf',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'mse',
    },
    # Sine experiments with different loss functions (for comparison)
    'sine_correlation': {
        'description': 'Sine + Correlation - Compare with RBF',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'correlation',
    },
    'sine_relative_l2': {
        'description': 'Sine + Relative L2 - Compare with RBF',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'relative_l2',
    },
    # Experiments with Prior + Barrier Regularization to prevent saturation
    'sobolev_prior_barrier': {
        'description': 'Sobolev + Prior + Barrier - Prevent saturation at bounds',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'pde_weight': 1.0,         # Stronger PDE enforcement
        'prior_weight': 0.01,      # Pull toward mean (6500 Pa)
        'barrier_weight': 0.001,   # Penalize hitting bounds
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',
    },
    'correlation_prior_barrier': {
        'description': 'Correlation + Prior + Barrier - Stable with regularization',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'pde_weight': 1.0,
        'prior_weight': 0.01,
        'barrier_weight': 0.001,
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'correlation',
    },
    'sobolev_barrier_only': {
        'description': 'Sobolev + Barrier (no prior) - Test barrier effectiveness',
        'displacement_mode': 'z_component',
        'bc_strategy': 'box',
        'rho_omega2': None,
        'n_wave_neurons': 1000,
        'omega_basis': 170.0,
        'mu_range': (3000.0, 10000.0),
        'bc_weight': 10.0,
        'data_weight': 0.0,
        'tv_weight': 0.0,
        'pde_weight': 1.0,
        'prior_weight': 0.0,       # No prior
        'barrier_weight': 0.01,    # Stronger barrier
        'use_cnn_mu': False,
        'basis_type': 'sin',
        'iterations': 5000,
        'lr': 0.001,
        'loss_type': 'sobolev',
    },
}


def main():
    parser = argparse.ArgumentParser(description='Train BIOQIC-PIELM')
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment configuration')
    parser.add_argument('--subsample', type=int, default=5000,
                        help='Number of points to subsample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    parser.add_argument('--data_dir', type=str,
                        default='data/processed/phase1_box',
                        help='Path to BIOQIC data')
    parser.add_argument('--pde_weight', type=float, default=0.0,
                        help='PDE residual loss weight')
    args = parser.parse_args()

    # Get experiment config
    config = EXPERIMENTS[args.experiment].copy()
    print("\n" + "=" * 70)
    print(f"BIOQIC-PIELM: {args.experiment}")
    print(f"Description: {config['description']}")
    print("=" * 70)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    loader = BIOQICDataLoader(
        data_dir=args.data_dir,
        displacement_mode=config['displacement_mode'],
        subsample=args.subsample,
        seed=args.seed,
        adaptive_sampling=config.get('adaptive_sampling', False),
        blob_sample_ratio=config.get('blob_sample_ratio', 0.5),
        boundary_sample_ratio=config.get('boundary_sample_ratio', 0.3)
    )
    data = loader.load()
    
    # For physical experiments (with rho_omega2=None), use RAW SI units instead of normalized
    use_raw_units = (config['rho_omega2'] is None)
    
    if use_raw_units:
        print(f"\nUsing RAW SI units for {args.experiment} experiment")
        coords = data['coords']  # meters
        u_raw = data['u_raw']    # meters (signed displacement)
        mu_raw = data['mu_raw']  # Pa (raw stiffness)
        
        x = torch.from_numpy(coords).float().to(device)
        u_meas = torch.from_numpy(u_raw).float().to(device)
        mu_true = torch.from_numpy(mu_raw).float().to(device)
        
        tensors = {
            'x': x,
            'u_meas': u_meas,
            'mu_true': mu_true,
            'coords': coords,
            'scales': data['scales'],
            'params': data['params']
        }
        
        print(f"  u_meas range (raw): [{u_meas.min():.6e}, {u_meas.max():.6e}] m")
        print(f"  mu_true range (raw): [{mu_true.min():.0f}, {mu_true.max():.0f}] Pa")
    else:
        # Use normalized data for other experiments
        tensors = loader.to_tensors(data, device)

    # Get boundary indices
    # Handle Box Strategy manually if requested
    if config['bc_strategy'] == 'box':
        print("\n  Strategy: Box (Selecting all 6 faces)")
        coords = data['coords']
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        tol = 1e-4
        mask = (np.abs(coords[:, 0] - x_min) < tol) | (np.abs(coords[:, 0] - x_max) < tol) | \
               (np.abs(coords[:, 1] - y_min) < tol) | (np.abs(coords[:, 1] - y_max) < tol) | \
               (np.abs(coords[:, 2] - z_min) < tol) | (np.abs(coords[:, 2] - z_max) < tol)
        bc_indices_np = np.where(mask)[0]
    else:
        bc_indices_np = loader.get_boundary_indices(
            data['coords'],
            strategy=config['bc_strategy']
        )
    bc_indices = torch.from_numpy(bc_indices_np).long().to(device)

    # BC values from measured displacement
    u_bc_vals = tensors['u_meas'][bc_indices]

    print(f"\nBoundary conditions ({config['bc_strategy']}):")
    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(tensors['x']):.1f}%)")
    print(f"  BC u_meas range: [{u_bc_vals.min():.4f}, {u_bc_vals.max():.4f}]")

    # IMPORTANT: BC values should have non-zero variation to drive waves
    # Check if BC values are too uniform
    bc_std = u_bc_vals.std()
    if bc_std < 1e-4:
        print(f"  WARNING: BC values have low variation (std={bc_std:.2e})")
        print(f"           This may cause trivial u=0 solution")

    # Physics parameter
    if config['rho_omega2'] is None:
        # Use physical value
        rho_omega2 = data['scales']['rho_omega2_physical']
        print(f"\nUsing physical rho_omega2: {rho_omega2:.0f}")
    else:
        rho_omega2 = config['rho_omega2']
        print(f"\nUsing effective rho_omega2: {rho_omega2:.0f}")

    # Determine mu range (physical or normalized)
    mu_min, mu_max = config.get('mu_range', (1.0, 2.0))  # Default to normalized if not specified
    
    # Create model
    basis_type = config.get('basis_type', 'sin')  # Default to 'sin' if not specified
    model = ForwardMREModel(
        n_wave_neurons=config['n_wave_neurons'],
        input_dim=3,  # 3D BIOQIC data
        omega_basis=config['omega_basis'],
        hidden_dim=64,
        n_fourier=20,  # Increased from 10 to 20 for better spatial frequency
        mu_min=mu_min,  # Now uses 3000.0 for physical_box experiment
        mu_max=mu_max,  # Now uses 10000.0 for physical_box experiment
        seed=args.seed,
        basis_type=basis_type
    ).to(device)
    
    print(f"\nModel configuration:")
    print(f"  Basis type: {basis_type}")
    print(f"  Wave neurons: {config['n_wave_neurons']}")
    print(f"  Omega basis: {config['omega_basis']}")
    if basis_type == 'rbf':
        rbf_sigma = 1.0 / config['omega_basis']
        print(f"  RBF sigma: {rbf_sigma*1000:.2f} mm")
    print(f"  Mu range: [{mu_min}, {mu_max}]")

    # Optionally replace the default mu network with a CNN-based parameterization
    if config.get('use_cnn_mu', False):
        try:
            from .cnn_stiffness import CNNStiffnessNetwork
        except Exception:
            from cnn_stiffness import CNNStiffnessNetwork

        grid_shape = tuple(data['params'].get('grid_shape', (50, 50, 10)))
        print(f"\nReplacing mu network with CNN parameterization, grid_shape={grid_shape}")
        cnn_net = CNNStiffnessNetwork(grid_shape=grid_shape, mu_min=mu_min, mu_max=mu_max, seed=args.seed)
        model.mu_net = cnn_net.to(device)

    # Create trainer with loss type
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / 'outputs' / args.experiment
    loss_type = config.get('loss_type', 'correlation')  # Default to correlation
    trainer = MRETrainer(model, device, str(output_dir), loss_type=loss_type)
    
    print(f"\nLoss function: {loss_type}")

    # Train
    results = trainer.train(
        x=tensors['x'],
        u_meas=tensors['u_meas'],
        mu_true=tensors['mu_true'],
        bc_indices=bc_indices,
        u_bc_vals=u_bc_vals,
        rho_omega2=rho_omega2,
        iterations=config['iterations'],
        lr=config['lr'],
        bc_weight=config['bc_weight'],
        data_weight=config['data_weight'],
        tv_weight=config['tv_weight'],
        pde_weight=config.get('pde_weight', args.pde_weight),
        prior_weight=config.get('prior_weight', 0.0),
        barrier_weight=config.get('barrier_weight', 0.0),
        log_interval=100,
        save_interval=500
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Data Loss: {results['final_loss']:.6e}")
    print(f"  Mu MSE:    {results['final_mu_mse']:.6e}")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
