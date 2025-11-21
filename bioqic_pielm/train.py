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
                        default='../data/processed/phase1_box',
                        help='Path to BIOQIC data')
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
        seed=args.seed
    )
    data = loader.load()
    tensors = loader.to_tensors(data, device)

    # Get boundary indices
    bc_indices_np = loader.get_boundary_indices(
        data['coords'],
        strategy=config['bc_strategy']
    )
    bc_indices = torch.from_numpy(bc_indices_np).long().to(device)

    # BC values from measured displacement
    u_bc_vals = tensors['u_meas'][bc_indices]

    print(f"\nBoundary conditions ({config['bc_strategy']}):")
    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(tensors['x']):.1f}%)")

    # Physics parameter
    if config['rho_omega2'] is None:
        # Use physical value
        rho_omega2 = data['scales']['rho_omega2_physical']
        print(f"\nUsing physical rho_omega2: {rho_omega2:.0f}")
    else:
        rho_omega2 = config['rho_omega2']
        print(f"\nUsing effective rho_omega2: {rho_omega2:.0f}")

    # Create model
    model = ForwardMREModel(
        n_wave_neurons=config['n_wave_neurons'],
        input_dim=3,  # 3D BIOQIC data
        omega_basis=config['omega_basis'],
        hidden_dim=64,
        n_fourier=10,
        mu_min=0.0,
        mu_max=1.0,
        seed=args.seed
    ).to(device)

    # Create trainer
    output_dir = f"./outputs/{args.experiment}"
    trainer = MRETrainer(model, device, output_dir)

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
        log_interval=500,
        save_interval=1000
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
