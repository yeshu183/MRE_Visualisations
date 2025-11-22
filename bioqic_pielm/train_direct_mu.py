"""
Direct Mu Training Script
==========================

Train MRE inverse problem using DIRECT gradient-based optimization.

Key Difference from train.py:
- NO neural network
- μ is directly optimized at each point
- Classical adjoint gradient descent

Usage:
    python train_direct_mu.py                           # Default: MSE loss
    python train_direct_mu.py --loss_type sobolev       # Sobolev loss
    python train_direct_mu.py --subsample 5000          # Quick test
    python train_direct_mu.py --lr 20.0                 # Higher learning rate
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from data_loader import BIOQICDataLoader
from direct_mu_model import DirectMuModel
from direct_mu_trainer import DirectMuTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Direct Mu Inverse')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'correlation', 'relative_l2', 'sobolev'],
                        help='Loss function type')
    parser.add_argument('--subsample', type=int, default=5000,
                        help='Number of points to subsample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    parser.add_argument('--data_dir', type=str,
                        default='data/processed/phase1_box',
                        help='Path to BIOQIC data')
    parser.add_argument('--lr', type=float, default=10.0,
                        help='Learning rate (higher for direct optimization)')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Number of iterations')
    parser.add_argument('--bc_weight', type=float, default=1000.0,
                        help='BC enforcement weight')
    parser.add_argument('--tv_weight', type=float, default=0.0,
                        help='TV regularization weight')
    parser.add_argument('--init_mode', type=str, default='constant',
                        choices=['constant', 'random', 'uniform'],
                        help='Mu initialization mode')
    parser.add_argument('--mu_init', type=float, default=5000.0,
                        help='Initial mu value (Pa) for constant init')
    parser.add_argument('--pde_weight', type=float, default=0.0,
                        help='PDE residual loss weight')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"Direct Mu Optimization: {args.loss_type.upper()} Loss")
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
        displacement_mode='z_component',
        subsample=args.subsample,
        seed=args.seed
    )
    data = loader.load()

    # Use RAW SI units (same as physical_box experiment)
    print(f"\nUsing RAW SI units")
    coords = data['coords']  # meters
    u_raw = data['u_raw']    # meters (signed displacement)
    mu_raw = data['mu_raw']  # Pa (raw stiffness)

    x = torch.from_numpy(coords).float().to(device)
    u_meas = torch.from_numpy(u_raw).float().to(device)
    mu_true = torch.from_numpy(mu_raw).float().to(device)

    print(f"  u_meas range: [{u_meas.min():.6e}, {u_meas.max():.6e}] m")
    print(f"  mu_true range: [{mu_true.min():.0f}, {mu_true.max():.0f}] Pa")

    # Boundary conditions (box strategy)
    print("\n  BC Strategy: Box (all 6 faces)")
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    tol = 1e-4
    mask = (np.abs(coords[:, 0] - x_min) < tol) | (np.abs(coords[:, 0] - x_max) < tol) | \
           (np.abs(coords[:, 1] - y_min) < tol) | (np.abs(coords[:, 1] - y_max) < tol) | \
           (np.abs(coords[:, 2] - z_min) < tol) | (np.abs(coords[:, 2] - z_max) < tol)
    bc_indices_np = np.where(mask)[0]
    bc_indices = torch.from_numpy(bc_indices_np).long().to(device)
    u_bc_vals = u_meas[bc_indices]

    print(f"  BC points: {len(bc_indices)} ({100*len(bc_indices)/len(x):.1f}%)")
    print(f"  BC u_meas range: [{u_bc_vals.min():.6e}, {u_bc_vals.max():.6e}]")

    # Physics parameter (physical value)
    rho_omega2 = data['scales']['rho_omega2_physical']
    print(f"\nPhysical rho_omega2: {rho_omega2:.0f}")

    # Create DIRECT MU MODEL (no neural network!)
    n_points = len(x)
    model = DirectMuModel(
        n_points=n_points,
        n_wave_neurons=1000,  # Same as physical_box
        input_dim=3,
        omega_basis=170.0,
        mu_init=args.mu_init,
        mu_min=3000.0,
        mu_max=10000.0,
        seed=args.seed,
        basis_type='sin',
        init_mode=args.init_mode
    ).to(device)

    print(f"\nModel configuration:")
    print(f"  Parameterization: DIRECT (no NN)")
    print(f"  Mu shape: {model.mu_field.shape}")
    print(f"  Mu init mode: {args.init_mode}")
    print(f"  Mu range: [3000.0, 10000.0] Pa")
    print(f"  Wave neurons: 1000")
    print(f"  Omega basis: 170.0")

    # Create trainer
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / 'outputs' / f'direct_mu_{args.loss_type}'
    trainer = DirectMuTrainer(model, device, str(output_dir), loss_type=args.loss_type)

    print(f"\nLoss function: {args.loss_type}")
    print(f"Output directory: {output_dir}")

    # Train
    results = trainer.train(
        x=x,
        u_meas=u_meas,
        mu_true=mu_true,
        bc_indices=bc_indices,
        u_bc_vals=u_bc_vals,
        rho_omega2=rho_omega2,
        iterations=args.iterations,
        lr=args.lr,
        bc_weight=args.bc_weight,
        data_weight=0.0,  # Pure physics-based (no data in forward solve)
        tv_weight=args.tv_weight,
        pde_weight=args.pde_weight,  # NEW: PDE residual loss
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
