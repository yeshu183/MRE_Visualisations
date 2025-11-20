"""
BIOQIC Phase 1 Training Script
================================

Main orchestration script that brings together all components:
- Data loading with flexible displacement modes
- Physics-informed boundary detection  
- Configurable stiffness network
- Forward model with physics scaling options
- Comprehensive training loop with visualization

Theory: Same gradient-based optimization as approach folder
------------------------------------------------------------
The core methodology is IDENTICAL:
1. Parameterize μ(x) with neural network
2. Solve forward problem: u = PIELM_solve(μ, BCs)
3. Backpropagate through differentiable solver
4. Update μ network weights with Adam
5. Physics: Helmholtz equation ∇·(μ∇u) + ρω²u = 0

Improvements:
- Better boundary detection (actuator vs tolerance)
- Flexible stiffness network bounds
- Multiple displacement modes
- Physics scaling options (physical/effective)
- Comprehensive visualization

Usage:
    python train_bioqic.py --experiment baseline
    python train_bioqic.py --experiment actuator
    python train_bioqic.py --experiment vector
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
import sys

# Import our modules
from data_loader import BIOQICDataLoader
from boundary_detection import BoundaryDetector
from stiffness_network import FlexibleStiffnessNetwork
from forward_model import ForwardMREModel
from trainer import MRETrainer


# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENTS = {
    'baseline': {
        'description': 'Simplest configuration - minimal BCs, z-component, effective physics',
        'displacement_mode': 'z_component',
        'boundary_strategy': 'minimal',
        'stiffness_strategy': 'direct',
        'physics_mode': 'effective',
        'mu_min': 0.2,
        'mu_max': 1.2,
        'data_weight': 100.0,
        'bc_weight': 10.0,
        'tv_weight': 0.001,
        'l2_weight': 0.0,
        'n_wave_neurons': 100,
        'lr': 0.01,
        'iterations': 3000,
        'subsample': 1000
    },
    
    'actuator': {
        'description': 'Physics-informed BCs - actuator at top Y-face',
        'displacement_mode': 'z_component',
        'boundary_strategy': 'actuator',
        'stiffness_strategy': 'direct',
        'physics_mode': 'effective',
        'mu_min': 0.2,
        'mu_max': 1.2,
        'data_weight': 50.0,
        'bc_weight': 50.0,
        'tv_weight': 0.001,
        'l2_weight': 0.0,
        'n_wave_neurons': 100,
        'lr': 0.01,
        'iterations': 3000,
        'subsample': 1000
    },
    
    'vector': {
        'description': 'Full 3-component vector displacement',
        'displacement_mode': '3_components',
        'boundary_strategy': 'actuator',
        'stiffness_strategy': 'direct',
        'physics_mode': 'effective',
        'mu_min': 0.2,
        'mu_max': 1.2,
        'data_weight': 50.0,
        'bc_weight': 50.0,
        'tv_weight': 0.001,
        'l2_weight': 0.0,
        'n_wave_neurons': 100,
        'lr': 0.01,
        'iterations': 3000,
        'subsample': 1000
    },
    
    'physical': {
        'description': 'Physical ρω² with coordinate rescaling',
        'displacement_mode': 'z_component',
        'boundary_strategy': 'actuator',
        'stiffness_strategy': 'direct',
        'physics_mode': 'physical',
        'mu_min': 0.2,
        'mu_max': 1.2,
        'data_weight': 50.0,
        'bc_weight': 50.0,
        'tv_weight': 0.001,
        'l2_weight': 0.0,
        'n_wave_neurons': 100,
        'lr': 0.005,  # Lower LR for physical mode
        'iterations': 3000,
        'subsample': 1000
    },
    
    'strong_tv': {
        'description': 'Strong TV regularization for piecewise constant',
        'displacement_mode': 'z_component',
        'boundary_strategy': 'actuator',
        'stiffness_strategy': 'direct',
        'physics_mode': 'effective',
        'mu_min': 0.2,
        'mu_max': 1.2,
        'data_weight': 50.0,
        'bc_weight': 50.0,
        'tv_weight': 0.01,  # 10× stronger TV
        'l2_weight': 0.0,
        'n_wave_neurons': 100,
        'lr': 0.01,
        'iterations': 3000,
        'subsample': 1000
    },
    
    'more_data': {
        'description': 'More data points (5000) for better coverage',
        'displacement_mode': 'z_component',
        'boundary_strategy': 'actuator',
        'stiffness_strategy': 'direct',
        'physics_mode': 'effective',
        'mu_min': 0.2,
        'mu_max': 1.2,
        'data_weight': 50.0,
        'bc_weight': 50.0,
        'tv_weight': 0.001,
        'l2_weight': 0.0,
        'n_wave_neurons': 150,
        'lr': 0.01,
        'iterations': 5000,
        'subsample': 5000
    }
}


def print_experiment_config(name: str, config: dict):
    """Print experiment configuration."""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {name}")
    print("="*80)
    print(f"\n📝 Description: {config['description']}")
    print(f"\n⚙️  Configuration:")
    print(f"   Displacement mode: {config['displacement_mode']}")
    print(f"   Boundary strategy: {config['boundary_strategy']}")
    print(f"   Stiffness strategy: {config['stiffness_strategy']}")
    print(f"   Physics mode: {config['physics_mode']}")
    print(f"   μ range: [{config['mu_min']}, {config['mu_max']}]")
    print(f"   Data weight: {config['data_weight']}")
    print(f"   BC weight: {config['bc_weight']}")
    print(f"   TV weight: {config['tv_weight']}")
    print(f"   L2 weight: {config['l2_weight']}")
    print(f"   Wave neurons: {config['n_wave_neurons']}")
    print(f"   Learning rate: {config['lr']}")
    print(f"   Iterations: {config['iterations']:,}")
    print(f"   Subsampled points: {config['subsample']:,}")


def run_experiment(experiment_name: str, device: torch.device, seed: int = 42):
    """Run a single experiment."""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get configuration
    if experiment_name not in EXPERIMENTS:
        print(f"❌ Unknown experiment: {experiment_name}")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        return
    
    config = EXPERIMENTS[experiment_name]
    print_experiment_config(experiment_name, config)
    
    # Create output directory
    output_dir = Path('outputs') / experiment_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    loader = BIOQICDataLoader(
        displacement_mode=config['displacement_mode'],
        subsample=config['subsample'],
        seed=seed
    )
    
    # Load and convert to tensors
    raw_data = loader.load()
    data = loader.to_tensors(raw_data, device)
    
    x = data['x']
    u_meas = data['u_meas']
    mu_true = data['mu_true']
    coords = data['coords']
    coords_norm = data['coords_norm']
    scales = data['scales']
    
    # Get physics parameters based on config
    physics_params = loader.get_physics_params(strategy=config['physics_mode'])
    rho_omega2 = physics_params['rho_omega2']
    
    print(f"\n✅ Data loaded:")
    print(f"   x shape: {x.shape}")
    print(f"   u_meas shape: {u_meas.shape}")
    print(f"   mu_true shape: {mu_true.shape}")
    print(f"   ρω²: {rho_omega2:.1f}")
    
    # =========================================================================
    # Step 2: Boundary Detection
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: BOUNDARY DETECTION")
    print("="*80)
    
    detector = BoundaryDetector(strategy=config['boundary_strategy'])
    bc_indices, u_bc_vals, bc_info = detector.detect(
        coords, coords_norm, u_meas, device, subsample=5
    )
    
    # If weighted strategy, extract weights
    weights = bc_info.get('weights', None)
    
    # =========================================================================
    # Step 3: Initialize Models
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: MODEL INITIALIZATION")
    print("="*80)
    
    # Stiffness network
    print("\n🏗️  Creating stiffness network...")
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3,
        mu_min=config['mu_min'],
        mu_max=config['mu_max'],
        output_strategy=config['stiffness_strategy'],
        hidden_dim=64,
        n_layers=3,
        n_fourier=10
    ).to(device)
    
    n_params = sum(p.numel() for p in mu_net.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   Output range: [{config['mu_min']}, {config['mu_max']}]")
    
    # Forward model
    print("\n🏗️  Creating forward model...")
    model = ForwardMREModel(
        n_wave_neurons=config['n_wave_neurons'],
        input_dim=3,
        mu_network=mu_net,
        physics_mode=config['physics_mode'],
        omega_basis=15.0,
        seed=seed
    ).to(device)
    
    print(f"   Wave neurons: {config['n_wave_neurons']}")
    print(f"   Physics mode: {config['physics_mode']}")
    
    # Optimizer
    optimizer = torch.optim.Adam(mu_net.parameters(), lr=config['lr'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.8
    )
    
    # =========================================================================
    # Step 4: Training
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    trainer = MRETrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir
    )
    
    trainer.train(
        x=x,
        u_meas=u_meas,
        mu_true=mu_true,
        bc_indices=bc_indices,
        u_bc_vals=u_bc_vals,
        rho_omega2=rho_omega2,  # Use the physics parameter we retrieved
        scales=scales,
        n_iterations=config['iterations'],
        weights=weights,
        data_weight=config['data_weight'],
        bc_weight=config['bc_weight'],
        tv_weight=config['tv_weight'],
        l2_weight=config['l2_weight'],
        mu_prior_mean=0.5,
        grad_clip_norm=1.0,
        log_interval=50,
        plot_interval=500,
        early_stopping_patience=1000,
        save_best=True
    )
    
    print("\n" + "="*80)
    print(f"✅ EXPERIMENT '{experiment_name}' COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - config.json")
    print(f"   - training_history.json")
    print(f"   - best_model.pt")
    print(f"   - progress_iter_*.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train MRE inverse problem on BIOQIC Phase 1 data'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='baseline',
        choices=list(EXPERIMENTS.keys()),
        help='Experiment configuration to run'
    )
    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='List available experiments and exit'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.list_experiments:
        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        for name, config in EXPERIMENTS.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")
        print("\n" + "="*80)
        return
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "="*80)
    print("BIOQIC PHASE 1 TRAINING - Modular MRE Inversion")
    print("="*80)
    print(f"\n🖥️  Device: {device}")
    print(f"🎲 Seed: {args.seed}")
    
    # Run experiment
    run_experiment(args.experiment, device, args.seed)
    
    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
