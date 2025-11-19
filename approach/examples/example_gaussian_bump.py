"""Example 1: Gaussian Bump - Single stiffness inclusion.

This script demonstrates the inverse problem solution for a simple
Gaussian bump in the stiffness distribution.
"""

import torch
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_generators import generate_gaussian_bump
from core.solver import train_inverse_problem, evaluate_reconstruction
from core.visualization import plot_results

# Configuration - get path relative to this file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', 'config_forward.json')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Run Gaussian bump example."""
    # Load configuration
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    print("="*70)
    print("EXAMPLE: Gaussian Bump (Single Inclusion)")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Config: {config}")
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_gaussian_bump(
        n_points=config['n_points'],
        n_wave_neurons=config['n_wave_neurons'],
        device=DEVICE,
        seed=config['seed']
    )
    print(f"  Data generated: {len(x)} points")
    print(f"  Ground truth mu range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
    print()
    
    # Train inverse problem
    model, history = train_inverse_problem(
        x, u_meas, mu_true, bc_indices, u_bc_vals,
        config, device=DEVICE
    )
    
    # Get final predictions
    with torch.no_grad():
        u_pred, mu_pred = model(x, bc_indices, u_bc_vals, 
                               config['rho_omega2'], 
                               bc_weight=config['bc_weight'])
    
    # Evaluate
    metrics = evaluate_reconstruction(
        mu_pred, mu_true, history['data_loss'][-1], verbose=True
    )
    
    # Visualize
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_results(
        x, u_meas, u_pred, u_true, mu_true, mu_pred, history,
        save_path=os.path.join(results_dir, 'gaussian_bump_data.png'),
        title_suffix=" (Gaussian Bump)"
    )
    
    # Pass/fail criteria
    if metrics['data_loss'] < 1e-4 and metrics['relative_mse'] < 0.1:
        print("\n✅ GAUSSIAN BUMP: PASSED")
        return True
    else:
        print("\n⚠️  GAUSSIAN BUMP: Needs improvement")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
