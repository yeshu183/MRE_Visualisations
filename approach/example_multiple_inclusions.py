"""Example 2: Multiple Inclusions - Two Gaussian peaks.

This script tests the framework's ability to recover multiple
spatially-separated stiffness variations.
"""

import torch
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import (
    generate_multiple_inclusions,
    train_inverse_problem,
    evaluate_reconstruction,
    plot_results
)

# Configuration (can override config_forward.json)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Run multiple inclusions example."""
    # Load base config and customize
    with open('approach/config_forward.json', 'r') as f:
        config = json.load(f)
    
    # Customize for this test
    config['seed'] = 42  # Different seed
    config['tv_weight'] = 0.0  # No TV for smooth features
    
    print("="*70)
    print("EXAMPLE: Multiple Inclusions (Two Peaks)")
    print("="*70)
    print(f"Device: {DEVICE}")
    print()
    
    # Generate data
    print("Generating synthetic data...")
    x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_multiple_inclusions(
        n_points=config['n_points'],
        n_wave_neurons=config['n_wave_neurons'],
        device=DEVICE,
        seed=config['seed']
    )
    print(f"  Ground truth mu range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
    print()
    
    # Train
    model, history = train_inverse_problem(
        x, u_meas, mu_true, bc_indices, u_bc_vals,
        config, device=DEVICE
    )
    
    # Evaluate
    with torch.no_grad():
        u_pred, mu_pred = model(x, bc_indices, u_bc_vals, 
                               config['rho_omega2'], 
                               bc_weight=config['bc_weight'])
    
    metrics = evaluate_reconstruction(
        mu_pred, mu_true, history['data_loss'][-1], verbose=True
    )
    
    # Visualize
    plot_results(
        x, u_meas, u_pred, u_true, mu_true, mu_pred, history,
        save_path='approach/results_multiple_inclusions.png',
        title_suffix=" (Two Peaks)"
    )
    
    # Pass/fail
    if metrics['data_loss'] < 1e-3 and metrics['relative_mse'] < 0.2:
        print("\n✅ MULTIPLE INCLUSIONS: PASSED")
        return True
    else:
        print("\n⚠️  MULTIPLE INCLUSIONS: Needs improvement")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
