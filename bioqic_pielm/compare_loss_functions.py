"""
Compare Different Loss Functions for MRE Inversion
==================================================

Tests correlation, relative L2, Sobolev, and MSE losses to determine
which provides better sensitivity for detecting stiffness variations.

Usage:
    python compare_loss_functions.py
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def run_experiment(experiment_name, subsample=5000):
    """Run training with specified loss function."""
    print(f"\n{'='*70}")
    print(f"Running: {experiment_name}")
    print('='*70)
    
    cmd = [
        'python', 'bioqic_pielm/train.py',
        '--experiment', experiment_name,
        '--subsample', str(subsample)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    return True


def load_results(experiment_name):
    """Load training history from experiment."""
    output_dir = Path('bioqic_pielm/outputs') / experiment_name
    
    # Try to load history (if saved as pickle or json)
    history_file = output_dir / 'history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    
    return None


def visualize_comparison():
    """Create comprehensive comparison plots."""
    experiments = [
        'physical_box',          # Correlation
        'physical_relative_l2',  # Relative L2
        'physical_sobolev',      # Sobolev (gradient-enhanced)
        'physical_mse',          # Standard MSE
    ]
    
    loss_names = [
        'Correlation',
        'Relative L2',
        'Sobolev',
        'MSE'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Loss Function Comparison for MRE Inversion', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Collect histories
    histories = []
    for exp in experiments:
        hist = load_results(exp)
        histories.append(hist)
    
    # Plot 1: Data Loss Evolution
    ax = axes[0, 0]
    for i, (exp, name, color) in enumerate(zip(experiments, loss_names, colors)):
        hist = histories[i]
        if hist:
            ax.plot(hist.get('data_loss', []), label=name, color=color, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Data Loss')
    ax.set_title('Data Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Mu MSE (Reconstruction Error)
    ax = axes[0, 1]
    for i, (exp, name, color) in enumerate(zip(experiments, loss_names, colors)):
        hist = histories[i]
        if hist:
            ax.plot(hist.get('mu_mse', []), label=name, color=color, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mu MSE (Pa²)')
    ax.set_title('Stiffness Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Gradient Norm (Learning Signal)
    ax = axes[0, 2]
    for i, (exp, name, color) in enumerate(zip(experiments, loss_names, colors)):
        hist = histories[i]
        if hist:
            ax.plot(hist.get('grad_norm', []), label=name, color=color, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitude (Learning Signal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Correlation Score
    ax = axes[1, 0]
    for i, (exp, name, color) in enumerate(zip(experiments, loss_names, colors)):
        hist = histories[i]
        if hist:
            ax.plot(hist.get('correlation', []), label=name, color=color, linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Perfect')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Correlation')
    ax.set_title('Cosine Similarity (Shape Match)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: MSE Loss
    ax = axes[1, 1]
    for i, (exp, name, color) in enumerate(zip(experiments, loss_names, colors)):
        hist = histories[i]
        if hist:
            ax.plot(hist.get('mse_loss', []), label=name, color=color, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Mean Squared Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 6: Mu Range Evolution
    ax = axes[1, 2]
    for i, (exp, name, color) in enumerate(zip(experiments, loss_names, colors)):
        hist = histories[i]
        if hist:
            mu_range = np.array(hist.get('mu_max', [])) - np.array(hist.get('mu_min', []))
            ax.plot(mu_range, label=name, color=color, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mu Range (Pa)')
    ax.set_title('Stiffness Variation (Max - Min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'bioqic_pielm/outputs/loss_function_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_file}")
    plt.close()


def print_summary():
    """Print summary statistics for each loss function."""
    experiments = [
        ('physical_box', 'Correlation'),
        ('physical_relative_l2', 'Relative L2'),
        ('physical_sobolev', 'Sobolev'),
        ('physical_mse', 'MSE'),
    ]
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Loss Type':<20} {'Final MuMSE':<15} {'Final Corr':<15} {'Mu Range':<15}")
    print("-"*70)
    
    for exp_name, loss_name in experiments:
        hist = load_results(exp_name)
        if hist:
            final_mu_mse = hist.get('mu_mse', [0])[-1]
            final_corr = hist.get('correlation', [0])[-1]
            final_mu_range = hist.get('mu_max', [0])[-1] - hist.get('mu_min', [0])[-1]
            
            print(f"{loss_name:<20} {final_mu_mse:<15.2e} {final_corr:<15.4f} {final_mu_range:<15.1f}")
    
    print("="*70)
    print("\nInterpretation:")
    print("  • Lower MuMSE = Better stiffness reconstruction")
    print("  • Higher Correlation = Better shape/phase match")
    print("  • Larger Mu Range = More spatial variation learned")
    print("  • Expected true range: ~7000 Pa (3000 to 10000 Pa)")
    print("="*70)


def main():
    """Run all experiments and compare results."""
    print("\n" + "="*70)
    print("LOSS FUNCTION COMPARISON FOR MRE INVERSION")
    print("="*70)
    print("\nTesting 4 loss functions:")
    print("  1. Correlation (cosine similarity)")
    print("  2. Relative L2 (normalized MSE)")
    print("  3. Sobolev (gradient-enhanced)")
    print("  4. MSE (standard baseline)")
    print("\nEach uses: Actuator BC, 1000 neurons, data_weight=10.0")
    print("="*70)
    
    experiments = [
        'physical_box',          # Correlation
        'physical_relative_l2',  # Relative L2
        'physical_sobolev',      # Sobolev
        'physical_mse',          # MSE
    ]
    
    # Run all experiments
    for exp in experiments:
        success = run_experiment(exp, subsample=5000)
        if not success:
            print(f"WARNING: {exp} failed, continuing with others...")
    
    # Visualize comparison
    print("\nGenerating comparison plots...")
    visualize_comparison()
    
    # Print summary
    print_summary()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nRecommendations:")
    print("  • Sobolev loss should show strongest gradients (best sensitivity)")
    print("  • Relative L2 normalizes amplitude issues")
    print("  • Correlation ignores amplitude completely")
    print("  • MSE baseline may struggle with amplitude mismatch")
    print("="*70)


if __name__ == "__main__":
    main()
