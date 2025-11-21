"""
Analyze training results and identify bottlenecks.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_training_history(experiment='baseline'):
    """Analyze training history to identify issues."""
    
    output_dir = Path(__file__).parent / 'outputs' / experiment
    
    # Load training history
    with open(output_dir / 'training_history.json', 'r') as f:
        history = json.load(f)
    
    # Convert to numpy arrays
    iteration = np.array(history['iteration'])
    loss_total = np.array(history['loss_total'])
    loss_data = np.array(history['loss_data'])
    loss_bc = np.array(history['loss_bc'])
    loss_tv = np.array(history['loss_tv'])
    grad_norm = np.array(history['grad_norm'])
    lr = np.array(history['lr'])
    
    print(f"\n{'='*80}")
    print(f"TRAINING ANALYSIS - {experiment.upper()}")
    print(f"{'='*80}")
    
    # 1. Loss convergence analysis
    print(f"\n📊 LOSS CONVERGENCE:")
    print(f"   Initial loss: {loss_total[0]:.6f}")
    print(f"   Final loss: {loss_total[-1]:.6f}")
    print(f"   Best loss: {loss_total.min():.6f} at iter {iteration[loss_total.argmin()]}")
    print(f"   Reduction: {(loss_total[0] - loss_total[-1]) / loss_total[0] * 100:.2f}%")
    
    # 2. Data loss analysis (most important for reconstruction)
    print(f"\n📉 DATA LOSS (MSE):")
    print(f"   Initial: {loss_data[0]:.6e}")
    print(f"   Final: {loss_data[-1]:.6e}")
    print(f"   Best: {loss_data.min():.6e} at iter {iteration[loss_data.argmin()]}")
    print(f"   Reduction: {(loss_data[0] - loss_data[-1]) / loss_data[0] * 100:.2f}%")
    print(f"   ⚠️  PLATEAU: {loss_data[-1]:.6e} (should be < 1e-6 for good fit)")
    
    # 3. Check if losses are balanced
    print(f"\n⚖️  LOSS BALANCE (at iteration {iteration[-1]}):")
    data_weighted = loss_data[-1] * 100  # data_weight = 100
    bc_weighted = loss_bc[-1] * 10  # bc_weight = 10
    tv_weighted = loss_tv[-1] * 0.001  # tv_weight = 0.001
    
    print(f"   Data loss × 100: {data_weighted:.6e}")
    print(f"   BC loss × 10: {bc_weighted:.6e}")
    print(f"   TV loss × 0.001: {tv_weighted:.6e}")
    print(f"   Ratio (Data:BC:TV): {data_weighted:.1f}:{bc_weighted:.1f}:{tv_weighted:.4f}")
    
    if data_weighted < bc_weighted:
        print(f"   ⚠️  BC loss dominates! Data fit may be compromised.")
    
    # 4. Gradient flow analysis
    print(f"\n🌊 GRADIENT FLOW:")
    print(f"   Initial grad norm: {grad_norm[0]:.3e}")
    print(f"   Final grad norm: {grad_norm[-1]:.3e}")
    print(f"   Mean grad norm: {grad_norm.mean():.3e}")
    print(f"   Max grad norm: {grad_norm.max():.3e}")
    
    if grad_norm[-1] > 1.0:
        print(f"   ⚠️  Large gradients persisting - may need better initialization or scaling")
    
    # 5. Learning rate schedule
    print(f"\n📈 LEARNING RATE:")
    print(f"   Initial: {lr[0]:.4e}")
    print(f"   Final: {lr[-1]:.4e}")
    lr_changes = np.where(np.diff(lr) != 0)[0]
    print(f"   LR decays: {len(lr_changes)} times")
    
    # 6. Oscillation detection
    loss_diffs = np.diff(loss_total)
    oscillations = np.sum((loss_diffs[:-1] > 0) & (loss_diffs[1:] < 0))
    print(f"\n🌊 TRAINING STABILITY:")
    print(f"   Loss oscillations: {oscillations} times")
    print(f"   Loss std (last 500 iters): {loss_total[-500:].std():.6e}")
    
    if oscillations > len(loss_total) * 0.3:
        print(f"   ⚠️  High oscillation - consider lowering learning rate")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Total loss
    ax = axes[0, 0]
    ax.plot(iteration, loss_total, 'b-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Evolution')
    ax.grid(alpha=0.3)
    
    # Plot 2: Component losses
    ax = axes[0, 1]
    ax.plot(iteration, loss_data, 'r-', label='Data', alpha=0.7)
    ax.plot(iteration, loss_bc, 'g-', label='BC', alpha=0.7)
    ax.plot(iteration, loss_tv * 1000, 'b-', label='TV × 1000', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Data loss closeup
    ax = axes[0, 2]
    ax.plot(iteration, loss_data, 'r-', linewidth=1)
    ax.axhline(1e-4, color='green', linestyle='--', label='Good fit (1e-4)')
    ax.axhline(1e-6, color='blue', linestyle='--', label='Excellent fit (1e-6)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Data MSE')
    ax.set_title('Data Loss (Critical for Reconstruction)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Gradient norm
    ax = axes[1, 0]
    ax.plot(iteration, grad_norm, 'purple', alpha=0.7, linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Flow')
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 5: Learning rate
    ax = axes[1, 1]
    ax.plot(iteration, lr, 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 6: Loss components weighted
    ax = axes[1, 2]
    ax.plot(iteration, loss_data * 100, 'r-', label='Data × 100', alpha=0.7)
    ax.plot(iteration, loss_bc * 10, 'g-', label='BC × 10', alpha=0.7)
    ax.plot(iteration, loss_tv * 0.001 * 1e6, 'b-', label='TV × 0.001 × 1e6', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weighted Loss')
    ax.set_title('Weighted Loss Components')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: {save_path}")
    plt.show()
    
    # 7. Key findings summary
    print(f"\n{'='*80}")
    print(f"🔍 KEY FINDINGS:")
    print(f"{'='*80}")
    
    findings = []
    
    # Data loss plateau
    if loss_data[-1] > 1e-4:
        findings.append("❌ Data loss plateaued at {:.2e} (should be < 1e-4)".format(loss_data[-1]))
        findings.append("   → Forward solver may not be accurate enough")
        findings.append("   → Or: mu_network lacks expressiveness")
    
    # Loss balance
    if data_weighted < bc_weighted:
        findings.append("❌ BC loss dominates over data loss")
        findings.append("   → Reduce bc_weight or increase data_weight")
    
    # Gradient issues
    if grad_norm[-1] > 10:
        findings.append("⚠️  Large gradients ({:.1f}) - optimization struggling".format(grad_norm[-1]))
        findings.append("   → Check gradient flow through forward solver")
    
    # TV regularization
    if tv_weighted > data_weighted * 0.1:
        findings.append("⚠️  TV regularization too strong")
        findings.append("   → May be over-smoothing the solution")
    
    for finding in findings:
        print(finding)
    
    print(f"\n{'='*80}")
    print(f"💡 RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    recommendations = []
    
    if loss_data[-1] > 1e-4:
        recommendations.append("1. Increase data_weight from 100 to 500-1000")
        recommendations.append("2. Add more wave neurons (100 → 200-300)")
        recommendations.append("3. Try deeper mu_network (3 layers → 5 layers)")
    
    if data_weighted < bc_weighted:
        recommendations.append("4. Reduce BC weight from 10 to 1-5")
    
    if grad_norm.mean() > 5:
        recommendations.append("5. Lower learning rate (0.01 → 0.001)")
        recommendations.append("6. Use gradient clipping more aggressively")
    
    recommendations.append("7. Try 'actuator' boundary strategy (physics-informed)")
    recommendations.append("8. Test with more training data (1000 → 2000-5000 points)")
    
    for i, rec in enumerate(recommendations, 1):
        if isinstance(rec, str) and rec[0].isdigit():
            print(f"   {rec}")
        else:
            print(f"   {rec}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='baseline')
    args = parser.parse_args()
    
    analyze_training_history(args.experiment)
