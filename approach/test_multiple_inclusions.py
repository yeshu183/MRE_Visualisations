"""Test MRE inversion with multiple stiffness inclusions.

This validates that the framework works for different ground truth patterns,
not just the single Gaussian bump.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import ForwardMREModel

# Test configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POINTS = 100
N_WAVE_NEURONS = 60  # Increased for better representation
ITERATIONS = 5000  # More iterations for convergence
LR = 0.01  # Moderate learning rate
RHO_OMEGA2 = 400.0
BC_WEIGHT = 200.0
NOISE_STD = 0.001
SEED = 42  # Different seed for variety

torch.manual_seed(SEED)


def generate_multiple_inclusions():
    """Generate stiffness field with two inclusions."""
    x = torch.linspace(0, 1, N_POINTS).reshape(-1, 1).to(DEVICE)
    
    # Two stiff inclusions at different locations
    inclusion1 = 1.5 * torch.exp(-200 * (x - 0.3)**2)  # Left inclusion
    inclusion2 = 0.8 * torch.exp(-150 * (x - 0.7)**2)  # Right inclusion
    mu_true = (1.0 + inclusion1 + inclusion2).to(DEVICE)
    
    bc_indices = torch.tensor([0, N_POINTS - 1], dtype=torch.long, device=DEVICE)
    u_bc_vals = torch.tensor([[0.0], [0.1]], device=DEVICE)
    
    # Generate data with same basis as training
    gen_model = ForwardMREModel(n_neurons_wave=N_WAVE_NEURONS, input_dim=1, seed=SEED).to(DEVICE)
    u_true, _ = gen_model.solve_given_mu(x, mu_true, bc_indices, u_bc_vals, RHO_OMEGA2, bc_weight=BC_WEIGHT)
    u_meas = u_true + NOISE_STD * torch.randn_like(u_true)
    
    return x, u_meas, mu_true, u_true, bc_indices, u_bc_vals


def train_test():
    print("=" * 70)
    print("TEST: Multiple Inclusions (Two Peaks)")
    print("=" * 70)
    
    x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_multiple_inclusions()
    
    model = ForwardMREModel(n_neurons_wave=N_WAVE_NEURONS, input_dim=1, seed=SEED).to(DEVICE)
    optimizer = torch.optim.Adam(model.mu_net.parameters(), lr=LR)
    # Learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)
    
    print(f"Ground truth mu range: [{mu_true.min():.3f}, {mu_true.max():.3f}]")
    print(f"Starting optimization with {ITERATIONS} iterations...\n")
    
    best_loss = float('inf')
    patience = 1500
    patience_counter = 0
    
    # Loss tracking
    history = {
        'data_loss': [],
        'grad_norm': [],
        'mu_min': [],
        'mu_max': [],
        'mu_mean': []
    }
    
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        u_pred, mu_pred = model(x, bc_indices, u_bc_vals, RHO_OMEGA2, bc_weight=BC_WEIGHT)
        
        loss_data = torch.mean((u_pred - u_meas) ** 2)
        
        # No TV regularization - let deep network learn structure
        loss = loss_data
        
        loss.backward()
        
        # Gradient clipping for stability with deep network
        torch.nn.utils.clip_grad_norm_(model.mu_net.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track losses
        grad_norm = sum(p.grad.norm().item()**2 for p in model.mu_net.parameters() if p.grad is not None)**0.5
        history['data_loss'].append(loss_data.item())
        history['grad_norm'].append(grad_norm)
        history['mu_min'].append(mu_pred.min().item())
        history['mu_max'].append(mu_pred.max().item())
        history['mu_mean'].append(mu_pred.mean().item())
        
        # Early stopping if loss plateaus
        if loss_data.item() < best_loss:
            best_loss = loss_data.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at iteration {i}: loss plateaued")
                break
        
        if i % 500 == 0 or i == ITERATIONS - 1:
            mu_min, mu_max, mu_mean = mu_pred.min().item(), mu_pred.max().item(), mu_pred.mean().item()
            print(f"Iter {i:4d}: DataLoss={loss_data.item():.6e}, GradNorm={grad_norm:.3e}")
            print(f"           Mu: min={mu_min:.3f}, max={mu_max:.3f}, mean={mu_mean:.3f}")
    
    # Compute final metrics
    final_loss = loss_data.item()
    mu_error = torch.mean((mu_pred.detach() - mu_true)**2).item()
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Final DataLoss: {final_loss:.6e}")
    print(f"  Mu MSE Error:   {mu_error:.6e}")
    print(f"  True mu range:  [{mu_true.min():.3f}, {mu_true.max():.3f}]")
    print(f"  Pred mu range:  [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
    
    # Visualization
    plot_comparison(x, u_meas, u_pred.detach(), u_true, mu_true, mu_pred.detach(), history)
    
    # Pass/Fail criteria
    if final_loss < 1e-4 and mu_error < 0.5:
        print(f"\n✅ TEST PASSED: Loss converged and mu reasonably recovered")
        return True
    else:
        print(f"\n❌ TEST FAILED: Loss or mu error too high")
        return False


def plot_comparison(x, u_meas, u_pred, u_true, mu_true, mu_pred, history):
    x_np = x.cpu().numpy()
    
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: Wave field and Stiffness reconstruction
    plt.subplot(2, 3, 1)
    plt.title("Wave Field Reconstruction", fontsize=12, fontweight='bold')
    plt.plot(x_np, u_true.cpu().numpy(), 'k', label='True (Clean)', linewidth=2)
    plt.plot(x_np, u_meas.cpu().numpy(), 'k--', label='Measured (Noisy)', alpha=0.7)
    plt.plot(x_np, u_pred.cpu().numpy(), 'r', label='Predicted', linewidth=2)
    plt.xlabel('Position x')
    plt.ylabel('Displacement u')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.title("Stiffness Reconstruction", fontsize=12, fontweight='bold')
    plt.plot(x_np, mu_true.cpu().numpy(), 'k', label='Ground Truth', linewidth=2)
    plt.plot(x_np, mu_pred.cpu().numpy(), 'b', label='Recovered', linewidth=2)
    plt.xlabel('Position x')
    plt.ylabel('Stiffness μ')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.title("Pointwise Reconstruction Error", fontsize=12, fontweight='bold')
    mu_error = (mu_pred - mu_true).cpu().numpy()
    plt.plot(x_np, mu_error, 'r', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Position x')
    plt.ylabel('Error (μ_pred - μ_true)')
    plt.grid(alpha=0.3)
    
    # Row 2: Loss curves
    iterations = range(len(history['data_loss']))
    
    plt.subplot(2, 3, 4)
    plt.title("Data Loss (MSE)", fontsize=12, fontweight='bold')
    plt.semilogy(iterations, history['data_loss'], 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.title("Gradient Norm", fontsize=12, fontweight='bold')
    plt.semilogy(iterations, history['grad_norm'], 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (log scale)')
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.title("Mu Statistics Over Training", fontsize=12, fontweight='bold')
    plt.plot(iterations, history['mu_min'], 'b-', linewidth=2, label='Min μ')
    plt.plot(iterations, history['mu_max'], 'r-', linewidth=2, label='Max μ')
    plt.plot(iterations, history['mu_mean'], 'g-', linewidth=2, label='Mean μ')
    plt.axhline(mu_true.min().item(), color='b', linestyle='--', alpha=0.5, label='True min')
    plt.axhline(mu_true.max().item(), color='r', linestyle='--', alpha=0.5, label='True max')
    plt.xlabel('Iteration')
    plt.ylabel('Stiffness μ')
    plt.legend(loc='best', fontsize=8)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('approach/test_multiple_inclusions_result.png', dpi=120)
    print(f"\n  Plot saved: approach/test_multiple_inclusions_result.png")


if __name__ == "__main__":
    success = train_test()
    exit(0 if success else 1)
