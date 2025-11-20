import torch
import numpy as np
import matplotlib.pyplot as plt
from models import ForwardMREModel


# Configuration
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models import ForwardMREModel

# --------------------------------------------------
# Configuration loading (medium improvement)
# --------------------------------------------------
DEFAULT_CFG = {
    "n_points": 100,
    "n_wave_neurons": 80,
    "iterations": 2000,
    "lr": 0.01,
    "rho_omega2": 400.0,
    "noise_std": 0.01,
    "bc_weight": 1000.0,
    "seed": 0
}

CFG_PATH = Path(__file__).parent / "config_forward.json"
if CFG_PATH.exists():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        CFG = {**DEFAULT_CFG, **json.load(f)}
else:
    CFG = DEFAULT_CFG

torch.manual_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_POINTS = CFG["n_points"]
N_WAVE_NEURONS = CFG["n_wave_neurons"]
ITERATIONS = CFG["iterations"]
LR = CFG["lr"]
RHO_OMEGA2 = CFG["rho_omega2"]
BC_WEIGHT = CFG["bc_weight"]
NOISE_STD = CFG["noise_std"]


def generate_consistent_synthetic():
    """Generate synthetic stiffness and wavefield consistent with PDE by solving.

    Returns:
        x: (N,1) coordinates
        u_meas: (N,1) noisy measured displacement
        mu_true: (N,1) ground-truth stiffness field
        u_true: (N,1) clean PDE-consistent displacement
        bc_indices: (K,) boundary indices
        u_bc_vals: (K,1) boundary displacements used during generation
    """
    x = torch.linspace(0, 1, N_POINTS).reshape(-1, 1).to(DEVICE)
    # Ground truth stiffness: Gaussian inclusion
    mu_true = (1.0 + 2.0 * torch.exp(-100 * (x - 0.5) ** 2)).to(DEVICE)

    # Boundary conditions: Assign non-zero values to break degeneracy
    bc_indices = torch.tensor([0, N_POINTS - 1], dtype=torch.long, device=DEVICE)
    u_bc_vals = torch.tensor([[0.0], [0.1]], device=DEVICE)  # Non-homogeneous BC

    # CRITICAL FIX: Use SAME seed as training model so basis functions match
    # This way, if mu is correct, the model CAN reproduce the data perfectly
    # Otherwise there's always a representation error that masks the mu error
    gen_model = ForwardMREModel(n_neurons_wave=N_WAVE_NEURONS, input_dim=1, seed=CFG["seed"]).to(DEVICE)
    u_true, _C = gen_model.solve_given_mu(x, mu_true, bc_indices, u_bc_vals, RHO_OMEGA2, bc_weight=BC_WEIGHT)

    # Add measurement noise
    u_meas = u_true + NOISE_STD * torch.randn_like(u_true)
    return x, u_meas, mu_true, u_true, bc_indices, u_bc_vals


def train():
    # 1. Data generation (consistent with PDE)
    x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_consistent_synthetic()

    # 2. Model
    model = ForwardMREModel(n_neurons_wave=N_WAVE_NEURONS, input_dim=1, seed=CFG["seed"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.mu_net.parameters(), lr=LR)
    
    # Learning rate scheduler and early stopping (for stability with real data)
    lr_decay_step = CFG.get("lr_decay_step", 1000)
    lr_decay_gamma = CFG.get("lr_decay_gamma", 0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    
    early_stopping_patience = CFG.get("early_stopping_patience", 800)
    grad_clip_max_norm = CFG.get("grad_clip_max_norm", 1.0)
    
    best_loss = float('inf')
    patience_counter = 0

    print(f"Starting Forward Optimization on {DEVICE} with config: {CFG}")
    
    # Loss tracking
    history = {
        'data_loss': [],
        'tv_loss': [],
        'total_loss': [],
        'grad_norm': [],
        'mu_min': [],
        'mu_max': [],
        'mu_mean': []
    }

    for i in range(ITERATIONS):
        optimizer.zero_grad()
        verbose = (i == 0)  # Enable diagnostics on first iteration
        u_pred, mu_pred = model(x, bc_indices, u_bc_vals, RHO_OMEGA2, bc_weight=BC_WEIGHT, verbose=verbose)
        
        # Primary loss: match measured wavefield
        loss_data = torch.mean((u_pred - u_meas) ** 2)
        
        # Total Variation (TV) regularization - promotes piecewise constant solutions
        # Good for both smooth features and sharp edges
        tv_loss = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))
        
        # Combined loss (TV weight configurable)
        tv_weight = CFG.get("tv_weight", 0.0)  # Default 0 for backward compatibility
        loss_total = loss_data + tv_weight * tv_loss
        
        loss_total.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.mu_net.parameters(), max_norm=grad_clip_max_norm)
        
        # Track losses
        grad_norm = sum(p.grad.norm().item()**2 for p in model.mu_net.parameters() if p.grad is not None)**0.5
        history['data_loss'].append(loss_data.item())
        history['tv_loss'].append(tv_loss.item())
        history['total_loss'].append(loss_total.item())
        history['grad_norm'].append(grad_norm)
        history['mu_min'].append(mu_pred.min().item())
        history['mu_max'].append(mu_pred.max().item())
        history['mu_mean'].append(mu_pred.mean().item())
        
        # Gradient monitoring
        if i <= 2 or i % 200 == 0:
            mu_min, mu_max, mu_mean = mu_pred.min().item(), mu_pred.max().item(), mu_pred.mean().item()
            if tv_weight > 0:
                print(f"Iter {i}: DataLoss={loss_data.item():.6e}, TVLoss={tv_loss.item():.6e}, GradNorm={grad_norm:.3e}")
            else:
                print(f"Iter {i}: DataLoss={loss_data.item():.6e}, GradNorm={grad_norm:.3e}")
            print(f"         Mu: min={mu_min:.3f}, max={mu_max:.3f}, mean={mu_mean:.3f} (true range: 1.0-3.0)")
        
        optimizer.step()
        scheduler.step()
        
        # Early stopping if loss plateaus/increases
        if loss_data.item() < best_loss:
            best_loss = loss_data.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping at iteration {i}: loss plateaued for {early_stopping_patience} iterations")
                break
        
        if i % 200 == 0 or i == ITERATIONS - 1:
            pass  # Already tracked in history

    plot_results(x, u_meas, u_pred.detach(), mu_true, mu_pred.detach(), u_true, history)


def plot_results(x, u_meas, u_pred, mu_true, mu_pred, u_true, history):
    x_np = x.cpu().numpy()
    
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: Wave field and Stiffness reconstruction
    plt.subplot(2, 3, 1)
    plt.title("Wave Field", fontsize=12, fontweight='bold')
    plt.plot(x_np, u_true.cpu().numpy(), 'k', label='True (PDE)', linewidth=2)
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
    plt.title("Pointwise Stiffness Error", fontsize=12, fontweight='bold')
    error = (mu_pred.cpu().numpy() - mu_true.cpu().numpy())
    plt.plot(x_np, error, 'r', linewidth=2)
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
    plt.title("Gradient Norm & TV Loss", fontsize=12, fontweight='bold')
    ax1 = plt.gca()
    ax1.semilogy(iterations, history['grad_norm'], 'g-', linewidth=2, label='Grad Norm')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gradient Norm (log scale)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(alpha=0.3)
    
    if max(history['tv_loss']) > 1e-10:
        ax2 = ax1.twinx()
        ax2.semilogy(iterations, history['tv_loss'], 'orange', linewidth=2, label='TV Loss')
        ax2.set_ylabel('TV Loss (log scale)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
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
    plt.savefig('approach/forward_optimization_result.png', dpi=120)
    print(f"\n  Plots saved: approach/forward_optimization_result.png")
    #plt.show()


if __name__ == "__main__":
    train()