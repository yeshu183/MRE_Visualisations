"""Training and evaluation functions for inverse problems.

Provides general-purpose training loop and evaluation metrics
that can be configured via config dictionary.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ForwardMREModel


def train_inverse_problem(x, u_meas, mu_true, bc_indices, u_bc_vals, config, device='cpu'):
    """Train inverse problem to recover stiffness from wave measurements.
    
    Args:
        x: Spatial coordinates (N, 1)
        u_meas: Measured wave field (N, 1)
        mu_true: Ground truth stiffness (N, 1) - for monitoring only
        bc_indices: Boundary condition indices
        u_bc_vals: Boundary condition values
        config: Configuration dictionary with keys:
            - n_wave_neurons: Number of wave basis functions
            - iterations: Number of training iterations
            - lr: Learning rate
            - lr_decay_step: Steps between LR decay
            - lr_decay_gamma: LR decay factor
            - rho_omega2: PDE parameter
            - bc_weight: Boundary condition weight
            - tv_weight: Total variation regularization weight
            - grad_clip_max_norm: Gradient clipping threshold
            - early_stopping_patience: Patience for early stopping
            - seed: Random seed
        device: 'cpu' or 'cuda'
        
    Returns:
        tuple: (model, history)
            - model: Trained ForwardMREModel
            - history: Dictionary of training metrics
    """
    # Extract config
    n_wave_neurons = config['n_wave_neurons']
    iterations = config['iterations']
    lr = config['lr']
    lr_decay_step = config.get('lr_decay_step', 1000)
    lr_decay_gamma = config.get('lr_decay_gamma', 0.9)
    rho_omega2 = config['rho_omega2']
    bc_weight = config.get('bc_weight', 0.0)
    data_weight = config.get('data_weight', 0.0)
    tv_weight = config.get('tv_weight', 0.0)
    grad_clip_max_norm = config.get('grad_clip_max_norm', 1.0)
    early_stopping_patience = config.get('early_stopping_patience', 1000)
    seed = config['seed']
    
    # Initialize model
    model = ForwardMREModel(
        n_neurons_wave=n_wave_neurons,
        input_dim=1,
        seed=seed
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.mu_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
    )
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # Loss tracking
    history = {
        'data_loss': [],
        'tv_loss': [],
        'total_loss': [],
        'grad_norm': [],
        'mu_min': [],
        'mu_max': [],
        'mu_mean': [],
        'mu_mse': []
    }
    
    print(f"Starting training with {iterations} iterations...")
    if data_weight > 0:
        print(f"Using data constraints (weight={data_weight})")
    if bc_weight > 0:
        print(f"Using boundary constraints (weight={bc_weight})")
    if tv_weight > 0:
        print(f"Using TV regularization (weight={tv_weight})")
    print()
    
    # Training loop
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass with data constraints
        verbose = (i == 0)
        u_pred, mu_pred = model(x, bc_indices, u_bc_vals, rho_omega2, 
                               bc_weight=bc_weight, 
                               u_data=u_meas,
                               data_weight=data_weight,
                               verbose=verbose)
        
        # Data loss
        loss_data = torch.mean((u_pred - u_meas) ** 2)
        
        # Compute PDE residual for monitoring
        phi, phi_lap = model.get_basis_and_laplacian(x)
        pde_residual = mu_pred * phi_lap + rho_omega2 * phi
        _, C_u = model.solve_given_mu(x, mu_pred, bc_indices, u_bc_vals, rho_omega2, bc_weight, 
                                       u_data=u_meas, data_weight=data_weight)
        pde_loss_val = torch.mean((pde_residual @ C_u) ** 2).item()
        
        # TV regularization
        tv_loss = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))
        
        loss_total = loss_data + tv_weight * tv_loss
        
        # Backward pass
        loss_total.backward()
        if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.mu_net.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        grad_norm = sum(p.grad.norm().item()**2 for p in model.mu_net.parameters() 
                       if p.grad is not None)**0.5
        mu_mse = torch.mean((mu_pred.detach() - mu_true)**2).item()
        
        history['data_loss'].append(loss_data.item())
        history['tv_loss'].append(tv_loss.item())
        history['total_loss'].append(loss_total.item())
        history['grad_norm'].append(grad_norm)
        history['mu_min'].append(mu_pred.min().item())
        history['mu_max'].append(mu_pred.max().item())
        history['mu_mean'].append(mu_pred.mean().item())
        history['mu_mse'].append(mu_mse)
        
        # Logging
        if i % 500 == 0 or i == iterations - 1:
            mu_min, mu_max, mu_mean = mu_pred.min().item(), mu_pred.max().item(), mu_pred.mean().item()
            mu_true_min, mu_true_max = mu_true.min().item(), mu_true.max().item()
            print(f"Iter {i:4d}: PDE={pde_loss_val:.3e}, Data={loss_data.item():.3e}, MuMSE={mu_mse:.3e}, Grad={grad_norm:.3e}")
            print(f"           mu=[{mu_min:.3f}, {mu_max:.3f}], target=[{mu_true_min:.3f}, {mu_true_max:.3f}]")
            if tv_weight > 0:
                print(f"           TV={tv_loss.item():.3e}")
        
        # Early stopping
        if loss_data.item() < best_loss:
            best_loss = loss_data.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping at iteration {i}: loss plateaued for {early_stopping_patience} iterations")
                break
    
    return model, history


def evaluate_reconstruction(mu_pred, mu_true, loss_final, verbose=True):
    """Evaluate reconstruction quality.
    
    Args:
        mu_pred: Predicted stiffness (N, 1)
        mu_true: Ground truth stiffness (N, 1)
        loss_final: Final data loss value
        verbose: Whether to print results
        
    Returns:
        dict: Evaluation metrics
    """
    mu_mse = torch.mean((mu_pred - mu_true)**2).item()
    mu_mae = torch.mean(torch.abs(mu_pred - mu_true)).item()
    mu_max_error = torch.max(torch.abs(mu_pred - mu_true)).item()
    
    # Relative errors
    mu_range = mu_true.max().item() - mu_true.min().item()
    relative_mse = mu_mse / (mu_range**2) if mu_range > 0 else float('inf')
    relative_mae = mu_mae / mu_range if mu_range > 0 else float('inf')
    
    metrics = {
        'data_loss': loss_final,
        'mu_mse': mu_mse,
        'mu_mae': mu_mae,
        'mu_max_error': mu_max_error,
        'relative_mse': relative_mse,
        'relative_mae': relative_mae,
        'true_min': mu_true.min().item(),
        'true_max': mu_true.max().item(),
        'pred_min': mu_pred.min().item(),
        'pred_max': mu_pred.max().item()
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS:")
        print(f"  Final Data Loss: {loss_final:.6e}")
        print(f"  Mu MSE:          {mu_mse:.6e}")
        print(f"  Mu MAE:          {mu_mae:.6e}")
        print(f"  Mu Max Error:    {mu_max_error:.6e}")
        print(f"  Relative MSE:    {relative_mse:.6e}")
        print(f"  Relative MAE:    {relative_mae:.6e}")
        print(f"  True mu range:   [{metrics['true_min']:.3f}, {metrics['true_max']:.3f}]")
        print(f"  Pred mu range:   [{metrics['pred_min']:.3f}, {metrics['pred_max']:.3f}]")
        print(f"{'='*70}")
    
    return metrics
