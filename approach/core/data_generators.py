"""Synthetic data generation for MRE inverse problems.

Provides various ground truth stiffness distributions and corresponding
wave field solutions for testing the inversion framework.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ForwardMREModel


def generate_gaussian_bump(n_points=100, n_wave_neurons=60, device='cpu', seed=0):
    """Generate data with single Gaussian inclusion.
    
    Args:
        n_points: Number of spatial points
        n_wave_neurons: Number of wave basis functions
        device: 'cpu' or 'cuda'
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (x, u_meas, mu_true, u_true, bc_indices, u_bc_vals)
    """
    x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
    
    # Ground truth: Gaussian bump (1.0 baseline, peak at 3.0)
    mu_true = 1.0 + 2.0 * torch.exp(-((x - 0.5) ** 2) / (2 * 0.1**2))
    
    return generate_synthetic_data(
        x, mu_true, n_wave_neurons, device, seed,
        rho_omega2=400.0, bc_weight=200.0, noise_std=0.001
    )


def generate_multiple_inclusions(n_points=100, n_wave_neurons=60, device='cpu', seed=42):
    """Generate data with two Gaussian inclusions.
    
    Args:
        n_points: Number of spatial points
        n_wave_neurons: Number of wave basis functions
        device: 'cpu' or 'cuda'
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (x, u_meas, mu_true, u_true, bc_indices, u_bc_vals)
    """
    x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
    
    # Ground truth: Two Gaussian peaks
    peak1 = 1.5 * torch.exp(-((x - 0.3) ** 2) / (2 * 0.08**2))
    peak2 = 0.8 * torch.exp(-((x - 0.7) ** 2) / (2 * 0.08**2))
    mu_true = 1.0 + peak1 + peak2
    
    return generate_synthetic_data(
        x, mu_true, n_wave_neurons, device, seed,
        rho_omega2=400.0, bc_weight=200.0, noise_std=0.001
    )


def generate_step_function(n_points=100, n_wave_neurons=60, device='cpu', seed=123):
    """Generate data with sharp stiffness transition.
    
    Args:
        n_points: Number of spatial points
        n_wave_neurons: Number of wave basis functions
        device: 'cpu' or 'cuda'
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (x, u_meas, mu_true, u_true, bc_indices, u_bc_vals)
    """
    x = torch.linspace(0, 1, n_points, device=device).reshape(-1, 1)
    
    # Ground truth: Smoothed step function (layer boundary)
    # Use tanh for smooth but sharp transition
    steepness = 50  # Controls sharpness
    mu_true = 1.0 + 0.75 * (torch.tanh(steepness * (x - 0.5)) + 1)
    
    return generate_synthetic_data(
        x, mu_true, n_wave_neurons, device, seed,
        rho_omega2=400.0, bc_weight=200.0, noise_std=0.001
    )


def generate_synthetic_data(x, mu_true, n_wave_neurons, device, seed,
                            rho_omega2=400.0, bc_weight=200.0, noise_std=0.001):
    """Generate synthetic measurement data from ground truth stiffness.
    
    This function:
    1. Creates a forward model with given stiffness
    2. Solves the PDE to get clean wave field
    3. Adds noise to simulate measurements
    4. Extracts boundary conditions
    
    Args:
        x: Spatial coordinates (N, 1)
        mu_true: Ground truth stiffness (N, 1) or (N,)
        n_wave_neurons: Number of wave basis functions
        device: 'cpu' or 'cuda'
        seed: Random seed for wave basis
        rho_omega2: PDE parameter (ρω²)
        bc_weight: Boundary condition weight
        noise_std: Standard deviation of measurement noise
        
    Returns:
        tuple: (x, u_meas, mu_true, u_true, bc_indices, u_bc_vals)
            - x: Spatial points (N, 1)
            - u_meas: Noisy measurements (N, 1)
            - mu_true: Ground truth stiffness (N, 1)
            - u_true: Clean wave field (N, 1)
            - bc_indices: Boundary point indices
            - u_bc_vals: Boundary condition values
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Ensure mu_true is (N, 1) shaped
    if mu_true.dim() == 1:
        mu_true = mu_true.reshape(-1, 1)
    elif mu_true.shape[1] != 1:
        mu_true = mu_true.reshape(-1, 1)
    
    # Create forward model with deterministic seed
    gen_model = ForwardMREModel(
        n_neurons_wave=n_wave_neurons,
        input_dim=1,
        seed=seed
    ).to(device)
    
    # Boundary conditions: sinusoidal excitation at boundaries (non-zero!)
    # This simulates the vibration source in MRE experiments
    bc_indices = torch.tensor([0, len(x) - 1], dtype=torch.long, device=device)
    u_bc_vals = torch.tensor([[0.01], [0.0]], device=device)  # Left boundary vibrates with 10x stronger amplitude
    
    # Solve forward problem with ground truth mu
    with torch.no_grad():
        u_true, _ = gen_model.solve_given_mu(
            x, mu_true, bc_indices, u_bc_vals,
            rho_omega2, bc_weight
        )
    
    # Add measurement noise
    u_meas = u_true + noise_std * torch.randn_like(u_true)
    
    return x, u_meas, mu_true, u_true, bc_indices, u_bc_vals
