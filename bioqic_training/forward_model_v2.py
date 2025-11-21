"""
Fixed Forward MRE Model - Using approach folder's PIELM construction.

Key fix: Build overdetermined system [H_pde; H_bc; H_data] instead of forming
H^T H explicitly. This is more numerically stable.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Tuple

# Add approach folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'approach'))
from pielm_solver import pielm_solve


class ForwardMREModel(nn.Module):
    """
    Forward MRE model using PIELM with improved numerical stability.
    
    Matches the approach folder's construction:
    - Stack PDE, BC, and data constraints as rows
    - Let pielm_solve handle H^T H formation internally
    """
    
    def __init__(
        self,
        mu_network: nn.Module,
        n_wave_neurons: int = 100,
        input_dim: int = 3,
        omega_basis: float = 1.0,
        physics_mode: str = 'effective',
        rho_omega2_effective: float = 400.0,
        seed: Optional[int] = None
    ):
        """Initialize forward model matching approach folder style."""
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.mu_network = mu_network
        self.n_wave_neurons = n_wave_neurons
        self.input_dim = input_dim
        self.physics_mode = physics_mode
        self.rho_omega2_effective = rho_omega2_effective
        
        # Random wave basis: ω ~ N(0, omega_basis²)
        self.register_buffer(
            'omega_basis',
            torch.randn(input_dim, n_wave_neurons) * omega_basis
        )
        
        print(f"  Forward model initialized:")
        print(f"    Wave neurons: {n_wave_neurons}")
        print(f"    Input dim: {input_dim}D")
        print(f"    Physics mode: {physics_mode}")
        if physics_mode == 'effective':
            print(f"    ρω² (effective): {rho_omega2_effective}")
    
    def get_basis_and_laplacian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute wave basis and Laplacians.
        
        φ_i(x) = sin(ω_i · x)
        ∇²φ_i = -‖ω_i‖² φ_i
        """
        z = x @ self.omega_basis  # (N, M)
        phi = torch.sin(z)
        
        omega_sq = torch.sum(self.omega_basis ** 2, dim=0)  # (M,)
        phi_lap = -omega_sq.unsqueeze(0) * phi  # (N, M)
        
        return phi, phi_lap
    
    def build_pielm_system(
        self,
        mu: torch.Tensor,
        phi: torch.Tensor,
        phi_lap: torch.Tensor,
        rho_omega2: float,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        u_data: Optional[torch.Tensor] = None,
        data_indices: Optional[torch.Tensor] = None,
        data_weight: float = 0.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build PIELM system using approach folder's row-stacking method.
        
        System: [H_pde  ]     [0      ]
                [H_bc   ] c = [u_bc   ]
                [H_data ]     [u_data ]
        
        This is more stable than forming H^T H explicitly.
        """
        N, M = phi.shape
        
        if verbose:
            print(f"\n    Building PIELM system (approach-style):")
            print(f"      N collocation points: {N}")
            print(f"      M basis functions: {M}")
            rho_val = rho_omega2.item() if torch.is_tensor(rho_omega2) else rho_omega2
            print(f"      ρω²: {rho_val:.1f}")
        
        # PDE residual: μ∇²u + ρω²u = 0
        # With u = Σ c_i φ_i: μ∇²φ + ρω²φ = 0
        # So: (μ·∇²φ + ρω²·φ) c ≈ 0
        H_pde = mu * phi_lap + rho_omega2 * phi  # (N, M)
        b_pde = torch.zeros(N, 1, device=phi.device, dtype=phi.dtype)
        
        # Start with PDE rows
        H_rows = [H_pde]
        b_rows = [b_pde]
        
        if verbose:
            print(f"      PDE rows: {H_pde.shape[0]}")
        
        # Add boundary conditions
        if bc_indices is not None and len(bc_indices) > 0:
            H_bc = bc_weight * phi[bc_indices]  # (K, M)
            b_bc = bc_weight * u_bc_vals  # (K, 1)
            
            H_rows.append(H_bc)
            b_rows.append(b_bc)
            
            if verbose:
                print(f"      BC rows: {H_bc.shape[0]} (weight={bc_weight:.1f})")
        
        # Add data constraints
        if u_data is not None and data_weight > 0:
            if data_indices is not None:
                phi_data = phi[data_indices]
                u_data_subset = u_data[data_indices] if u_data.shape[0] == N else u_data
            else:
                phi_data = phi
                u_data_subset = u_data
            
            H_data = data_weight * phi_data  # (N_data, M)
            b_data = data_weight * u_data_subset  # (N_data, 1)
            
            H_rows.append(H_data)
            b_rows.append(b_data)
            
            if verbose:
                print(f"      Data rows: {H_data.shape[0]} (weight={data_weight:.1f})")
        
        # Stack all rows
        H = torch.cat(H_rows, dim=0)  # (N_total, M)
        b = torch.cat(b_rows, dim=0)  # (N_total, 1)
        
        if verbose:
            print(f"      Total system: H {H.shape}, b {b.shape}")
        
        return H, b
    
    def forward(
        self,
        x: torch.Tensor,
        rho_omega2: float,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        u_data: Optional[torch.Tensor] = None,
        data_indices: Optional[torch.Tensor] = None,
        data_weight: float = 0.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict displacement and stiffness."""
        # Predict stiffness
        mu_pred = self.mu_network(x)
        
        # Get wave basis
        phi, phi_lap = self.get_basis_and_laplacian(x)
        
        # Build system (approach-style)
        H, b = self.build_pielm_system(
            mu_pred, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_indices, data_weight,
            verbose
        )
        
        # Solve (pielm_solve handles H^T H internally)
        c = pielm_solve(H, b, verbose=verbose)
        
        # Reconstruct displacement
        u_pred = phi @ c
        
        return u_pred, mu_pred
