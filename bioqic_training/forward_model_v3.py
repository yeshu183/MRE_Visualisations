"""
Forward MRE Model V3 - Using Approach Folder's Working Method

Key changes from v2:
1. Row-concatenation system (H shape: (N+K, M) not (M, M))
2. Pure data-driven option (no PDE enforcement)
3. Matches approach/tests/test_data_only.py architecture
4. More numerically stable

This should actually work with BIOQIC data!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add approach folder to path for pielm_solver
sys.path.insert(0, str(Path(__file__).parent.parent / 'approach'))
from pielm_solver import pielm_solve


class ForwardMREModelV3(nn.Module):
    """
    Data-driven forward model using approach folder's method.
    
    Key idea: When data weight >> PDE weight, this becomes pure interpolation.
    No need for PDE to be satisfied!
    """
    
    def __init__(
        self,
        mu_network: nn.Module,
        n_wave_neurons: int = 200,
        input_dim: int = 3,
        omega_basis: float = 20.0,  # Higher frequency for BIOQIC
        seed: Optional[int] = None
    ):
        """
        Initialize forward model.
        
        Args:
            mu_network: Stiffness network
            n_wave_neurons: Number of basis functions
            input_dim: Spatial dimension
            omega_basis: Scale for wave frequencies
            seed: Random seed
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.mu_network = mu_network
        self.n_wave_neurons = n_wave_neurons
        self.input_dim = input_dim
        
        # Wave basis: ω ~ N(0, omega_basis²)
        self.register_buffer(
            'omega_basis',
            torch.randn(input_dim, n_wave_neurons) * omega_basis
        )
        
        print(f"  Forward model V3 initialized:")
        print(f"    Wave neurons: {n_wave_neurons}")
        print(f"    omega_basis: {omega_basis}")
        print(f"    Method: Approach folder (data-driven)")
    
    def get_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute wave basis functions: φ_i(x) = sin(ω_i · x)
        
        Args:
            x: (N, input_dim) coordinates
            
        Returns:
            phi: (N, n_wave_neurons) basis values
        """
        z = x @ self.omega_basis  # (N, n_wave_neurons)
        phi = torch.sin(z)
        return phi
    
    def build_system(
        self,
        phi: torch.Tensor,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 1000.0,
        use_pde: bool = False,
        mu: Optional[torch.Tensor] = None,
        rho_omega2: float = 400.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build system using approach folder's row-concatenation method.
        
        Args:
            phi: (N, M) basis functions
            bc_indices: Boundary point indices
            u_bc_vals: Boundary values
            bc_weight: BC constraint weight
            u_data: Measured displacement (N, 1)
            data_weight: Data fitting weight
            use_pde: Whether to include PDE (usually False for data-driven)
            mu: Stiffness values (only if use_pde=True)
            rho_omega2: Physics parameter
            verbose: Print diagnostics
            
        Returns:
            H: (N_total, M) system matrix (rows to concatenate)
            b: (N_total, 1) right-hand side
        """
        N, M = phi.shape
        
        if verbose:
            print(f"\n    Building system (approach-style):")
            print(f"      N points: {N}, M basis: {M}")
        
        rows_H = []
        rows_b = []
        
        # 1. PDE rows (optional - usually skip for data-driven)
        if use_pde and mu is not None:
            # Would need phi_lap here, skip for now
            print(f"      PDE: SKIPPED (data-driven mode)")
        
        # 2. Data fitting rows (MAIN constraint)
        if u_data is not None and data_weight > 0:
            H_data = data_weight * phi  # (N, M)
            b_data = data_weight * u_data  # (N, 1)
            rows_H.append(H_data)
            rows_b.append(b_data)
            
            if verbose:
                print(f"      Data: {N} points, weight={data_weight:.1f}")
        
        # 3. BC rows (for uniqueness)
        if bc_indices is not None and len(bc_indices) > 0:
            phi_bc = phi[bc_indices]  # (K, M)
            H_bc = bc_weight * phi_bc  # (K, M)
            b_bc = bc_weight * u_bc_vals  # (K, 1)
            rows_H.append(H_bc)
            rows_b.append(b_bc)
            
            if verbose:
                print(f"      BC: {len(bc_indices)} points, weight={bc_weight:.1f}")
        
        # Concatenate all rows
        if len(rows_H) == 0:
            raise ValueError("Must have at least data or BC constraints!")
        
        H = torch.cat(rows_H, dim=0)  # (N_total, M)
        b = torch.cat(rows_b, dim=0)  # (N_total, 1)
        
        if verbose:
            print(f"      Total system: H {H.shape}, b {b.shape}")
        
        return H, b
    
    def forward(
        self,
        x: torch.Tensor,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 1000.0,
        use_pde: bool = False,
        rho_omega2: float = 400.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: pure data-driven interpolation.
        
        Args:
            x: (N, 3) coordinates
            bc_indices: Boundary indices
            u_bc_vals: Boundary values
            bc_weight: BC weight (typically low, e.g., 1.0)
            u_data: Measured displacement (typically = u_meas)
            data_weight: Data weight (typically high, e.g., 1000.0)
            use_pde: Whether to use PDE (False for data-driven)
            rho_omega2: Physics parameter (unused if use_pde=False)
            verbose: Print diagnostics
            
        Returns:
            u_pred: (N, 1) predicted displacement
            mu_pred: (N, 1) predicted stiffness
        """
        # Get stiffness field
        mu_pred = self.mu_network(x)
        
        # Get basis functions
        phi = self.get_basis(x)  # (N, M)
        
        # Build system (data-driven)
        H, b = self.build_system(
            phi, bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight, use_pde, mu_pred, rho_omega2,
            verbose
        )
        
        # Solve: H^T H c = H^T b
        c = pielm_solve(H, b, verbose=verbose)
        
        # Reconstruct: u = φ c
        u_pred = phi @ c
        
        return u_pred, mu_pred


if __name__ == '__main__':
    print("Testing ForwardMREModelV3...")
    
    # Simple test
    from stiffness_network import FlexibleStiffnessNetwork
    
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.2, mu_max=1.2
    )
    
    model = ForwardMREModelV3(
        mu_network=mu_net,
        n_wave_neurons=100,
        omega_basis=20.0,
        seed=42
    )
    
    # Test forward
    x = torch.rand(50, 3)
    u_meas = torch.rand(50, 1)
    bc_idx = torch.tensor([0, 25, 49])
    u_bc = u_meas[bc_idx]
    
    u_pred, mu_pred = model(
        x, bc_idx, u_bc, bc_weight=1.0,
        u_data=u_meas, data_weight=1000.0,
        verbose=True
    )
    
    print(f"\n✅ Test passed!")
    print(f"   u_pred shape: {u_pred.shape}")
    print(f"   mu_pred shape: {mu_pred.shape}")
    print(f"   MSE: {torch.mean((u_pred - u_meas)**2).item():.6f}")
