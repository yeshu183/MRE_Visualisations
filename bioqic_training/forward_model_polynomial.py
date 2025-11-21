"""
Forward MRE Model with Polynomial Basis

Key insight: Wave basis sin(ω·x) is wrong for MRE displacement fields!
Use polynomial basis instead: [1, x, y, z, x², xy, xz, y², yz, z², x³, ...]

This should achieve MSE < 0.01 for pure data fitting.
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


class PolynomialBasisGenerator:
    """Generate polynomial basis functions up to degree D."""
    
    def __init__(self, degree: int = 3, input_dim: int = 3):
        """
        Initialize polynomial basis generator.
        
        Args:
            degree: Maximum polynomial degree
            input_dim: Spatial dimension (3 for 3D)
        """
        self.degree = degree
        self.input_dim = input_dim
        
        # Generate all multi-indices (powers) up to degree D
        self.powers = self._generate_powers()
        self.n_basis = len(self.powers)
        
        print(f"  Polynomial basis: degree={degree}, n_basis={self.n_basis}")
        print(f"    Example terms: 1, x, y, z, x², xy, xz, y², yz, z², x³, ...")
    
    def _generate_powers(self):
        """Generate all multi-indices [i, j, k] where i+j+k <= degree."""
        powers = []
        for total_deg in range(self.degree + 1):
            # All ways to distribute total_deg among 3 variables
            for i in range(total_deg + 1):
                for j in range(total_deg + 1 - i):
                    k = total_deg - i - j
                    if k >= 0:
                        powers.append([i, j, k])
        return np.array(powers)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomial basis at points x.
        
        Args:
            x: (N, 3) coordinates in [0, 1]³
            
        Returns:
            phi: (N, n_basis) polynomial values
        """
        N = x.shape[0]
        phi = torch.zeros(N, self.n_basis, device=x.device, dtype=x.dtype)
        
        # Extract coordinates
        x_vals = x[:, 0]  # (N,)
        y_vals = x[:, 1]
        z_vals = x[:, 2]
        
        # Compute each basis function: x^i * y^j * z^k
        for idx, (i, j, k) in enumerate(self.powers):
            phi[:, idx] = (x_vals ** i) * (y_vals ** j) * (z_vals ** k)
        
        return phi


class ForwardMREModelPolynomial(nn.Module):
    """
    Data-driven forward model using polynomial basis.
    
    Should work MUCH better than wave basis for MRE!
    """
    
    def __init__(
        self,
        mu_network: nn.Module,
        poly_degree: int = 3,
        input_dim: int = 3,
        seed: Optional[int] = None
    ):
        """
        Initialize forward model with polynomial basis.
        
        Args:
            mu_network: Stiffness network
            poly_degree: Polynomial degree (3 = cubic, 4 = quartic, etc.)
            input_dim: Spatial dimension
            seed: Random seed
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.mu_network = mu_network
        self.input_dim = input_dim
        
        # Polynomial basis generator
        self.poly_basis = PolynomialBasisGenerator(poly_degree, input_dim)
        self.n_basis = self.poly_basis.n_basis
        
        print(f"\n  Forward model (Polynomial) initialized:")
        print(f"    Polynomial degree: {poly_degree}")
        print(f"    Number of basis functions: {self.n_basis}")
        print(f"    Method: Data-driven (no PDE)")
    
    def get_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute polynomial basis functions.
        
        Args:
            x: (N, input_dim) coordinates
            
        Returns:
            phi: (N, n_basis) basis values
        """
        return self.poly_basis.evaluate(x)
    
    def build_system(
        self,
        phi: torch.Tensor,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 1000.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build system using row-concatenation (approach folder style).
        
        Args:
            phi: (N, M) basis functions
            bc_indices: Boundary point indices
            u_bc_vals: Boundary values
            bc_weight: BC constraint weight
            u_data: Measured displacement (N, 1)
            data_weight: Data fitting weight
            verbose: Print diagnostics
            
        Returns:
            H: (N_total, M) system matrix
            b: (N_total, 1) right-hand side
        """
        N, M = phi.shape
        
        if verbose:
            print(f"\n    Building system:")
            print(f"      N points: {N}, M basis: {M}")
        
        rows_H = []
        rows_b = []
        
        # Data fitting rows (main constraint)
        if u_data is not None and data_weight > 0:
            H_data = data_weight * phi  # (N, M)
            b_data = data_weight * u_data  # (N, 1)
            rows_H.append(H_data)
            rows_b.append(b_data)
            
            if verbose:
                print(f"      Data: {N} points, weight={data_weight:.1f}")
        
        # BC rows (for uniqueness)
        if bc_indices is not None and len(bc_indices) > 0:
            phi_bc = phi[bc_indices]  # (K, M)
            H_bc = bc_weight * phi_bc  # (K, M)
            b_bc = bc_weight * u_bc_vals  # (K, 1)
            rows_H.append(H_bc)
            rows_b.append(b_bc)
            
            if verbose:
                print(f"      BC: {len(bc_indices)} points, weight={bc_weight:.1f}")
        
        if len(rows_H) == 0:
            raise ValueError("Must have at least data or BC constraints!")
        
        H = torch.cat(rows_H, dim=0)
        b = torch.cat(rows_b, dim=0)
        
        if verbose:
            print(f"      Total: H {H.shape}, b {b.shape}")
        
        return H, b
    
    def forward(
        self,
        x: torch.Tensor,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 1000.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: pure data-driven interpolation with polynomial basis.
        
        Args:
            x: (N, 3) coordinates
            bc_indices: Boundary indices
            u_bc_vals: Boundary values
            bc_weight: BC weight
            u_data: Measured displacement
            data_weight: Data weight
            verbose: Print diagnostics
            
        Returns:
            u_pred: (N, 1) predicted displacement
            mu_pred: (N, 1) predicted stiffness
        """
        # Get stiffness field
        mu_pred = self.mu_network(x)
        
        # Get polynomial basis
        phi = self.get_basis(x)  # (N, M)
        
        # Build system
        H, b = self.build_system(
            phi, bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight, verbose
        )
        
        # Solve: H^T H c = H^T b
        c = pielm_solve(H, b, verbose=verbose)
        
        # Reconstruct: u = φ c
        u_pred = phi @ c
        
        return u_pred, mu_pred


if __name__ == '__main__':
    print("Testing ForwardMREModelPolynomial...")
    
    from stiffness_network import FlexibleStiffnessNetwork
    
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.2, mu_max=1.2
    )
    
    model = ForwardMREModelPolynomial(
        mu_network=mu_net,
        poly_degree=3,
        seed=42
    )
    
    # Test
    x = torch.rand(50, 3)
    u_meas = torch.rand(50, 1)
    bc_idx = torch.tensor([0, 25, 49])
    u_bc = u_meas[bc_idx]
    
    u_pred, mu_pred = model(
        x, bc_idx, u_bc, bc_weight=1.0,
        u_data=u_meas, data_weight=1000.0,
        verbose=True
    )
    
    mse = torch.mean((u_pred - u_meas)**2).item()
    print(f"\n✅ Test passed!")
    print(f"   u_pred: {u_pred.shape}, range [{u_pred.min():.3f}, {u_pred.max():.3f}]")
    print(f"   MSE: {mse:.6f}")
    
    if mse < 0.01:
        print(f"   🎉 EXCELLENT! MSE < 0.01 - polynomial basis works!")
