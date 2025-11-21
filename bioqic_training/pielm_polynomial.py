"""
PIELM with Polynomial Basis - PROPER VERSION

Key changes:
1. ✅ PDE rows (main physics constraint)
2. ✅ BC rows (high weight for boundary conditions)
3. ✅ Data rows (low weight for regularization)
4. ✅ Use mu_network predicted values (not constant)

This is TRUE PIELM: Physics-Informed + data regularization
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
        self.degree = degree
        self.input_dim = input_dim
        self.powers = self._generate_powers()
        self.n_basis = len(self.powers)
        
        print(f"  Polynomial basis: degree={degree}, n_basis={self.n_basis}")
    
    def _generate_powers(self):
        """Generate all multi-indices [i, j, k] where i+j+k <= degree."""
        powers = []
        for total_deg in range(self.degree + 1):
            for i in range(total_deg + 1):
                for j in range(total_deg + 1 - i):
                    k = total_deg - i - j
                    if k >= 0:
                        powers.append([i, j, k])
        return np.array(powers)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial basis at points x."""
        N = x.shape[0]
        phi = torch.zeros(N, self.n_basis, device=x.device, dtype=x.dtype)
        
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        z_vals = x[:, 2]
        
        for idx, (i, j, k) in enumerate(self.powers):
            phi[:, idx] = (x_vals ** i) * (y_vals ** j) * (z_vals ** k)
        
        return phi
    
    def evaluate_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Laplacian of polynomial basis: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        
        For φ = x^i * y^j * z^k:
        ∂²φ/∂x² = i(i-1) x^(i-2) y^j z^k
        ∂²φ/∂y² = j(j-1) x^i y^(j-2) z^k
        ∂²φ/∂z² = k(k-1) x^i y^j z^(k-2)
        """
        N = x.shape[0]
        phi_lap = torch.zeros(N, self.n_basis, device=x.device, dtype=x.dtype)
        
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        z_vals = x[:, 2]
        
        for idx, (i, j, k) in enumerate(self.powers):
            # ∂²/∂x²
            if i >= 2:
                phi_lap[:, idx] += i * (i - 1) * (x_vals ** (i - 2)) * (y_vals ** j) * (z_vals ** k)
            
            # ∂²/∂y²
            if j >= 2:
                phi_lap[:, idx] += j * (j - 1) * (x_vals ** i) * (y_vals ** (j - 2)) * (z_vals ** k)
            
            # ∂²/∂z²
            if k >= 2:
                phi_lap[:, idx] += k * (k - 1) * (x_vals ** i) * (y_vals ** j) * (z_vals ** (k - 2))
        
        return phi_lap


class PIELMPolyModel(nn.Module):
    """
    TRUE PIELM with polynomial basis:
    - Physics: μ∇²u + ρω²u = 0
    - BC: u(x_bc) = u_bc (high weight)
    - Data: u(x_data) ≈ u_measured (low weight, regularization)
    """
    
    def __init__(
        self,
        mu_network: nn.Module,
        poly_degree: int = 5,
        input_dim: int = 3,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.mu_network = mu_network
        self.input_dim = input_dim
        self.poly_basis = PolynomialBasisGenerator(poly_degree, input_dim)
        self.n_basis = self.poly_basis.n_basis
        
        print(f"\n  PIELM Polynomial Model initialized:")
        print(f"    Degree: {poly_degree}, Basis: {self.n_basis}")
        print(f"    PDE: μ∇²u + ρω²u = 0")
    
    def get_basis(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get basis functions and their Laplacians."""
        phi = self.poly_basis.evaluate(x)
        phi_lap = self.poly_basis.evaluate_laplacian(x)
        return phi, phi_lap
    
    def build_system(
        self,
        mu: torch.Tensor,
        phi: torch.Tensor,
        phi_lap: torch.Tensor,
        rho_omega2: float,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 100.0,  # HIGH - enforce BCs strongly
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 0.1,  # LOW - just regularization
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build PIELM system with row-concatenation.
        
        System:
        [    A    ] c = [ 0 ]     ← PDE: μ∇²u + ρω²u = 0 (N rows)
        [ w*φ_bc ]     [w*u_bc]   ← BC constraints (K rows, high weight)
        [ w*φ_data]    [w*u_data] ← Data regularization (N rows, low weight)
        
        Args:
            mu: (N, 1) stiffness from network
            phi: (N, M) basis functions
            phi_lap: (N, M) basis Laplacians
            rho_omega2: Physics parameter ρω²
            bc_indices: Boundary indices
            u_bc_vals: Boundary values
            bc_weight: BC weight (HIGH, e.g., 100)
            u_data: Measured displacement
            data_weight: Data weight (LOW, e.g., 0.1)
            verbose: Diagnostics
        """
        N, M = phi.shape
        
        if verbose:
            print(f"\n    Building PIELM system:")
            print(f"      N: {N}, M: {M}, ρω²: {rho_omega2:.1f}")
        
        rows_H = []
        rows_b = []
        
        # 1. PDE rows (main physics constraint)
        # PDE: μ∇²u + ρω²u = 0
        # Substitute u = φc: μ∇²φ·c + ρω²φ·c = 0
        # A = μ·∇²φ + ρω²·φ
        A_pde = mu * phi_lap + rho_omega2 * phi  # (N, M)
        b_pde = torch.zeros(N, 1, device=phi.device, dtype=phi.dtype)
        
        rows_H.append(A_pde)
        rows_b.append(b_pde)
        
        if verbose:
            print(f"      PDE: {N} rows (physics)")
            print(f"        mu range: [{mu.min():.3f}, {mu.max():.3f}]")
        
        # 2. BC rows (high weight - enforce strongly)
        if bc_indices is not None and len(bc_indices) > 0:
            phi_bc = phi[bc_indices]  # (K, M)
            H_bc = bc_weight * phi_bc
            b_bc = bc_weight * u_bc_vals
            
            rows_H.append(H_bc)
            rows_b.append(b_bc)
            
            if verbose:
                print(f"      BC: {len(bc_indices)} rows, weight={bc_weight:.1f} (HIGH)")
        
        # 3. Data rows (low weight - regularization only)
        if u_data is not None and data_weight > 0:
            H_data = data_weight * phi
            b_data = data_weight * u_data
            
            rows_H.append(H_data)
            rows_b.append(b_data)
            
            if verbose:
                print(f"      Data: {N} rows, weight={data_weight:.3f} (LOW - regularization)")
        
        # Concatenate
        H = torch.cat(rows_H, dim=0)
        b = torch.cat(rows_b, dim=0)
        
        if verbose:
            print(f"      Total: H {H.shape}, b {b.shape}")
        
        return H, b
    
    def forward(
        self,
        x: torch.Tensor,
        rho_omega2: float,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 100.0,
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 0.1,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PIELM forward pass.
        
        Args:
            x: (N, 3) coordinates
            rho_omega2: ρω² physics parameter
            bc_indices: Boundary indices
            u_bc_vals: Boundary values
            bc_weight: BC weight (HIGH, default 100)
            u_data: Measured data (optional, for regularization)
            data_weight: Data weight (LOW, default 0.1)
            verbose: Print diagnostics
        """
        # Get stiffness from network
        mu_pred = self.mu_network(x)  # (N, 1)
        
        # Get basis and Laplacians
        phi, phi_lap = self.get_basis(x)  # (N, M), (N, M)
        
        # Build system
        H, b = self.build_system(
            mu_pred, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight, verbose
        )
        
        # Solve PIELM
        c = pielm_solve(H, b, verbose=verbose)
        
        # Reconstruct
        u_pred = phi @ c
        
        return u_pred, mu_pred


if __name__ == '__main__':
    print("Testing PIELM with polynomial basis...")
    
    from stiffness_network import FlexibleStiffnessNetwork
    
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.2, mu_max=1.2
    )
    
    model = PIELMPolyModel(mu_network=mu_net, poly_degree=5, seed=42)
    
    # Test
    x = torch.rand(100, 3)
    u_meas = torch.rand(100, 1)
    bc_idx = torch.randperm(100)[:10]
    u_bc = u_meas[bc_idx]
    
    u_pred, mu_pred = model(
        x, rho_omega2=400.0,
        bc_indices=bc_idx, u_bc_vals=u_bc, bc_weight=100.0,
        u_data=u_meas, data_weight=0.1,
        verbose=True
    )
    
    print(f"\n✅ Test passed!")
    print(f"   u_pred: {u_pred.shape}, range [{u_pred.min():.3f}, {u_pred.max():.3f}]")
    print(f"   mu_pred: {mu_pred.shape}, range [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
