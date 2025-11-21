"""
PIELM with NORMALIZED PDE formulation.

KEY FIX: Divide PDE term by ρω² to balance magnitudes!

Original PDE: μ∇²u + ρω²u = 0
Normalized:   (μ/ρω²)∇²u + u = 0

This makes PDE rows O(1) instead of O(10^8)!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'approach'))
from pielm_solver import pielm_solve


class PolynomialBasisGenerator:
    """Generate polynomial basis up to degree D."""
    
    def __init__(self, degree: int = 3, input_dim: int = 3):
        self.degree = degree
        self.input_dim = input_dim
        self.powers = self._generate_powers()
        self.n_basis = len(self.powers)
        
        print(f"  Polynomial basis: degree={degree}, n_basis={self.n_basis}")
    
    def _generate_powers(self):
        powers = []
        for total_deg in range(self.degree + 1):
            for i in range(total_deg + 1):
                for j in range(total_deg + 1 - i):
                    k = total_deg - i - j
                    if k >= 0:
                        powers.append([i, j, k])
        return np.array(powers)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        phi = torch.zeros(N, self.n_basis, device=x.device, dtype=x.dtype)
        
        x_vals, y_vals, z_vals = x[:, 0], x[:, 1], x[:, 2]
        
        for idx, (i, j, k) in enumerate(self.powers):
            phi[:, idx] = (x_vals ** i) * (y_vals ** j) * (z_vals ** k)
        
        return phi
    
    def evaluate_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        phi_lap = torch.zeros(N, self.n_basis, device=x.device, dtype=x.dtype)
        
        x_vals, y_vals, z_vals = x[:, 0], x[:, 1], x[:, 2]
        
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


class PIELMNormalized(nn.Module):
    """
    PIELM with NORMALIZED PDE formulation.
    
    Key improvement: Divide PDE by ρω² to balance row magnitudes!
    
    PDE: (μ/ρω²)∇²u + u = 0
    
    This makes all rows O(1) magnitude.
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
        self.poly_basis = PolynomialBasisGenerator(poly_degree, input_dim)
        self.n_basis = self.poly_basis.n_basis
        
        print(f"\n  PIELM NORMALIZED initialized:")
        print(f"    Degree: {poly_degree}, Basis: {self.n_basis}")
        print(f"    PDE: (μ/ρω²)∇²u + u = 0  ← NORMALIZED!")
    
    def get_basis(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        bc_weight: float = 100.0,
        u_data: Optional[torch.Tensor] = None,
        data_weight: float = 0.1,
        pde_weight: float = 1.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build NORMALIZED PIELM system.
        
        PDE: (μ/ρω²)∇²u + u = 0
        Substitute u = φc:
        A = (μ/ρω²)·∇²φ + φ
        
        This normalization makes A ~ O(1) instead of O(10^8)!
        """
        N, M = phi.shape
        
        if verbose:
            print(f"\n    Building NORMALIZED PIELM system:")
            print(f"      N: {N}, M: {M}, ρω²: {rho_omega2:.1e}")
        
        rows_H = []
        rows_b = []
        
        # 1. NORMALIZED PDE rows
        # A = (μ/ρω²)·∇²φ + φ
        mu_normalized = mu / rho_omega2  # Scale down mu
        A_pde = pde_weight * (mu_normalized * phi_lap + phi)  # Apply pde_weight
        b_pde = torch.zeros(N, 1, device=phi.device, dtype=phi.dtype)
        
        rows_H.append(A_pde)
        rows_b.append(b_pde)
        
        if verbose:
            print(f"      PDE (NORMALIZED): {N} rows, weight={pde_weight:.3f}")
            print(f"        μ/(ρω²) range: [{mu_normalized.min().item():.2e}, {mu_normalized.max().item():.2e}]")
            print(f"        A_pde norm: {torch.norm(A_pde).item():.2e} (should be ~O(1-10))")
        
        # 2. BC rows
        if bc_indices is not None and len(bc_indices) > 0:
            phi_bc = phi[bc_indices]
            H_bc = bc_weight * phi_bc
            b_bc = bc_weight * u_bc_vals
            
            rows_H.append(H_bc)
            rows_b.append(b_bc)
            
            if verbose:
                print(f"      BC: {len(bc_indices)} rows, weight={bc_weight:.1f}")
                print(f"        H_bc norm: {torch.norm(H_bc).item():.2e}")
        
        # 3. Data rows
        if u_data is not None and data_weight > 0:
            H_data = data_weight * phi
            b_data = data_weight * u_data
            
            rows_H.append(H_data)
            rows_b.append(b_data)
            
            if verbose:
                print(f"      Data: {N} rows, weight={data_weight:.3f}")
                print(f"        H_data norm: {torch.norm(H_data).item():.2e}")
        
        H = torch.cat(rows_H, dim=0)
        b = torch.cat(rows_b, dim=0)
        
        if verbose:
            print(f"      Total: H {H.shape}, b {b.shape}")
            print(f"      H norm: {torch.norm(H).item():.2e}")
            print(f"      b norm: {torch.norm(b).item():.2e}")
        
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
        pde_weight: float = 1.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get stiffness
        mu_pred = self.mu_network(x)
        
        # Get basis
        phi, phi_lap = self.get_basis(x)
        
        # Build NORMALIZED system
        H, b = self.build_system(
            mu_pred, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight, pde_weight, verbose
        )
        
        # Solve
        c = pielm_solve(H, b, verbose=verbose)
        
        # Reconstruct
        u_pred = phi @ c
        
        return u_pred, mu_pred


if __name__ == '__main__':
    print("Testing NORMALIZED PIELM...")
    
    from stiffness_network import FlexibleStiffnessNetwork
    
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3, hidden_dim=64, n_layers=3,
        output_strategy='direct', mu_min=0.0, mu_max=1.0
    )
    
    model = PIELMNormalized(mu_network=mu_net, poly_degree=4, seed=42)
    
    # Test
    x = torch.rand(50, 3)
    u_meas = torch.rand(50, 1)
    bc_idx = torch.randperm(50)[:5]
    u_bc = u_meas[bc_idx]
    
    u_pred, mu_pred = model(
        x, rho_omega2=1.42e8,
        bc_indices=bc_idx, u_bc_vals=u_bc, bc_weight=100.0,
        u_data=u_meas, data_weight=0.1,
        verbose=True
    )
    
    mse = torch.mean((u_pred - u_meas)**2).item()
    print(f"\n✅ Test passed!")
    print(f"   u_pred: range [{u_pred.min():.3f}, {u_pred.max():.3f}]")
    print(f"   MSE: {mse:.6f}")
    
    if u_pred.abs().max() > 0.01:
        print(f"   ✅ Solution is NON-ZERO! Normalization works!")
    else:
        print(f"   ⚠️  Solution still collapsed")
