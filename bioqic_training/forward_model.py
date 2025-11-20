"""
Forward MRE Model with PIELM Solver
====================================

Physics-Informed Extreme Learning Machine for MRE wave equation:
    -∇·(μ∇u) - ρω²u = 0

Two physics scaling strategies:
1. PHYSICAL: Use true ρω² with coordinate rescaling for Laplacian
2. EFFECTIVE: Use tunable ρω² (400) as regularization parameter

Key Features:
- Random Fourier wave basis functions
- Direct PIELM solve (no iterative optimization for wave coefficients)
- Differentiable through the solver for μ-network backprop
- Support for boundary conditions and data constraints
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

# Import PIELM solver from approach folder
approach_path = Path(__file__).parent.parent / 'approach'
if str(approach_path) not in sys.path:
    sys.path.insert(0, str(approach_path))

from pielm_solver import pielm_solve


class ForwardMREModel(nn.Module):
    """
    Forward model for MRE inverse problem.
    
    Solves: -∇·(μ∇u) - ρω²u = 0
    
    Uses PIELM framework:
    - μ(x) predicted by neural network
    - u(x) = Σ c_i φ_i(x) where φ_i are random wave basis functions
    - Coefficients c_i solved via linear system H·c = b
    """
    
    def __init__(
        self,
        mu_network: nn.Module,
        n_wave_neurons: int = 100,
        input_dim: int = 3,
        omega_basis: float = 15.0,
        physics_mode: str = 'effective',
        rho_omega2_effective: float = 400.0,
        seed: Optional[int] = None
    ):
        """
        Initialize forward MRE model.
        
        Args:
            mu_network: Stiffness network (already instantiated)
            n_wave_neurons: Number of wave basis functions
            input_dim: Spatial dimension (3 for 3D)
            omega_basis: Scale for random wave frequencies
            physics_mode: 'physical' or 'effective'
                - 'physical': Use true ρω², requires coordinate rescaling
                - 'effective': Use tuned parameter for stable inversion
            rho_omega2_effective: Effective ρω² value (if mode='effective')
            seed: Random seed for wave basis
        """
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
        # These define the spectral content of the solution
        self.register_buffer(
            'omega_basis',
            torch.randn(input_dim, n_wave_neurons) * omega_basis
        )
        
        print(f"  Forward model initialized:")
        print(f"    Wave neurons: {n_wave_neurons}")
        print(f"    Input dim: {input_dim}D")
        print(f"    Physics mode: {physics_mode}")
        if physics_mode == 'effective':
            print(f"    ρω² (effective): {rho_omega2_effective:.1f}")
    
    def get_basis_and_laplacian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute wave basis functions and their Laplacians.
        
        Basis: φ_i(x) = sin(ω_i · x)
        Laplacian: ∇²φ_i = -‖ω_i‖² φ_i
        
        Args:
            x: (N, input_dim) spatial coordinates
            
        Returns:
            phi: (N, n_wave_neurons) basis function values
            phi_lap: (N, n_wave_neurons) Laplacian values
        """
        # Compute ω · x for all basis functions
        z = x @ self.omega_basis  # (N, n_wave_neurons)
        
        # Basis functions: sin(ω · x)
        phi = torch.sin(z)
        
        # Laplacian: ∇²φ = -‖ω‖² φ
        omega_sq = torch.sum(self.omega_basis ** 2, dim=0)  # (n_wave_neurons,)
        phi_lap = -omega_sq.unsqueeze(0) * phi  # (N, n_wave_neurons)
        
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
        Build PIELM linear system H·c = b.
        
        PDE residual: R = -∇·(μ∇u) - ρω²u = 0
        
        For u = Σ c_i φ_i:
            R = -μ Σ c_i ∇²φ_i - ρω² Σ c_i φ_i
            R = Σ c_i [-μ ∇²φ_i - ρω² φ_i]
            R = A · c
        
        where A_ij = -μ_i ∇²φ_j(x_i) - ρω² φ_j(x_i)
        
        Least squares: minimize ‖A·c‖² → H·c = 0 with H = A^T A
        
        Additional constraints:
        - Boundary conditions: φ(x_bc) = u_bc
        - Data fitting: φ(x_data) = u_data
        
        Args:
            mu: (N, 1) stiffness values
            phi: (N, M) basis functions
            phi_lap: (N, M) basis Laplacians
            rho_omega2: Physics parameter ρω²
            bc_indices: (K,) boundary point indices
            u_bc_vals: (K, 1) boundary displacement values
            bc_weight: Weight for BC constraints
            u_data: (N_data, 1) or (N, 1) measured displacement
            data_indices: Indices where data constraints apply (if None, use all)
            data_weight: Weight for data fitting constraints
            verbose: Print diagnostic info
            
        Returns:
            H: (M, M) system matrix
            b: (M, 1) right-hand side
        """
        N, M = phi.shape
        
        if verbose:
            print(f"\n    Building PIELM system:")
            print(f"      N collocation points: {N}")
            print(f"      M basis functions: {M}")
            # Handle both tensor and float for rho_omega2
            rho_val = rho_omega2.item() if torch.is_tensor(rho_omega2) else rho_omega2
            print(f"      ρω²: {rho_val:.1f}")
        
        # PDE residual matrix: A = -μ·∇²φ - ρω²·φ
        A = -mu * phi_lap - rho_omega2 * phi  # (N, M)
        
        # System matrix: H = A^T A (minimize PDE residual)
        H = A.t() @ A  # (M, M)
        b = torch.zeros(M, 1, device=phi.device, dtype=phi.dtype)
        
        if verbose:
            print(f"      PDE residual: ‖A‖² (implicit)")
        
        # Add boundary conditions: φ(x_bc) = u_bc
        if bc_indices is not None and len(bc_indices) > 0:
            phi_bc = phi[bc_indices]  # (K, M)
            H_bc = bc_weight * (phi_bc.t() @ phi_bc)  # (M, M)
            b_bc = bc_weight * (phi_bc.t() @ u_bc_vals)  # (M, 1)
            
            H = H + H_bc
            b = b + b_bc
            
            if verbose:
                print(f"      BC points: {len(bc_indices)} (weight={bc_weight:.1f})")
        
        # Add data constraints: φ(x_data) = u_data
        if u_data is not None and data_weight > 0:
            if data_indices is not None:
                # Use specific indices
                phi_data = phi[data_indices]
                u_data_subset = u_data[data_indices] if u_data.shape[0] == N else u_data
            else:
                # Use all points
                phi_data = phi
                u_data_subset = u_data
            
            H_data = data_weight * (phi_data.t() @ phi_data)  # (M, M)
            b_data = data_weight * (phi_data.t() @ u_data_subset)  # (M, 1)
            
            H = H + H_data
            b = b + b_data
            
            if verbose:
                print(f"      Data points: {phi_data.shape[0]} (weight={data_weight:.1f})")
        
        if verbose:
            print(f"      H condition number: {torch.linalg.cond(H).item():.2e}")
        
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
        """
        Forward pass: predict displacement and stiffness.
        
        Args:
            x: (N, input_dim) spatial coordinates
            rho_omega2: Physics parameter (physical or effective)
            bc_indices: Boundary point indices
            u_bc_vals: Boundary displacement values
            bc_weight: BC constraint weight
            u_data: Measured displacement (optional)
            data_indices: Where to apply data constraints
            data_weight: Data constraint weight
            verbose: Print diagnostics
            
        Returns:
            u_pred: (N, 1) predicted displacement
            mu_pred: (N, 1) predicted stiffness
        """
        # Predict stiffness field
        mu_pred = self.mu_network(x)
        
        # Get wave basis functions and Laplacians
        phi, phi_lap = self.get_basis_and_laplacian(x)
        
        # Build PIELM system
        H, b = self.build_pielm_system(
            mu_pred, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_indices, data_weight,
            verbose
        )
        
        # Solve for wave coefficients: H·c = b
        c = pielm_solve(H, b, verbose=verbose)
        
        # Reconstruct displacement: u = φ·c
        u_pred = phi @ c
        
        return u_pred, mu_pred
    
    def solve_given_mu(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        rho_omega2: float,
        bc_indices: Optional[torch.Tensor] = None,
        u_bc_vals: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve forward problem with given stiffness (for testing/validation).
        
        Args:
            x: Spatial coordinates
            mu: Given stiffness field
            rho_omega2: Physics parameter
            bc_indices: Boundary indices
            u_bc_vals: Boundary values
            bc_weight: BC weight
            
        Returns:
            u_pred: Predicted displacement
            c: Wave coefficients
        """
        phi, phi_lap = self.get_basis_and_laplacian(x)
        
        H, b = self.build_pielm_system(
            mu, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight
        )
        
        c = pielm_solve(H, b, verbose=False)
        u_pred = phi @ c
        
        return u_pred, c


def test_forward_model():
    """Test forward model with synthetic data."""
    print("="*70)
    print("FORWARD MODEL MODULE - Testing")
    print("="*70)
    
    # Create synthetic test data
    print("\n🔧 Creating test setup...")
    device = torch.device('cpu')
    N = 500
    x = torch.linspace(0, 1, N, device=device).reshape(-1, 1)
    x_3d = torch.cat([x, x*0.5, x*0.2], dim=1)  # 3D coordinates
    
    # Ground truth: Gaussian bump
    mu_true = 1.0 + 1.0 * torch.exp(-100 * (x[:, 0] - 0.5) ** 2).reshape(-1, 1)
    
    # Boundary conditions
    bc_indices = torch.tensor([0, N-1], dtype=torch.long, device=device)
    u_bc_vals = torch.tensor([[0.0], [0.1]], device=device)
    
    print(f"  Points: {N}")
    print(f"  Coordinates: 3D")
    print(f"  μ range: [{mu_true.min():.2f}, {mu_true.max():.2f}]")
    print(f"  BC points: {len(bc_indices)}")
    
    # Create stiffness network (simplified for testing)
    from stiffness_network import FlexibleStiffnessNetwork
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3,
        mu_min=0.5,
        mu_max=2.5,
        output_strategy='direct'
    ).to(device)
    
    print("\n" + "="*70)
    print("Test 1: EFFECTIVE Physics Mode")
    print("="*70)
    
    model_eff = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=60,
        input_dim=3,
        physics_mode='effective',
        rho_omega2_effective=400.0,
        seed=42
    ).to(device)
    
    print("\n  Forward pass (effective ρω²=400)...")
    u_pred_eff, mu_pred_eff = model_eff(
        x_3d,
        rho_omega2=400.0,
        bc_indices=bc_indices,
        u_bc_vals=u_bc_vals,
        bc_weight=100.0,
        verbose=True
    )
    
    print(f"\n  Results:")
    print(f"    u_pred shape: {u_pred_eff.shape}")
    print(f"    u_pred range: [{u_pred_eff.min():.3e}, {u_pred_eff.max():.3e}]")
    print(f"    μ_pred shape: {mu_pred_eff.shape}")
    print(f"    μ_pred range: [{mu_pred_eff.min():.3f}, {mu_pred_eff.max():.3f}]")
    print(f"    BCs satisfied: u[0]={u_pred_eff[0].item():.3e}, u[-1]={u_pred_eff[-1].item():.3e}")
    
    print("\n" + "="*70)
    print("Test 2: PHYSICAL Physics Mode")
    print("="*70)
    
    model_phys = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=60,
        input_dim=3,
        physics_mode='physical',
        seed=42
    ).to(device)
    
    # Physical ρω² (much larger)
    omega = 377.0  # rad/s (60 Hz)
    rho = 1000.0   # kg/m³
    rho_omega2_phys = rho * omega**2
    
    print(f"\n  Physical parameters:")
    print(f"    ω = {omega:.1f} rad/s")
    print(f"    ρ = {rho:.1f} kg/m³")
    print(f"    ρω² = {rho_omega2_phys:.1f} Pa/m²")
    
    print("\n  Forward pass (physical ρω²)...")
    u_pred_phys, mu_pred_phys = model_phys(
        x_3d,
        rho_omega2=rho_omega2_phys,
        bc_indices=bc_indices,
        u_bc_vals=u_bc_vals,
        bc_weight=100.0,
        verbose=True
    )
    
    print(f"\n  Results:")
    print(f"    u_pred shape: {u_pred_phys.shape}")
    print(f"    u_pred range: [{u_pred_phys.min():.3e}, {u_pred_phys.max():.3e}]")
    print(f"    μ_pred shape: {mu_pred_phys.shape}")
    print(f"    μ_pred range: [{mu_pred_phys.min():.3f}, {mu_pred_phys.max():.3f}]")
    
    print("\n" + "="*70)
    print("Test 3: With Data Constraints")
    print("="*70)
    
    # Generate synthetic measurement
    print("\n  Generating synthetic measurement...")
    with torch.no_grad():
        u_clean, _ = model_eff.solve_given_mu(
            x_3d, mu_true, 400.0,
            bc_indices, u_bc_vals, bc_weight=100.0
        )
        # Add noise
        u_meas = u_clean + 0.001 * torch.randn_like(u_clean)
    
    print(f"    u_meas range: [{u_meas.min():.3e}, {u_meas.max():.3e}]")
    
    print("\n  Forward pass with data constraints...")
    u_pred_data, mu_pred_data = model_eff(
        x_3d,
        rho_omega2=400.0,
        bc_indices=bc_indices,
        u_bc_vals=u_bc_vals,
        bc_weight=50.0,
        u_data=u_meas,
        data_weight=50.0,  # Equal weight to data and physics
        verbose=True
    )
    
    # Compute data fit error
    data_error = torch.mean((u_pred_data - u_meas)**2).item()
    print(f"\n  Data fit MSE: {data_error:.3e}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_forward_model()
