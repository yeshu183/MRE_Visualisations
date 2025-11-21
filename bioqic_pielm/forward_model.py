"""
Forward MRE Model with PIELM
============================

Physics-Informed Extreme Learning Machine for solving the forward problem:
    ∇·(μ∇u) + ρω²u = 0  (Helmholtz equation)

The solution u is approximated as:
    u(x) = Σ C_j φ_j(x)

where φ_j are random Fourier basis functions and C_j are coefficients
found by solving a least-squares system.

For BIOQIC viscoelastic data:
    G*(ω) = μ + iωη (complex modulus)
    Option to use |G*| instead of μ for physics consistency.
"""

import torch
import torch.nn as nn
import numpy as np
from .pielm_solver import pielm_solve
from .stiffness_network import StiffnessNetwork


class ForwardMREModel(nn.Module):
    """Analysis-by-synthesis forward model for MRE inversion.

    Learns stiffness field μ(x) by iteratively:
    1. Predict μ from neural network
    2. Solve forward problem: ∇·(μ∇u) + ρω²u = 0
    3. Compare predicted u with measured u
    4. Backpropagate through solver to update μ-network
    """

    def __init__(
        self,
        n_wave_neurons: int = 60,
        input_dim: int = 3,
        omega_basis: float = 15.0,
        hidden_dim: int = 64,
        n_fourier: int = 10,
        mu_min: float = 0.0,
        mu_max: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize forward model.

        Args:
            n_wave_neurons: Number of wave basis functions
            input_dim: Spatial dimensions (1, 2, or 3)
            omega_basis: Scale for random wave frequencies
            hidden_dim: Stiffness network hidden dimension
            n_fourier: Fourier features for stiffness network
            mu_min, mu_max: Stiffness output bounds (normalized)
            seed: Random seed for reproducibility
        """
        super().__init__()

        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.n_wave = n_wave_neurons

        # Stiffness network
        self.mu_net = StiffnessNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_fourier=n_fourier,
            mu_min=mu_min,
            mu_max=mu_max,
            seed=seed
        )

        # Random Fourier wave basis (frozen)
        self.register_buffer(
            'B', torch.randn(input_dim, n_wave_neurons) * omega_basis
        )
        self.register_buffer(
            'phi_bias', torch.rand(1, n_wave_neurons) * 2 * np.pi
        )

    def get_basis_and_laplacian(self, x: torch.Tensor):
        """Compute basis functions and their Laplacians.

        Basis: φ_j(x) = sin(ω_j · x + b_j)
        Laplacian: ∇²φ_j = -||ω_j||² φ_j

        Args:
            x: (N, dim) spatial coordinates

        Returns:
            phi: (N, M) basis function values
            phi_lap: (N, M) Laplacian values
        """
        Z = x @ self.B + self.phi_bias  # (N, M)
        phi = torch.sin(Z)
        freq_sq = torch.sum(self.B ** 2, dim=0, keepdim=True)  # (1, M)
        phi_lap = -phi * freq_sq
        return phi, phi_lap

    def build_system(
        self,
        mu_field: torch.Tensor,
        phi: torch.Tensor,
        phi_lap: torch.Tensor,
        rho_omega2: float,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        bc_weight: float = 1.0,
        u_data: torch.Tensor = None,
        data_weight: float = 0.0
    ):
        """Assemble least-squares system [H, b].

        System rows:
        1. PDE residual: μ·∇²φ + ρω²·φ = 0
        2. Boundary conditions: φ(x_bc) = u_bc (weighted)
        3. Data constraints: φ(x) = u_meas (optional, weighted)

        Args:
            mu_field: (N, 1) stiffness values
            phi, phi_lap: (N, M) basis and Laplacian
            rho_omega2: Physics parameter ρω²
            bc_indices: (K,) boundary point indices
            u_bc_vals: (K, 1) boundary displacement values
            bc_weight: Weight for BC enforcement
            u_data: (N, 1) optional displacement measurements
            data_weight: Weight for data fitting

        Returns:
            H: Design matrix
            b: Target vector
        """
        # PDE rows: μ·∇²φ + ρω²·φ = 0
        H_pde = (mu_field * phi_lap) + (rho_omega2 * phi)
        b_pde = torch.zeros(phi.shape[0], 1, device=phi.device)

        H_list = [H_pde]
        b_list = [b_pde]

        # BC rows
        if bc_weight > 0 and bc_indices is not None and len(bc_indices) > 0:
            H_bc = phi[bc_indices, :] * bc_weight
            b_bc = u_bc_vals * bc_weight
            H_list.append(H_bc)
            b_list.append(b_bc)

        # Data constraint rows
        if data_weight > 0 and u_data is not None:
            H_data = phi * data_weight
            b_data = u_data * data_weight
            H_list.append(H_data)
            b_list.append(b_data)

        H = torch.cat(H_list, dim=0)
        b = torch.cat(b_list, dim=0)
        return H, b

    def solve_given_mu(
        self,
        x: torch.Tensor,
        mu_field: torch.Tensor,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        rho_omega2: float,
        bc_weight: float = 1.0,
        u_data: torch.Tensor = None,
        data_weight: float = 0.0,
        verbose: bool = False
    ):
        """Solve for displacement given stiffness field.

        Args:
            x: (N, dim) coordinates
            mu_field: (N, 1) stiffness
            Other args: See build_system

        Returns:
            u_pred: (N, 1) predicted displacement
            C: (M, 1) basis coefficients
        """
        phi, phi_lap = self.get_basis_and_laplacian(x)
        H, b = self.build_system(
            mu_field, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight
        )
        C = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C
        return u_pred, C

    def forward(
        self,
        x: torch.Tensor,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        rho_omega2: float,
        bc_weight: float = 1.0,
        u_data: torch.Tensor = None,
        data_weight: float = 0.0,
        verbose: bool = False
    ):
        """Forward pass: predict stiffness and solve for displacement.

        Args:
            x: (N, dim) coordinates
            bc_indices: (K,) boundary indices
            u_bc_vals: (K, 1) boundary values
            rho_omega2: Physics parameter
            bc_weight: BC enforcement weight
            u_data: (N, 1) measured displacement
            data_weight: Data fitting weight
            verbose: Print solver info

        Returns:
            u_pred: (N, 1) predicted displacement
            mu_pred: (N, 1) predicted stiffness
        """
        # Predict stiffness from network
        mu_pred = self.mu_net(x)

        # Solve forward problem
        phi, phi_lap = self.get_basis_and_laplacian(x)
        H, b = self.build_system(
            mu_pred, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight
        )
        C = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C

        return u_pred, mu_pred


if __name__ == "__main__":
    # Test forward model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = ForwardMREModel(
        n_wave_neurons=60,
        input_dim=3,
        omega_basis=15.0
    ).to(device)

    # Test data
    N = 100
    x = torch.rand(N, 3, device=device)
    bc_indices = torch.tensor([0, N - 1], device=device)
    u_bc = torch.tensor([[0.01], [0.0]], device=device)

    # Forward pass
    u_pred, mu_pred = model(x, bc_indices, u_bc, rho_omega2=400.0, bc_weight=200.0)

    print(f"Input: x {x.shape}")
    print(f"Output: u_pred {u_pred.shape}, mu_pred {mu_pred.shape}")
    print(f"  u range: [{u_pred.min():.4f}, {u_pred.max():.4f}]")
    print(f"  μ range: [{mu_pred.min():.4f}, {mu_pred.max():.4f}]")

    # Test gradient flow
    loss = torch.mean(u_pred ** 2)
    loss.backward()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.mu_net.parameters()
                    if p.grad is not None) ** 0.5
    print(f"  Gradient norm: {grad_norm:.6f}")
