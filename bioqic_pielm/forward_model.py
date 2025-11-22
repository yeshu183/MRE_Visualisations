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
try:
    from .pielm_solver import pielm_solve
    from .stiffness_network import StiffnessNetwork
except ImportError:
    from pielm_solver import pielm_solve
    from stiffness_network import StiffnessNetwork


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
        mu_min: float = 1.0,  # Changed: normalized range [1, 2] like approach folder
        mu_max: float = 2.0,
        seed: int = 42,
        basis_type: str = 'sin'  # 'sin' or 'tanh'
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
            basis_type: 'sin' for Fourier basis, 'tanh' for tanh basis
        """
        super().__init__()

        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.n_wave = n_wave_neurons
        self.basis_type = basis_type
        self.omega_basis = omega_basis

        # Stiffness network
        self.mu_net = StiffnessNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_fourier=n_fourier,
            mu_min=mu_min,
            mu_max=mu_max,
            seed=seed
        )

        # Random basis weights and biases (frozen)
        # W: (input_dim, n_neurons), b: (1, n_neurons)
        self.register_buffer(
            'B', torch.randn(input_dim, n_wave_neurons) * omega_basis
        )
        self.register_buffer(
            'phi_bias', torch.randn(1, n_wave_neurons) * omega_basis  # Random bias
        )

        # RBF centers: random points in domain [0, 0.1]^dim (SI units, ~0.1m domain)
        self.register_buffer(
            'rbf_centers', torch.rand(n_wave_neurons, input_dim) * 0.1
        )
        # RBF width parameter (sigma): omega_basis = 1/sigma
        self.rbf_sigma = 1.0 / omega_basis if omega_basis > 0 else 0.1

    def get_basis_and_laplacian(self, x: torch.Tensor):
        """Compute basis functions and their Laplacians.

        For sin basis:
            φ_j(x) = sin(w_j · x + b_j)
            ∇²φ_j = -||w_j||² φ_j

        For tanh basis:
            φ_j(x) = tanh(w_j · x + b_j)
            ∇²φ_j = -2 * tanh * sech² * ||w_j||²
                  = -2 * φ * (1 - φ²) * ||w_j||²

        Args:
            x: (N, dim) spatial coordinates

        Returns:
            phi: (N, M) basis function values
            phi_lap: (N, M) Laplacian values
        """
        Z = x @ self.B + self.phi_bias  # (N, M)
        freq_sq = torch.sum(self.B ** 2, dim=0, keepdim=True)  # (1, M)

        if self.basis_type == 'sin':
            phi = torch.sin(Z)
            phi_lap = -phi * freq_sq
        elif self.basis_type == 'tanh':
            phi = torch.tanh(Z)
            # d²/dz² tanh(z) = -2 * tanh(z) * sech²(z) = -2 * tanh(z) * (1 - tanh²(z))
            # For φ = tanh(w·x + b), ∇²φ = φ'' * ||w||²
            phi_lap = -2 * phi * (1 - phi ** 2) * freq_sq
        elif self.basis_type == 'rbf':
            # RBF: φ_j(x) = exp(-||x - c_j||² / (2σ²))
            # Compute squared distances: ||x - c||²
            # x: (N, dim), centers: (M, dim)
            diff = x.unsqueeze(1) - self.rbf_centers.unsqueeze(0)  # (N, M, dim)
            r_sq = torch.sum(diff ** 2, dim=2)  # (N, M)
            sigma_sq = self.rbf_sigma ** 2
            phi = torch.exp(-r_sq / (2 * sigma_sq))
            # Laplacian of Gaussian RBF:
            # ∇²φ = φ * (||x-c||² - dim*σ²) / σ⁴
            dim = self.input_dim
            phi_lap = phi * (r_sq - dim * sigma_sq) / (sigma_sq ** 2)
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

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
        # PDE rows: (μ/ρω²)·∇²φ + φ = 0  (normalized by ρω² for balanced terms)
        H_pde = (mu_field / rho_omega2) * phi_lap + phi
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
