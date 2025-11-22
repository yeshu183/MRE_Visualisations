"""
Direct Mu Parameterization Model
=================================

Direct gradient-based inverse solver WITHOUT neural network.
Uses classical adjoint method approach:

Algorithm:
    Initialize: μ⁰ (constant or random)
    For k = 0, 1, 2, ...
        1. Forward solve: Solve PDE to get u^k from μ^k
        2. Compute objective: J^k = ||u^k - u_meas||²
        3. Adjoint gradient: Compute ∇_μ J via autograd
        4. Update: μ^(k+1) = μ^k - α^k ∇_μ J^k
        5. Check convergence

Key difference from NN approach:
- μ is a direct learnable parameter (N, 1) instead of f_θ(x)
- Gradients update μ values directly, not network weights
- Classical optimization in parameter space, not weight space
"""

import torch
import torch.nn as nn
try:
    from .pielm_solver import pielm_solve
except ImportError:
    from pielm_solver import pielm_solve


class DirectMuModel(nn.Module):
    """Direct μ parameterization for MRE inverse problem.

    Instead of neural network, μ is directly optimized at each spatial point.
    """

    def __init__(
        self,
        n_points: int,
        n_wave_neurons: int = 60,
        input_dim: int = 3,
        omega_basis: float = 15.0,
        mu_init: float = 5000.0,  # Initial stiffness (Pa)
        mu_min: float = 3000.0,
        mu_max: float = 10000.0,
        seed: int = 42,
        basis_type: str = 'sin',
        init_mode: str = 'constant'  # 'constant', 'random', or 'uniform'
    ):
        """
        Initialize direct μ parameterization model.

        Args:
            n_points: Number of spatial points (N)
            n_wave_neurons: Number of wave basis functions
            input_dim: Spatial dimensions (1, 2, or 3)
            omega_basis: Scale for random wave frequencies
            mu_init: Initial value for μ (Pa)
            mu_min, mu_max: Bounds for μ (Pa)
            seed: Random seed for reproducibility
            basis_type: 'sin' for Fourier basis, 'tanh' for tanh basis
            init_mode: How to initialize μ
                - 'constant': All points = mu_init
                - 'random': Random in [mu_min, mu_max]
                - 'uniform': Uniform in [mu_min, mu_max]
        """
        super().__init__()

        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.n_wave = n_wave_neurons
        self.basis_type = basis_type
        self.omega_basis = omega_basis
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.n_points = n_points

        # === DIRECT MU PARAMETERIZATION ===
        # Instead of NN weights, μ is directly a learnable parameter
        # Shape: (N, 1) - one value per spatial point
        if init_mode == 'constant':
            mu_initial = torch.full((n_points, 1), mu_init, dtype=torch.float32)
        elif init_mode == 'random':
            mu_initial = torch.rand(n_points, 1) * (mu_max - mu_min) + mu_min
        elif init_mode == 'uniform':
            mu_initial = torch.linspace(mu_min, mu_max, n_points).view(-1, 1)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        # Register as learnable parameter
        self.mu_field = nn.Parameter(mu_initial)

        print(f"[DirectMu] Initialized μ field: shape={self.mu_field.shape}")
        print(f"           Mode={init_mode}, range=[{mu_initial.min():.0f}, {mu_initial.max():.0f}] Pa")

        # Random basis weights and biases (frozen, same as ForwardMREModel)
        self.register_buffer(
            'B', torch.randn(input_dim, n_wave_neurons) * omega_basis
        )
        self.register_buffer(
            'phi_bias', torch.randn(1, n_wave_neurons) * omega_basis
        )

        # RBF centers (for RBF basis type)
        self.register_buffer(
            'rbf_centers', torch.rand(n_wave_neurons, input_dim) * 0.1
        )
        self.rbf_sigma = 1.0 / omega_basis if omega_basis > 0 else 0.1

    def get_mu(self) -> torch.Tensor:
        """Get current μ field with clamping to valid range.

        Returns:
            mu: (N, 1) stiffness values clamped to [mu_min, mu_max]
        """
        # Clamp to physical bounds
        return torch.clamp(self.mu_field, self.mu_min, self.mu_max)

    def get_basis_and_laplacian(self, x: torch.Tensor):
        """Compute basis functions and their Laplacians.

        Identical to ForwardMREModel implementation.

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
            phi_lap = -2 * phi * (1 - phi ** 2) * freq_sq
        elif self.basis_type == 'rbf':
            diff = x.unsqueeze(1) - self.rbf_centers.unsqueeze(0)  # (N, M, dim)
            r_sq = torch.sum(diff ** 2, dim=2)  # (N, M)
            sigma_sq = self.rbf_sigma ** 2
            phi = torch.exp(-r_sq / (2 * sigma_sq))
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

        Identical to ForwardMREModel implementation.
        """
        # PDE rows
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
        """Forward pass: use current μ and solve for displacement.

        Key difference: μ is NOT predicted from NN, it's directly used.

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
            mu_pred: (N, 1) current stiffness (clamped)
        """
        # Get current μ (with clamping)
        mu_pred = self.get_mu()

        # Solve forward problem (UNCHANGED)
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
    # Test direct μ model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    N = 100
    model = DirectMuModel(
        n_points=N,
        n_wave_neurons=60,
        input_dim=3,
        omega_basis=15.0,
        mu_init=5000.0,
        mu_min=3000.0,
        mu_max=10000.0,
        init_mode='constant'
    ).to(device)

    # Test data
    x = torch.rand(N, 3, device=device)
    bc_indices = torch.tensor([0, N - 1], device=device)
    u_bc = torch.tensor([[0.01], [0.0]], device=device)

    # Forward pass
    u_pred, mu_pred = model(x, bc_indices, u_bc, rho_omega2=400.0, bc_weight=200.0)

    print(f"\nInput: x {x.shape}")
    print(f"Output: u_pred {u_pred.shape}, mu_pred {mu_pred.shape}")
    print(f"  u range: [{u_pred.min():.6e}, {u_pred.max():.6e}]")
    print(f"  μ range: [{mu_pred.min():.0f}, {mu_pred.max():.0f}] Pa")

    # Test gradient flow
    loss = torch.mean(u_pred ** 2)
    loss.backward()

    # Check gradient on μ_field (direct parameter)
    print(f"\nGradient on μ_field:")
    print(f"  Shape: {model.mu_field.grad.shape}")
    print(f"  Norm: {model.mu_field.grad.norm().item():.6e}")
    print(f"  Range: [{model.mu_field.grad.min():.6e}, {model.mu_field.grad.max():.6e}]")
