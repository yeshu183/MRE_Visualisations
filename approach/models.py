import torch
import torch.nn as nn
from pielm_solver import pielm_solve

class StiffnessGenerator(nn.Module):
    """
    Approximates the Stiffness Map mu(x).
    Input: Spatial coordinates (x)
    Output: Positive stiffness value
    
    Architecture: Deeper network (4 hidden layers, 128 units) for better capacity
    """
    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        # Deeper network without skip connections (for stability)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Careful initialization: start near middle of expected range [1.0, 3.0]
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                # Very small weight initialization for stability
                nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Reduced from 0.3
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        
        # Output layer: initialize to produce values near 1.8
        # Target: softplus(x) + 0.9 ≈ 1.8, so softplus(x) ≈ 0.9
        # Since softplus(0) ≈ 0.69, softplus(0.5) ≈ 0.97, we want x ≈ 0.5
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.001)  # Even smaller std
        nn.init.constant_(self.net[-1].bias, 0.5)
        
    def forward(self, x):
        # Forward pass - no clamping on intermediate values
        raw = self.net(x)
        # Softplus ensures positivity, add offset
        mu = torch.nn.functional.softplus(raw) + 0.9
        # Loose final bounds to allow learning
        return torch.clamp(mu, min=0.7, max=6.0)

class ForwardMREModel(nn.Module):
    """Analysis-by-Synthesis forward model for MRE with locally homogeneous mu assumption.

    Enhancements in this version:
    - Explicit boundary handling via index tensor instead of implicit slicing.
    - Helper methods to build systems and solve for a provided ``mu_field``.
    - Deterministic option through a seed for reproducible Fourier features.
    - Clear tensor shape documentation for easier testing and extension.
    """

    def __init__(self, n_neurons_wave, input_dim=1, omega_basis=15.0, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.mu_net = StiffnessGenerator(input_dim)
        self.n_wave = n_neurons_wave
        self.omega = omega_basis
        # Random frozen Fourier features
        self.B = nn.Parameter(torch.randn(input_dim, n_neurons_wave) * self.omega, requires_grad=False)
        self.phi_bias = nn.Parameter(torch.rand(1, n_neurons_wave) * 2 * 3.14159, requires_grad=False)

    def get_basis_and_laplacian(self, x: torch.Tensor):
        """Compute basis φ(x) and its Laplacian for 1D/ND sine Fourier features.

        Args:
            x: (N, dim) coordinates.
        Returns:
            phi: (N, M) basis values.
            phi_lap: (N, M) Laplacian values.
        """
        Z = x @ self.B + self.phi_bias  # (N,M)
        phi = torch.sin(Z)
        freq_sq = torch.sum(self.B**2, dim=0, keepdim=True)  # (1,M)
        phi_lap = -phi * freq_sq
        return phi, phi_lap

    def build_system(self,
                     mu_field: torch.Tensor,
                     phi: torch.Tensor,
                     phi_lap: torch.Tensor,
                     rho_omega2: float,
                     bc_indices: torch.Tensor,
                     u_bc_vals: torch.Tensor,
                     bc_weight: float = 1.0):
        """Assemble concatenated least-squares system (H, b).

        Args:
            mu_field: (N,1) stiffness values.
            phi: (N,M) basis values.
            phi_lap: (N,M) Laplacian values.
            rho_omega2: scalar ρ ω².
            bc_indices: (K,) long tensor of boundary indices.
            u_bc_vals: (K,1) boundary displacements.
            bc_weight: scalar weight to emphasize BC enforcement.
        Returns:
            H_total: (N+K, M) design matrix rows.
            b_total: (N+K, 1) targets.
        """
        H_pde = (mu_field * phi_lap) + (rho_omega2 * phi)
        b_pde = torch.zeros(phi.shape[0], 1, device=phi.device)
        H_bc = phi[bc_indices, :] * bc_weight
        b_bc = u_bc_vals * bc_weight
        H_total = torch.cat([H_pde, H_bc], dim=0)
        b_total = torch.cat([b_pde, b_bc], dim=0)
        return H_total, b_total

    def solve_given_mu(self,
                       x_col: torch.Tensor,
                       mu_field: torch.Tensor,
                       bc_indices: torch.Tensor,
                       u_bc_vals: torch.Tensor,
                       rho_omega2: float,
                       bc_weight: float = 1.0,
                       verbose: bool = False):
        """Solve wavefield for a provided stiffness (no dependence on mu_net).

        Returns:
            u_pred: (N,1) displacement.
            C_u: (M,1) coefficient vector.
        """
        phi, phi_lap = self.get_basis_and_laplacian(x_col)
        H, b = self.build_system(mu_field, phi, phi_lap, rho_omega2, bc_indices, u_bc_vals, bc_weight)
        C_u = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C_u
        return u_pred, C_u

    def forward(self,
                x_col: torch.Tensor,
                bc_indices: torch.Tensor,
                u_bc_vals: torch.Tensor,
                rho_omega2: float,
                bc_weight: float = 1.0,
                verbose: bool = False):
        """Compute (u_pred, mu_pred) using learned mu.

        Args:
            x_col: (N,dim) collocation points.
            bc_indices: (K,) indices for Dirichlet BC.
            u_bc_vals: (K,1) boundary displacements.
            rho_omega2: scalar ρ ω².
            bc_weight: emphasis weight for BC rows.
            verbose: enable solver diagnostics.
        Returns:
            u_pred: (N,1)
            mu_pred: (N,1)
        """
        mu_pred = self.mu_net(x_col)
        phi, phi_lap = self.get_basis_and_laplacian(x_col)
        H, b = self.build_system(mu_pred, phi, phi_lap, rho_omega2, bc_indices, u_bc_vals, bc_weight)
        C_u = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C_u
        return u_pred, mu_pred