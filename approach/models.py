import torch
import torch.nn as nn
from pielm_solver import pielm_solve

class StiffnessGenerator(nn.Module):
    """
    Approximates the Stiffness Map mu(x).
    Input: Spatial coordinates (x) ∈ [0, 1]
    Output: Positive stiffness value mu(x)
    
    Architecture: Fourier feature network to encourage spatial variation
    """
    def __init__(self, input_dim=1, hidden_dim=64, n_fourier=10):
        super().__init__()
        
        # Random Fourier features to help with spatial patterns
        self.register_buffer('B_fourier', torch.randn(input_dim, n_fourier) * 5.0)
        
        # Network takes both raw x and Fourier features
        feature_dim = input_dim + 2 * n_fourier  # x + sin + cos features
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Standard initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Output layer bias for centering
        with torch.no_grad():
            self.net[-1].bias.fill_(0.5)
    
    def forward(self, x):
        """Forward pass with Fourier features and positivity constraint.
        
        Args:
            x: (N, 1) spatial coordinates ∈ [0, 1]
            
        Returns:
            mu: (N, 1) stiffness values (positive, clamped to reasonable bounds)
        """
        # Create Fourier features for better spatial representation
        z = x @ self.B_fourier  # (N, n_fourier)
        fourier_features = torch.cat([torch.sin(2 * 3.14159 * z), 
                                     torch.cos(2 * 3.14159 * z)], dim=1)
        
        # Concatenate original x with Fourier features
        features = torch.cat([x, fourier_features], dim=1)
        
        # Pass through network
        raw = self.net(features)
        
        # Softplus for smooth positivity
        mu = torch.nn.functional.softplus(raw) + 0.5
        
        # Clamp to reasonable range
        return torch.clamp(mu, min=0.5, max=5.0)

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
                     bc_weight: float = 1.0,
                     u_data: torch.Tensor = None,
                     data_weight: float = 0.0):
        """Assemble concatenated least-squares system (H, b).

        Args:
            mu_field: (N,1) stiffness values.
            phi: (N,M) basis values.
            phi_lap: (N,M) Laplacian values.
            rho_omega2: scalar ρ ω².
            bc_indices: (K,) long tensor of boundary indices.
            u_bc_vals: (K,1) boundary displacements.
            bc_weight: scalar weight to emphasize BC enforcement.
            u_data: (N,1) optional displacement measurements for data constraints.
            data_weight: scalar weight for data fitting constraints.
        Returns:
            H_total: (N+K[+N], M) design matrix rows.
            b_total: (N+K[+N], 1) targets.
        """
        H_pde = (mu_field * phi_lap) + (rho_omega2 * phi)
        b_pde = torch.zeros(phi.shape[0], 1, device=phi.device)
        
        # Start with PDE rows
        H_list = [H_pde]
        b_list = [b_pde]
        
        # Add BC rows if using boundary constraints
        if bc_weight > 0 and bc_indices is not None:
            H_bc = phi[bc_indices, :] * bc_weight
            b_bc = u_bc_vals * bc_weight
            H_list.append(H_bc)
            b_list.append(b_bc)
        
        # Add data constraint rows if provided
        if data_weight > 0 and u_data is not None:
            H_data = phi * data_weight
            b_data = u_data * data_weight
            H_list.append(H_data)
            b_list.append(b_data)
        
        H_total = torch.cat(H_list, dim=0)
        b_total = torch.cat(b_list, dim=0)
        return H_total, b_total

    def solve_given_mu(self,
                       x_col: torch.Tensor,
                       mu_field: torch.Tensor,
                       bc_indices: torch.Tensor,
                       u_bc_vals: torch.Tensor,
                       rho_omega2: float,
                       bc_weight: float = 1.0,
                       u_data: torch.Tensor = None,
                       data_weight: float = 0.0,
                       verbose: bool = False):
        """Solve wavefield for a provided stiffness (no dependence on mu_net).

        Args:
            x_col: (N,dim) collocation points.
            mu_field: (N,1) stiffness values.
            bc_indices: (K,) indices for Dirichlet BC.
            u_bc_vals: (K,1) boundary displacements.
            rho_omega2: scalar ρ ω².
            bc_weight: emphasis weight for BC rows.
            u_data: (N,1) optional displacement measurements.
            data_weight: emphasis weight for data constraints.
            verbose: enable solver diagnostics.
        Returns:
            u_pred: (N,1) displacement.
            C_u: (M,1) coefficient vector.
        """
        phi, phi_lap = self.get_basis_and_laplacian(x_col)
        H, b = self.build_system(mu_field, phi, phi_lap, rho_omega2, 
                                 bc_indices, u_bc_vals, bc_weight,
                                 u_data, data_weight)
        C_u = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C_u
        return u_pred, C_u

    def forward(self,
                x_col: torch.Tensor,
                bc_indices: torch.Tensor,
                u_bc_vals: torch.Tensor,
                rho_omega2: float,
                bc_weight: float = 1.0,
                u_data: torch.Tensor = None,
                data_weight: float = 0.0,
                verbose: bool = False):
        """Compute (u_pred, mu_pred) using learned mu.

        Args:
            x_col: (N,dim) collocation points.
            bc_indices: (K,) indices for Dirichlet BC.
            u_bc_vals: (K,1) boundary displacements.
            rho_omega2: scalar ρ ω².
            bc_weight: emphasis weight for BC rows.
            u_data: (N,1) optional displacement measurements.
            data_weight: emphasis weight for data constraints.
            verbose: enable solver diagnostics.
        Returns:
            u_pred: (N,1)
            mu_pred: (N,1)
        """
        mu_pred = self.mu_net(x_col)
        phi, phi_lap = self.get_basis_and_laplacian(x_col)
        H, b = self.build_system(mu_pred, phi, phi_lap, rho_omega2, 
                                 bc_indices, u_bc_vals, bc_weight,
                                 u_data, data_weight)
        C_u = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C_u
        return u_pred, mu_pred