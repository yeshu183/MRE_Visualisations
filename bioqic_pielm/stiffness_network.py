"""
Stiffness Network
=================

Neural network for parameterizing the stiffness field μ(x).

Architecture:
- Random Fourier features for spatial encoding (prevents mode collapse)
- 2-layer Tanh MLP
- Bounded output via sigmoid + rescaling

For BIOQIC data:
- Normalized output in [0, 1] maps to [3000, 10000] Pa
"""

import torch
import torch.nn as nn
import numpy as np


class StiffnessNetwork(nn.Module):
    """Neural network for stiffness field μ(x).

    Uses Fourier feature encoding to capture spatial variation.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        n_fourier: int = 10,
        mu_min: float = 0.0,
        mu_max: float = 1.0,
        fourier_scale: float = 5.0,
        seed: int = 42
    ):
        """
        Initialize stiffness network.

        Args:
            input_dim: Spatial dimensions (1D, 2D, or 3D)
            hidden_dim: Hidden layer width
            n_fourier: Number of Fourier feature frequencies
            mu_min: Minimum output value (normalized)
            mu_max: Maximum output value (normalized)
            fourier_scale: Scale for random Fourier frequencies
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max

        # Set seed for reproducible Fourier features
        torch.manual_seed(seed)

        # Random Fourier features (frozen)
        self.register_buffer(
            'B_fourier',
            torch.randn(input_dim, n_fourier) * fourier_scale
        )

        # Network: [x, sin(Bx), cos(Bx)] -> hidden -> hidden -> output
        feature_dim = input_dim + 2 * n_fourier

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Output bias for centering
        with torch.no_grad():
            self.net[-1].bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute stiffness at spatial locations.

        Args:
            x: (N, dim) spatial coordinates in [0, 1]

        Returns:
            mu: (N, 1) stiffness values in [mu_min, mu_max]
        """
        # Fourier feature encoding
        z = x @ self.B_fourier  # (N, n_fourier)
        fourier_features = torch.cat([
            torch.sin(2 * np.pi * z),
            torch.cos(2 * np.pi * z)
        ], dim=1)

        # Concatenate raw coordinates with Fourier features
        features = torch.cat([x, fourier_features], dim=1)

        # Network forward pass
        raw = self.net(features)

        # Bounded output via sigmoid
        normalized = torch.sigmoid(raw)
        mu = self.mu_min + (self.mu_max - self.mu_min) * normalized

        return mu


class ComplexStiffnessNetwork(StiffnessNetwork):
    """Stiffness network that outputs complex modulus magnitude.

    For viscoelastic BIOQIC data: G* = μ + iωη
    Outputs |G*| = sqrt(μ² + (ωη)²)
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        n_fourier: int = 10,
        mu_min_pa: float = 3000.0,  # Physical units
        mu_max_pa: float = 10000.0,
        eta: float = 1.0,  # Viscosity (Pa·s)
        omega: float = 377.0,  # Angular frequency (rad/s)
        **kwargs
    ):
        """
        Args:
            mu_min_pa: Min storage modulus in Pa
            mu_max_pa: Max storage modulus in Pa
            eta: Shear viscosity (constant)
            omega: Angular frequency (2πf)
        """
        # Network outputs normalized [0, 1] which maps to storage modulus
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_fourier=n_fourier,
            mu_min=0.0,
            mu_max=1.0,
            **kwargs
        )
        self.mu_min_pa = mu_min_pa
        self.mu_max_pa = mu_max_pa
        self.eta = eta
        self.omega = omega
        self.omega_eta = omega * eta  # Loss modulus = ωη

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized stiffness.

        For inverse problem, we predict normalized μ in [0, 1].
        """
        return super().forward(x)

    def get_complex_magnitude(self, mu_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized μ to complex modulus magnitude |G*|.

        Args:
            mu_normalized: (N, 1) normalized stiffness in [0, 1]

        Returns:
            G_magnitude: (N, 1) |G*| = sqrt(μ² + (ωη)²) in Pa
        """
        # Denormalize to physical units
        mu_storage = self.mu_min_pa + mu_normalized * (self.mu_max_pa - self.mu_min_pa)

        # Complex modulus magnitude
        G_magnitude = torch.sqrt(mu_storage ** 2 + self.omega_eta ** 2)

        return G_magnitude


if __name__ == "__main__":
    # Test stiffness network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Basic network
    net = StiffnessNetwork(input_dim=3, hidden_dim=64).to(device)
    x = torch.rand(100, 3, device=device)
    mu = net(x)
    print(f"StiffnessNetwork output: {mu.shape}, range [{mu.min():.3f}, {mu.max():.3f}]")

    # Complex network
    net_complex = ComplexStiffnessNetwork(input_dim=3).to(device)
    mu_norm = net_complex(x)
    G_mag = net_complex.get_complex_magnitude(mu_norm)
    print(f"ComplexStiffnessNetwork:")
    print(f"  Normalized: [{mu_norm.min():.3f}, {mu_norm.max():.3f}]")
    print(f"  |G*| (Pa): [{G_mag.min():.0f}, {G_mag.max():.0f}]")
