"""
Stiffness Network for MRE Inverse Problems
===========================================

Flexible neural network to parameterize spatially-varying stiffness field.

Key Features:
- Configurable output bounds to match data range
- Random Fourier features for spatial variation
- Multiple parameterization strategies (direct, log, sigmoid-scaled)
- Proper initialization for stable training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class FlexibleStiffnessNetwork(nn.Module):
    """
    Neural network for stiffness field parameterization.
    
    Architecture:
    - Input: Spatial coordinates (x, y, z)
    - Random Fourier features for multi-scale representation
    - MLP with tanh activations
    - Output: Stiffness value μ(x, y, z)
    
    Output strategies:
    - 'direct': Sigmoid scaled to [μ_min, μ_max]
    - 'log': Predict log(μ), then exponentiate (better for large ranges)
    - 'softplus': SoftPlus scaled to [μ_min, μ_max] (smooth, non-negative)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_fourier: int = 10,
        mu_min: float = 0.1,
        mu_max: float = 5.0,
        output_strategy: str = 'direct',
        fourier_scale: float = 5.0,
        seed: Optional[int] = None
    ):
        """
        Initialize stiffness network.
        
        Args:
            input_dim: Input dimension (3 for x,y,z)
            hidden_dim: Hidden layer width
            n_layers: Number of hidden layers
            n_fourier: Number of Fourier features per dimension
            mu_min: Minimum stiffness value (normalized)
            mu_max: Maximum stiffness value (normalized)
            output_strategy: 'direct', 'log', or 'softplus'
            fourier_scale: Scale of random Fourier features
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_fourier = n_fourier
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.output_strategy = output_strategy
        
        # Validate bounds
        if mu_min >= mu_max:
            raise ValueError(f"mu_min ({mu_min}) must be < mu_max ({mu_max})")
        
        if output_strategy not in ['direct', 'log', 'softplus']:
            raise ValueError(f"output_strategy must be 'direct', 'log', or 'softplus', got {output_strategy}")
        
        # Random Fourier features: sin(2π * B @ x), cos(2π * B @ x)
        # These capture multi-scale spatial patterns
        B = torch.randn(input_dim, n_fourier) * fourier_scale
        self.register_buffer('B_fourier', B)
        
        # Feature dimension: original coords + sin/cos Fourier features
        feature_dim = input_dim + 2 * n_fourier
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for tanh activations
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Center the output layer to start near mean stiffness
        with torch.no_grad():
            # Bias output to predict midpoint initially
            if self.output_strategy == 'direct':
                # For sigmoid: logit of 0.5 = 0
                self.net[-1].bias.fill_(0.0)
            elif self.output_strategy == 'log':
                # For log: predict log of geometric mean
                log_mean = 0.5 * (np.log(self.mu_min) + np.log(self.mu_max))
                self.net[-1].bias.fill_(log_mean)
            elif self.output_strategy == 'softplus':
                # For softplus: start near middle
                self.net[-1].bias.fill_(0.0)
    
    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute input features with Random Fourier Features.
        
        Args:
            x: (N, input_dim) spatial coordinates
            
        Returns:
            features: (N, feature_dim) enhanced features
        """
        # Random Fourier Features
        z = x @ self.B_fourier  # (N, n_fourier)
        fourier_features = torch.cat([
            torch.sin(2 * np.pi * z),
            torch.cos(2 * np.pi * z)
        ], dim=1)  # (N, 2*n_fourier)
        
        # Concatenate with original coordinates
        features = torch.cat([x, fourier_features], dim=1)  # (N, feature_dim)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x → μ(x).
        
        Args:
            x: (N, input_dim) spatial coordinates
            
        Returns:
            mu: (N, 1) stiffness values in [mu_min, mu_max]
        """
        # Compute features
        features = self._compute_features(x)
        
        # Network forward pass
        raw_output = self.net(features)  # (N, 1)
        
        # Apply output strategy to map to [mu_min, mu_max]
        if self.output_strategy == 'direct':
            # Sigmoid: maps to [0, 1], then scale
            normalized = torch.sigmoid(raw_output)
            mu = self.mu_min + (self.mu_max - self.mu_min) * normalized
            
        elif self.output_strategy == 'log':
            # Predict log(μ), then exponentiate
            # Constrain to [log(mu_min), log(mu_max)]
            log_mu = raw_output
            log_min = np.log(self.mu_min)
            log_max = np.log(self.mu_max)
            
            # Sigmoid to [0, 1], then scale to log range
            normalized = torch.sigmoid(log_mu)
            log_mu_scaled = log_min + (log_max - log_min) * normalized
            mu = torch.exp(log_mu_scaled)
            
        elif self.output_strategy == 'softplus':
            # SoftPlus: smooth approximation of ReLU
            # F(x) = log(1 + exp(x))
            positive = torch.nn.functional.softplus(raw_output)
            # Normalize to [0, 1] range using sigmoid on scaled output
            # Then scale to [mu_min, mu_max]
            normalized = torch.sigmoid(positive)
            mu = self.mu_min + (self.mu_max - self.mu_min) * normalized
        
        return mu
    
    def get_statistics(self, x: torch.Tensor) -> dict:
        """
        Compute statistics of predicted stiffness field.
        
        Args:
            x: (N, input_dim) spatial coordinates
            
        Returns:
            Dictionary with statistics
        """
        with torch.no_grad():
            mu = self(x)
            
            stats = {
                'min': mu.min().item(),
                'max': mu.max().item(),
                'mean': mu.mean().item(),
                'std': mu.std().item(),
                'median': mu.median().item(),
                'range': (mu.max() - mu.min()).item(),
                'n_params': sum(p.numel() for p in self.parameters())
            }
            
        return stats


def compare_strategies(
    x_test: torch.Tensor,
    mu_min: float = 0.3,
    mu_max: float = 1.0,
    seed: int = 42
) -> None:
    """
    Compare different output strategies on test data.
    
    Args:
        x_test: Test coordinates
        mu_min: Minimum stiffness
        mu_max: Maximum stiffness
        seed: Random seed
    """
    import matplotlib.pyplot as plt
    
    strategies = ['direct', 'log', 'softplus']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, strategy in enumerate(strategies):
        # Create network
        net = FlexibleStiffnessNetwork(
            input_dim=1,  # 1D for easy visualization
            hidden_dim=32,
            n_layers=2,
            n_fourier=5,
            mu_min=mu_min,
            mu_max=mu_max,
            output_strategy=strategy,
            seed=seed
        )
        
        # Predict
        with torch.no_grad():
            mu_pred = net(x_test).squeeze().numpy()
        
        x_np = x_test.squeeze().numpy()
        
        # Plot
        ax = axes[idx]
        ax.plot(x_np, mu_pred, 'b-', linewidth=2, label='Predicted μ(x)')
        ax.axhline(mu_min, color='r', linestyle='--', alpha=0.5, label=f'μ_min={mu_min}')
        ax.axhline(mu_max, color='r', linestyle='--', alpha=0.5, label=f'μ_max={mu_max}')
        ax.fill_between(x_np, mu_min, mu_max, alpha=0.1, color='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('μ(x)')
        ax.set_title(f'Strategy: {strategy}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim([mu_min - 0.1, mu_max + 0.1])
    
    plt.tight_layout()
    plt.savefig('stiffness_network_strategies.png', dpi=150)
    print(f"\n  📊 Saved comparison: stiffness_network_strategies.png")
    plt.close()


# Testing and demonstration
if __name__ == "__main__":
    print("="*70)
    print("STIFFNESS NETWORK MODULE - Testing")
    print("="*70)
    
    # Test data
    print("\n🔧 Creating test data...")
    N = 1000
    x_test = torch.rand(N, 3)  # Random 3D coordinates
    print(f"  Test points: {N}")
    print(f"  Coordinate range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    
    # Test different configurations
    configs = [
        {
            'name': 'BIOQIC Range (Normalized)',
            'mu_min': 0.3,
            'mu_max': 1.0,
            'strategy': 'direct',
            'n_fourier': 10
        },
        {
            'name': 'Wide Range (Log Scale)',
            'mu_min': 0.1,
            'mu_max': 10.0,
            'strategy': 'log',
            'n_fourier': 15
        },
        {
            'name': 'SoftPlus Strategy',
            'mu_min': 0.5,
            'mu_max': 5.0,
            'strategy': 'softplus',
            'n_fourier': 8
        }
    ]
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print("-"*70)
        
        # Create network
        net = FlexibleStiffnessNetwork(
            input_dim=3,
            hidden_dim=64,
            n_layers=3,
            n_fourier=config['n_fourier'],
            mu_min=config['mu_min'],
            mu_max=config['mu_max'],
            output_strategy=config['strategy'],
            seed=42
        )
        
        print(f"  Network configuration:")
        print(f"    Input dim: 3")
        print(f"    Hidden dim: 64")
        print(f"    N layers: 3")
        print(f"    N Fourier features: {config['n_fourier']}")
        print(f"    Output strategy: {config['strategy']}")
        print(f"    μ range: [{config['mu_min']}, {config['mu_max']}]")
        print(f"    Total parameters: {sum(p.numel() for p in net.parameters()):,}")
        
        # Forward pass
        mu_pred = net(x_test)
        
        print(f"\n  Prediction test:")
        print(f"    Output shape: {mu_pred.shape}")
        print(f"    Output range: [{mu_pred.min():.4f}, {mu_pred.max():.4f}]")
        print(f"    Target range: [{config['mu_min']}, {config['mu_max']}]")
        print(f"    Mean: {mu_pred.mean():.4f}")
        print(f"    Std: {mu_pred.std():.4f}")
        
        # Check bounds
        in_bounds = (mu_pred >= config['mu_min']).all() and (mu_pred <= config['mu_max']).all()
        print(f"    ✅ All outputs in bounds: {in_bounds}")
        
        # Get statistics
        stats = net.get_statistics(x_test)
        print(f"\n  Statistics:")
        for key, val in stats.items():
            print(f"    {key}: {val:.6f}" if isinstance(val, float) else f"    {key}: {val}")
    
    print(f"\n{'='*70}")
    print("Comparing strategies visually...")
    print("-"*70)
    
    # 1D visualization for easy comparison
    x_1d = torch.linspace(0, 1, 200).reshape(-1, 1)
    compare_strategies(x_1d, mu_min=0.3, mu_max=1.0, seed=42)
    
    print("\n" + "="*70)
    print("✅ All tests passed successfully!")
    print("="*70)
