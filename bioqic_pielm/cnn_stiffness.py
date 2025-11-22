import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNStiffnessNetwork(nn.Module):
    """CNN-based parameterization of μ(x) on a 3D grid.

    - Maintains a learnable latent grid that is decoded via Conv3D to a mu-grid.
    - Provides trilinear sampling to return μ at arbitrary coordinates.

    Forward input:
    - x: (N,3) normalized coordinates in [0,1]

    Output:
    - mu: (N,1) stiffness values in [mu_min, mu_max]
    """

    def __init__(
        self,
        grid_shape=(50, 50, 10),
        latent_channels=8,
        hidden_channels=16,
        mu_min: float = 1.0,
        mu_max: float = 2.0,
        init_value: float = 1.07,
        seed: int = 0
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.grid_shape = tuple(grid_shape)  # (X, Y, Z) ordering matches data reshape
        self.mu_min = mu_min
        self.mu_max = mu_max

        D, H, W = self.grid_shape[2], self.grid_shape[1], self.grid_shape[0]

        # Learnable latent grid (small channel dimension)
        self.latent = nn.Parameter(torch.randn(1, latent_channels, D, H, W) * 0.01 + init_value)

        # Decoder conv stack -> produce single-channel mu grid
        layers = [
            nn.Conv3d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1),
        ]
        self.decoder = nn.Sequential(*layers)

        # Small initialization bias to produce initial flat field
        with torch.no_grad():
            for m in self.decoder.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return μ at given normalized coordinates x in [0,1].

        Args:
            x: (N,3) tensor with columns (x,y,z) normalized to [0,1]
        Returns:
            mu: (N,1) tensor in [mu_min, mu_max]
        """
        device = x.device

        # Decode latent to mu grid (1,1,D,H,W)
        mu_grid = self.decoder(self.latent)
        # Sigmoid to bound to [0,1], then scale to [mu_min, mu_max]
        mu_grid = torch.sigmoid(mu_grid)

        # Prepare sampling grid for F.grid_sample: coords in [-1,1]
        # grid_sample expects grid in (N, D_out, H_out, W_out, 3) with order (z, y, x)
        # We'll sample a set of points: construct grid of shape (1, N, 1, 1, 3)
        N = x.shape[0]
        # Convert x from (N,3) [X,Y,Z] normalized in [0,1] to [-1,1]
        grid_norm = x.clone().to(dtype=mu_grid.dtype, device=device)
        grid_norm = grid_norm * 2.0 - 1.0
        # grid_sample expects order (z, y, x) in the last dim
        # Our x is (x,y,z), so permute
        grid_sample_coords = grid_norm[:, [2, 1, 0]]  # (N,3) -> (z,y,x)
        # Reshape to (1, N, 1, 1, 3)
        grid_in = grid_sample_coords.view(1, N, 1, 1, 3)

        # Sample using grid_sample. Input must be (N_in, C, D, H, W)
        sampled = F.grid_sample(mu_grid, grid_in, mode='bilinear', padding_mode='border', align_corners=True)
        # sampled shape: (1, 1, N, 1, 1)
        sampled = sampled.view(1, 1, N).squeeze(0).squeeze(0)  # (N,)
        mu_norm = sampled.view(N, 1)

        # Map normalized mu to desired range
        mu = self.mu_min + (self.mu_max - self.mu_min) * mu_norm
        return mu


if __name__ == '__main__':
    # Quick unit test
    import torch
    net = CNNStiffnessNetwork(grid_shape=(50, 50, 10), mu_min=3000.0, mu_max=10000.0)
    x = torch.rand(200, 3)
    mu = net(x)
    print('mu:', mu.shape, mu.min().item(), mu.max().item())
