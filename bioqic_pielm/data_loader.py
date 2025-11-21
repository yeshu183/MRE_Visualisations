"""
BIOQIC Data Loader
==================

Loads BIOQIC FEM phantom data with flexible displacement representations.

Dataset: BIOQIC Phase 1 Box Phantom
- Grid: 80 x 100 x 10 mm (1mm voxels)
- Frequency: 60 Hz (ω = 377 rad/s)
- Material: Voigt viscoelastic (μ + iωη)
- Background: μ = 3 kPa, η = 1 Pa·s
- Inclusions: μ = 10 kPa (4 cylindrical targets)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple


class BIOQICDataLoader:
    """Load and preprocess BIOQIC Phase 1 box phantom data."""

    def __init__(
        self,
        data_dir: str = "../data/processed/phase1_box",
        displacement_mode: str = "z_component",
        subsample: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Path to preprocessed data
            displacement_mode: 'magnitude', 'z_component', or '3_components'
            subsample: Random subsample to this many points
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.displacement_mode = displacement_mode
        self.subsample = subsample
        self.seed = seed

        valid_modes = ['magnitude', 'z_component', '3_components']
        if displacement_mode not in valid_modes:
            raise ValueError(f"displacement_mode must be one of {valid_modes}")

        np.random.seed(seed)

    def load(self) -> Dict:
        """Load and preprocess BIOQIC data.

        Returns:
            Dictionary with coordinates, displacement, stiffness, and metadata
        """
        print("=" * 60)
        print("Loading BIOQIC Data")
        print("=" * 60)

        # Load arrays
        coords = np.load(self.data_dir / "coordinates.npy")
        coords_norm = np.load(self.data_dir / "coordinates_normalized.npy")
        displacement = np.load(self.data_dir / "displacement.npy")
        stiffness = np.load(self.data_dir / "stiffness_ground_truth.npy")
        params = np.load(self.data_dir / "preprocessing_params.npy", allow_pickle=True).item()

        print(f"  Grid: {params['grid_shape']}")
        print(f"  Frequency: {params['frequency_hz']} Hz")
        print(f"  Points: {len(coords):,}")

        # Subsample if requested
        if self.subsample and self.subsample < len(coords):
            print(f"  Subsampling: {len(coords):,} -> {self.subsample:,}")
            idx = np.random.choice(len(coords), self.subsample, replace=False)
            coords = coords[idx]
            coords_norm = coords_norm[idx]
            displacement = displacement[idx]
            stiffness = stiffness[idx]

        # Process displacement
        u_data = self._process_displacement(displacement)

        # Stiffness: use storage modulus (real part)
        mu_data = stiffness.real

        # Normalize to [0, 1]
        u_norm, u_scale = self._normalize(u_data)
        mu_norm, mu_scale = self._normalize(mu_data)

        # Physics parameters
        omega = 2 * np.pi * params['frequency_hz']
        rho = 1000.0  # kg/m³

        scales = {
            'u_min': float(u_data.min()),
            'u_max': float(u_data.max()),
            'u_scale': float(u_scale),
            'mu_min': float(mu_data.min()),
            'mu_max': float(mu_data.max()),
            'mu_scale': float(mu_scale),
            'omega': omega,
            'rho': rho,
            'rho_omega2_physical': rho * omega ** 2,
        }

        print(f"  Displacement range: [{u_data.min():.2e}, {u_data.max():.2e}] m")
        print(f"  Stiffness range: [{mu_data.min():.0f}, {mu_data.max():.0f}] Pa")
        print("=" * 60)

        return {
            'coords': coords,
            'coords_norm': coords_norm,
            'u_data': u_norm,
            'u_raw': u_data,
            'mu_data': mu_norm,
            'mu_raw': mu_data,
            'scales': scales,
            'params': params
        }

    def _process_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """Process displacement according to mode."""
        if self.displacement_mode == 'magnitude':
            u = np.sqrt(np.sum(np.abs(displacement) ** 2, axis=1, keepdims=True))
        elif self.displacement_mode == 'z_component':
            u = np.abs(displacement[:, 2:3])
        else:  # 3_components
            u = np.abs(displacement)
        return u

    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize to [0, 1]."""
        data_min, data_max = data.min(), data.max()
        scale = data_max - data_min
        if scale < 1e-12:
            return data, 1.0
        return (data - data_min) / scale, scale

    def to_tensors(self, data: Dict, device: torch.device) -> Dict:
        """Convert numpy arrays to torch tensors."""
        return {
            'x': torch.from_numpy(data['coords_norm']).float().to(device),
            'u_meas': torch.from_numpy(data['u_data']).float().to(device),
            'mu_true': torch.from_numpy(data['mu_data']).float().to(device),
            'coords': data['coords'],
            'coords_norm': data['coords_norm'],
            'scales': data['scales'],
            'params': data['params']
        }

    def get_boundary_indices(
        self,
        coords: np.ndarray,
        strategy: str = 'actuator',
        tolerance: float = 0.001
    ) -> np.ndarray:
        """Get boundary point indices.

        Args:
            coords: Physical coordinates (N, 3)
            strategy: 'actuator' (top Y-face) or 'minimal' (3 anchor points)
            tolerance: Distance tolerance in meters

        Returns:
            Array of boundary indices
        """
        if strategy == 'actuator':
            # Top Y-face (y = max, where actuator applies traction)
            y_max = coords[:, 1].max()
            bc_mask = np.abs(coords[:, 1] - y_max) < tolerance
            return np.where(bc_mask)[0]
        elif strategy == 'minimal':
            # 3 anchor points: corners
            n = len(coords)
            return np.array([0, n // 2, n - 1])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    loader = BIOQICDataLoader(subsample=5000)
    data = loader.load()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensors = loader.to_tensors(data, device)
    print(f"\nTensors on {device}:")
    print(f"  x: {tensors['x'].shape}")
    print(f"  u_meas: {tensors['u_meas'].shape}")
    print(f"  mu_true: {tensors['mu_true'].shape}")
