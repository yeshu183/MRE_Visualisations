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
        seed: int = 42,
        adaptive_sampling: bool = False,
        blob_sample_ratio: float = 0.5,
        boundary_sample_ratio: float = 0.3,
        allow_replacement: bool = True
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Path to preprocessed data
            displacement_mode: 'magnitude', 'z_component', or '3_components'
            subsample: Random subsample to this many points
            seed: Random seed for reproducibility
            adaptive_sampling: Use adaptive sampling (more points near/in blobs)
            blob_sample_ratio: Fraction of samples inside blobs (default 0.5)
            boundary_sample_ratio: Fraction of samples near blob boundaries (default 0.3)
            allow_replacement: Allow sampling with replacement if region has fewer points than requested (default True)
        """
        self.data_dir = Path(data_dir)
        self.displacement_mode = displacement_mode
        self.subsample = subsample
        self.seed = seed
        self.adaptive_sampling = adaptive_sampling
        self.blob_sample_ratio = blob_sample_ratio
        self.boundary_sample_ratio = boundary_sample_ratio
        self.allow_replacement = allow_replacement

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
            if self.adaptive_sampling:
                print(f"  Using ADAPTIVE sampling (blob_ratio={self.blob_sample_ratio}, boundary_ratio={self.boundary_sample_ratio})")
                idx = self._adaptive_subsample(coords, stiffness, self.subsample)
            else:
                print(f"  Using UNIFORM random sampling")
                idx = np.random.choice(len(coords), self.subsample, replace=False)
            coords = coords[idx]
            coords_norm = coords_norm[idx]
            displacement = displacement[idx]
            stiffness = stiffness[idx]

        # Process displacement
        u_data = self._process_displacement(displacement)

        # Stiffness: use storage modulus (real part)
        mu_data = stiffness.real

        # CRITICAL: Normalize mu to [1, 2] range for PIELM solver
        # This matches the working approach folder which uses mu ~ [1, 2] with rho_omega2=400
        # Physical: 3000 Pa -> 1.0, 10000 Pa -> ~2.0
        mu_min_pa = float(mu_data.min())
        mu_max_pa = float(mu_data.max())
        mu_norm = 1.0 + (mu_data - mu_min_pa) / (mu_max_pa - mu_min_pa)  # Maps to [1, 2]
        mu_scale = mu_max_pa - mu_min_pa

        # Displacement: scale to reasonable range ~[-0.01, +0.01] (symmetric around zero)
        # For signed wave fields, we scale by max absolute value to preserve sign
        u_max_abs = np.abs(u_data).max()
        u_scale_factor = 0.01 / u_max_abs if u_max_abs > 0 else 1.0
        u_norm = u_data * u_scale_factor  # Now amplitude is ~0.01 (symmetric)

        # Physics parameters
        omega = 2 * np.pi * params['frequency_hz']
        rho = 1000.0  # kg/m³

        scales = {
            'u_min': float(u_data.min()),
            'u_max': float(u_data.max()),
            'u_scale': float(u_scale_factor),  # Multiply raw by this to get normalized
            'mu_min_pa': mu_min_pa,
            'mu_max_pa': mu_max_pa,
            'mu_scale': float(mu_scale),
            'mu_norm_min': 1.0,  # Normalized range
            'mu_norm_max': 2.0,
            'omega': omega,
            'rho': rho,
            'rho_omega2_physical': rho * omega ** 2,
            'rho_omega2_effective': 400.0,  # Tuned for normalized system
        }

        print(f"  Displacement range: [{u_data.min():.2e}, {u_data.max():.2e}] (raw)")
        print(f"  Displacement normalized: [{u_norm.min():.4f}, {u_norm.max():.4f}]")
        print(f"  Stiffness range: [{mu_min_pa:.0f}, {mu_max_pa:.0f}] Pa")
        print(f"  Stiffness normalized: [{mu_norm.min():.3f}, {mu_norm.max():.3f}]")
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

    def _detect_blob_regions(
        self,
        coords: np.ndarray,
        stiffness: np.ndarray,
        blob_threshold: float = 8000.0,
        boundary_width: float = 0.005
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect blob interior, boundary, and background regions.

        Args:
            coords: Physical coordinates (N, 3) in meters
            stiffness: Complex stiffness values (N,)
            blob_threshold: Threshold in Pa to classify as blob (default 8000 Pa)
            boundary_width: Width of boundary region in meters (default 5mm)

        Returns:
            blob_mask: Boolean mask for blob interior points
            boundary_mask: Boolean mask for boundary points
            background_mask: Boolean mask for background points
        """
        # Use real part (storage modulus) for classification
        mu_real = stiffness.real
        is_blob = (mu_real > blob_threshold).astype(bool)

        # Memory-efficient boundary detection using KDTree
        from scipy.spatial import cKDTree

        # Get indices
        blob_indices = np.where(is_blob)[0]
        background_indices = np.where(is_blob == False)[0]

        if len(blob_indices) == 0 or len(background_indices) == 0:
            print(f"  Warning: No boundary detection possible")
            boundary_mask = np.zeros(len(coords), dtype=bool)
            background_mask = np.logical_not(is_blob)
            return is_blob, boundary_mask, background_mask

        # Build trees
        blob_tree = cKDTree(coords[blob_indices])
        background_tree = cKDTree(coords[background_indices])

        # Find boundaries
        boundary_mask = np.zeros(len(coords), dtype=bool)

        # Blob points near background
        dists, _ = background_tree.query(coords[blob_indices], k=1)
        near_boundary = dists < boundary_width
        boundary_mask[blob_indices[near_boundary]] = True

        # Background points near blobs
        dists, _ = blob_tree.query(coords[background_indices], k=1)
        near_boundary = dists < boundary_width
        boundary_mask[background_indices[near_boundary]] = True

        # Interior regions (exclude boundaries)
        blob_interior = np.zeros(len(coords), dtype=bool)
        blob_interior[blob_indices] = True
        blob_interior[boundary_mask] = False

        background_interior = np.zeros(len(coords), dtype=bool)
        background_interior[background_indices] = True
        background_interior[boundary_mask] = False

        print(f"  Region detection:")
        print(f"    Blob interior: {int(blob_interior.sum()):,} points ({100*blob_interior.sum()/len(coords):.1f}%)")
        print(f"    Boundaries: {int(boundary_mask.sum()):,} points ({100*boundary_mask.sum()/len(coords):.1f}%)")
        print(f"    Background: {int(background_interior.sum()):,} points ({100*background_interior.sum()/len(coords):.1f}%)")

        return blob_interior, boundary_mask, background_interior

    def _adaptive_subsample(
        self,
        coords: np.ndarray,
        stiffness: np.ndarray,
        target_count: int
    ) -> np.ndarray:
        """Adaptive subsampling with more points in/near blobs.

        Args:
            coords: Physical coordinates (N, 3)
            stiffness: Stiffness values (N,)
            target_count: Total number of samples to draw

        Returns:
            Array of selected indices
        """
        # Detect regions
        blob_mask, boundary_mask, background_mask = self._detect_blob_regions(coords, stiffness)

        # Get indices for each region
        blob_indices = np.where(blob_mask)[0]
        boundary_indices = np.where(boundary_mask)[0]
        background_indices = np.where(background_mask)[0]

        # Allocate samples according to ratios
        # If allow_replacement=False, cap requested samples at available points
        if self.allow_replacement:
            n_blob = int(target_count * self.blob_sample_ratio)
            n_boundary = int(target_count * self.boundary_sample_ratio)
            n_background = target_count - n_blob - n_boundary
        else:
            # Cap samples at available points, redistribute excess to other regions
            n_blob_requested = int(target_count * self.blob_sample_ratio)
            n_boundary_requested = int(target_count * self.boundary_sample_ratio)

            n_blob = min(n_blob_requested, len(blob_indices))
            n_boundary = min(n_boundary_requested, len(boundary_indices))

            # Redistribute excess to background
            excess = (n_blob_requested - n_blob) + (n_boundary_requested - n_boundary)
            n_background = target_count - n_blob - n_boundary

            if excess > 0:
                print(f"    Note: Capped sampling (no replacement), redistributing {excess} samples to background")

        print(f"  Sample allocation:")
        print(f"    Blob samples: {n_blob:,} ({100*n_blob/target_count:.1f}%)")
        print(f"    Boundary samples: {n_boundary:,} ({100*n_boundary/target_count:.1f}%)")
        print(f"    Background samples: {n_background:,} ({100*n_background/target_count:.1f}%)")

        # Sample from each region
        selected_indices = []

        if len(blob_indices) > 0 and n_blob > 0:
            replace_blob = self.allow_replacement and (n_blob > len(blob_indices))
            blob_sample = np.random.choice(blob_indices, n_blob, replace=replace_blob)
            selected_indices.append(blob_sample)
            if replace_blob:
                print(f"    Warning: Blob region has only {len(blob_indices)} points, sampling with replacement")

        if len(boundary_indices) > 0 and n_boundary > 0:
            replace_boundary = self.allow_replacement and (n_boundary > len(boundary_indices))
            boundary_sample = np.random.choice(boundary_indices, n_boundary, replace=replace_boundary)
            selected_indices.append(boundary_sample)
            if replace_boundary:
                print(f"    Warning: Boundary region has only {len(boundary_indices)} points, sampling with replacement")

        if len(background_indices) > 0 and n_background > 0:
            replace_background = self.allow_replacement and (n_background > len(background_indices))
            background_sample = np.random.choice(background_indices, n_background, replace=replace_background)
            selected_indices.append(background_sample)

        # Combine all selected indices
        all_indices = np.concatenate(selected_indices)

        # Shuffle to mix regions
        np.random.shuffle(all_indices)

        return all_indices

    def _process_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """Process displacement according to mode.
        
        CRITICAL: For Helmholtz equation solving, we need SIGNED displacement values.
        The wave field oscillates around zero (positive and negative).
        Taking absolute values destroys phase information and creates non-physical cusps.
        """
        if self.displacement_mode == 'magnitude':
            # Magnitude is okay for visualization, but NOT for Helmholtz inversion
            u = np.sqrt(np.sum(np.abs(displacement) ** 2, axis=1, keepdims=True))
        elif self.displacement_mode == 'z_component':
            # FIX: Take the REAL part (signed), do not use abs()
            # Assuming displacement might be complex-valued in the .npy file
            if np.iscomplexobj(displacement):
                u = displacement[:, 2:3].real
            else:
                u = displacement[:, 2:3]
        else:  # 3_components
            # FIX: Use real part for complex, raw values otherwise
            if np.iscomplexobj(displacement):
                u = displacement.real
            else:
                u = displacement
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
