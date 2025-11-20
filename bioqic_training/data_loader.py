"""
Data Loader for BIOQIC MRE Dataset
====================================

Flexible data loading with multiple displacement representations:
- Magnitude: |u| = sqrt(|u_x|^2 + |u_y|^2 + |u_z|^2)
- Single component: u_z (dominant for vertical excitation)
- Full 3-component: [u_x, u_y, u_z]
- Complex-valued: Real and imaginary parts

Also handles stiffness ground truth and coordinate grids.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Optional


class BIOQICDataLoader:
    """Load and preprocess BIOQIC Phase 1 box phantom data."""
    
    def __init__(
        self,
        data_dir: str = "../data/processed/phase1_box",
        displacement_mode: str = "magnitude",
        use_complex: bool = False,
        subsample: Optional[int] = None,
        frequency_idx: int = 1,  # 60 Hz
        seed: int = 42
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to preprocessed data directory
            displacement_mode: How to represent displacement
                - 'magnitude': Total magnitude across all components
                - 'z_component': Use only z-component (dominant)
                - '3_components': Use all [u_x, u_y, u_z]
            use_complex: If True, include imaginary part (double the output)
            subsample: If set, randomly subsample this many points
            frequency_idx: Which frequency to use (default 1 = 60 Hz)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.displacement_mode = displacement_mode
        self.use_complex = use_complex
        self.subsample = subsample
        self.frequency_idx = frequency_idx
        self.seed = seed
        
        # Validate mode
        valid_modes = ['magnitude', 'z_component', '3_components']
        if displacement_mode not in valid_modes:
            raise ValueError(f"displacement_mode must be one of {valid_modes}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Data containers
        self.coords = None
        self.coords_norm = None
        self.displacement = None
        self.stiffness = None
        self.params = None
        self.scales = None
    
    def load(self) -> Dict:
        """
        Load preprocessed BIOQIC data.
        
        Returns:
            Dictionary with all loaded arrays and metadata
        """
        print("="*70)
        print("BIOQIC DATA LOADER")
        print("="*70)
        print(f"\n📂 Loading from: {self.data_dir}")
        
        # Load arrays
        self.coords = np.load(self.data_dir / "coordinates.npy")
        self.coords_norm = np.load(self.data_dir / "coordinates_normalized.npy")
        self.displacement = np.load(self.data_dir / "displacement.npy")
        self.stiffness = np.load(self.data_dir / "stiffness_ground_truth.npy")
        self.params = np.load(self.data_dir / "preprocessing_params.npy", allow_pickle=True).item()
        
        print(f"  Coordinates: {self.coords.shape}")
        print(f"  Displacement: {self.displacement.shape} (complex)")
        print(f"  Stiffness: {self.stiffness.shape} (complex)")
        print(f"  Frequency: {self.params['frequency_hz']} Hz")
        print(f"  Grid: {self.params['grid_shape']}")
        print(f"  Voxel size: {self.params['voxel_size_m']*1000:.1f} mm")
        
        # Subsample if requested
        if self.subsample is not None and self.subsample < len(self.coords):
            print(f"\n🎲 Subsampling: {len(self.coords):,} → {self.subsample:,} points")
            indices = np.random.choice(len(self.coords), self.subsample, replace=False)
            self.coords = self.coords[indices]
            self.coords_norm = self.coords_norm[indices]
            self.displacement = self.displacement[indices]
            self.stiffness = self.stiffness[indices]
        
        # Process displacement based on mode
        u_data = self._process_displacement()
        
        # Process stiffness (use storage modulus = real part)
        mu_data = self.stiffness.real
        
        # Normalize
        u_normalized, u_scale = self._normalize_array(u_data, "Displacement")
        mu_normalized, mu_scale = self._normalize_array(mu_data, "Stiffness")
        
        # Store scales
        self.scales = {
            'u_scale': u_scale,
            'mu_scale': mu_scale,
            'u_min': u_data.min(),
            'u_max': u_data.max(),
            'mu_min': mu_data.min(),
            'mu_max': mu_data.max(),
            'mu_normalized_min': mu_normalized.min(),
            'mu_normalized_max': mu_normalized.max(),
            'frequency_hz': self.params['frequency_hz'],
            'omega': 2 * np.pi * self.params['frequency_hz'],
            'voxel_size_m': self.params['voxel_size_m']
        }
        
        print(f"\n✅ Data loaded and normalized")
        
        return {
            'coords': self.coords,
            'coords_norm': self.coords_norm,
            'u_data': u_normalized,
            'mu_data': mu_normalized,
            'scales': self.scales,
            'params': self.params
        }
    
    def _process_displacement(self) -> np.ndarray:
        """Process displacement field according to mode."""
        print(f"\n🌊 Processing displacement (mode: '{self.displacement_mode}')")
        
        if self.displacement_mode == 'magnitude':
            # Total magnitude: |u| = sqrt(|u_x|^2 + |u_y|^2 + |u_z|^2)
            u_mag = np.sqrt(
                np.abs(self.displacement[:, 0])**2 +
                np.abs(self.displacement[:, 1])**2 +
                np.abs(self.displacement[:, 2])**2
            )
            u_data = u_mag.reshape(-1, 1)
            print(f"  Using magnitude across all 3 components")
            print(f"  Output shape: {u_data.shape}")
            
        elif self.displacement_mode == 'z_component':
            # Z-component only (dominant for vertical excitation)
            u_z = np.abs(self.displacement[:, 2])
            u_data = u_z.reshape(-1, 1)
            print(f"  Using Z-component only (dominant direction)")
            print(f"  Output shape: {u_data.shape}")
            
        elif self.displacement_mode == '3_components':
            # All three components
            u_data = np.abs(self.displacement)  # (N, 3)
            print(f"  Using all 3 components [u_x, u_y, u_z]")
            print(f"  Output shape: {u_data.shape}")
        
        # Add complex part if requested
        if self.use_complex:
            print(f"  ⚠️  Complex mode not yet implemented, using magnitude only")
        
        print(f"  Range: [{u_data.min():.3e}, {u_data.max():.3e}] m")
        
        # Component-wise statistics
        if self.displacement_mode != 'magnitude':
            self._print_component_stats()
        
        return u_data
    
    def _print_component_stats(self):
        """Print statistics for each displacement component."""
        components = ['X', 'Y', 'Z']
        print(f"\n  Component-wise statistics:")
        for i, comp in enumerate(components):
            u_comp = np.abs(self.displacement[:, i])
            mean_val = u_comp.mean()
            print(f"    u_{comp}: mean={mean_val:.3e} m = {mean_val*1e6:.1f} μm")
    
    def _normalize_array(self, data: np.ndarray, name: str) -> Tuple[np.ndarray, float]:
        """
        Normalize array to [0, 1] range.
        
        Args:
            data: Array to normalize
            name: Name for logging
            
        Returns:
            normalized: Normalized array
            scale: Scale factor used
        """
        data_min = data.min()
        data_max = data.max()
        scale = data_max - data_min
        
        if scale < 1e-12:
            print(f"  ⚠️  {name} has zero range, skipping normalization")
            return data, 1.0
        
        # Normalize to [0, 1]
        normalized = (data - data_min) / scale
        
        print(f"\n  {name} normalization:")
        print(f"    Original: [{data_min:.3e}, {data_max:.3e}]")
        print(f"    Normalized: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"    Scale factor: {scale:.3e}")
        
        return normalized, scale
    
    def to_tensors(self, data_dict: Dict, device: torch.device) -> Dict:
        """
        Convert numpy arrays to torch tensors.
        
        Args:
            data_dict: Dictionary from load()
            device: Target device
            
        Returns:
            Dictionary with torch tensors
        """
        print(f"\n🔧 Converting to torch tensors (device: {device})")
        
        tensors = {
            'x': torch.from_numpy(data_dict['coords_norm']).float().to(device),
            'u_meas': torch.from_numpy(data_dict['u_data']).float().to(device),
            'mu_true': torch.from_numpy(data_dict['mu_data']).float().to(device),
            'coords': data_dict['coords'],  # Keep as numpy for boundary detection
            'coords_norm': data_dict['coords_norm'],  # Keep as numpy
            'scales': data_dict['scales'],
            'params': data_dict['params']
        }
        
        print(f"  x: {tensors['x'].shape} {tensors['x'].dtype}")
        print(f"  u_meas: {tensors['u_meas'].shape} {tensors['u_meas'].dtype}")
        print(f"  mu_true: {tensors['mu_true'].shape} {tensors['mu_true'].dtype}")
        
        return tensors
    
    def get_physics_params(self, strategy: str = 'effective') -> Dict:
        """
        Get physics parameters for PDE.
        
        Args:
            strategy: 'physical' or 'effective'
                - 'physical': Use actual ρω² = 1000 * (2πf)^2
                - 'effective': Use tuned value (e.g., 400) from synthetic tests
                
        Returns:
            Dictionary with physics parameters
        """
        omega = self.scales['omega']
        rho = 1000.0  # kg/m³ (tissue density)
        
        if strategy == 'physical':
            rho_omega2 = rho * omega**2
            print(f"\n⚙️  Physics parameters (PHYSICAL):")
            print(f"  ω = {omega:.1f} rad/s")
            print(f"  ρ = {rho:.1f} kg/m³")
            print(f"  ρω² = {rho_omega2:.1f} Pa/m²")
            print(f"  ⚠️  May need coordinate rescaling for Laplacian!")
            
        elif strategy == 'effective':
            rho_omega2 = 400.0  # Effective parameter from synthetic tests
            print(f"\n⚙️  Physics parameters (EFFECTIVE):")
            print(f"  ω = {omega:.1f} rad/s")
            print(f"  ρ = {rho:.1f} kg/m³")
            print(f"  ρω² (physical) = {rho * omega**2:.1f} Pa/m²")
            print(f"  ρω² (effective) = {rho_omega2:.1f} Pa/m²")
            print(f"  ℹ️  Using tuned value for stable inversion")
        
        else:
            raise ValueError(f"strategy must be 'physical' or 'effective', got {strategy}")
        
        return {
            'omega': omega,
            'rho': rho,
            'rho_omega2': rho_omega2,
            'strategy': strategy
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATA LOADER MODULE - Testing")
    print("="*70)
    
    # Test all displacement modes
    modes = ['magnitude', 'z_component', '3_components']
    
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Testing mode: {mode}")
        print(f"{'='*70}")
        
        loader = BIOQICDataLoader(
            displacement_mode=mode,
            subsample=1000,  # Use subset for testing
            seed=42
        )
        
        # Load data
        data = loader.load()
        
        # Get physics parameters (both strategies)
        print(f"\n{'='*70}")
        phys_physical = loader.get_physics_params('physical')
        print(f"\n{'-'*70}")
        phys_effective = loader.get_physics_params('effective')
        
        # Convert to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensors = loader.to_tensors(data, device)
        
        print(f"\n✅ Successfully loaded {mode} mode")
        print(f"{'='*70}\n")
    
    print("="*70)
    print("✅ All modes tested successfully!")
    print("="*70)
