"""
Boundary Detection for BIOQIC MRE Data
========================================

Physics-informed boundary condition identification for MRE inverse problems.

Key Insight from BIOQIC Documentation:
--------------------------------------
- Box phantom: "Traction force applied on the top x-z plane"
- This means actuator is at TOP Y-face (y = y_max)
- Motion direction: "Mostly along z-axis" (vertical excitation)

Three strategies implemented:
1. Physics-based: Detect actuator at top Y-face
2. Data-only: Minimal anchoring, rely on measurements
3. Interior-weighted: No hard BCs, weight interior higher
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict


class BoundaryDetector:
    """Detect and define boundary conditions for MRE inverse problems."""
    
    def __init__(self, strategy: str = 'actuator'):
        """
        Initialize boundary detector.
        
        Args:
            strategy: 'actuator', 'minimal', or 'weighted'
                - 'actuator': Physics-based detection of top Y-face
                - 'minimal': Single anchor point for uniqueness
                - 'weighted': No explicit BCs, return weights for data loss
        """
        valid_strategies = ['actuator', 'minimal', 'weighted']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got {strategy}")
        
        self.strategy = strategy
        self.info = {}
    
    def detect(
        self, 
        coords: np.ndarray,
        coords_norm: np.ndarray,
        u_meas: torch.Tensor,
        device: torch.device,
        subsample: int = 5
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        Detect boundary conditions based on strategy.
        
        Args:
            coords: (N, 3) physical coordinates in meters [x, y, z]
            coords_norm: (N, 3) normalized coordinates in [0, 1]
            u_meas: (N, 1) measured displacement tensor
            device: torch device
            subsample: Subsampling factor for actuator points
            
        Returns:
            bc_indices: (K,) tensor of boundary point indices (or None)
            u_bc_vals: (K, 1) tensor of boundary values (or None)
            info: Dictionary with detection information
        """
        print(f"🎯 Boundary Detection Strategy: '{self.strategy}'")
        print("="*70)
        
        if self.strategy == 'actuator':
            return self._detect_actuator(coords, u_meas, device, subsample)
        elif self.strategy == 'minimal':
            return self._detect_minimal(u_meas, device)
        elif self.strategy == 'weighted':
            return self._detect_weighted(coords, len(u_meas), device)
        
    def _detect_actuator(
        self,
        coords: np.ndarray,
        u_meas: torch.Tensor,
        device: torch.device,
        subsample: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Physics-based: Detect actuator at top Y-face.
        
        Based on BIOQIC FEM setup:
        - Traction force applied on top x-z plane
        - This is at y = y_max (index 1 in coords)
        """
        print("\n📍 Physics-Based Actuator Detection")
        print("-" * 70)
        
        # Find top Y-face (where actuator applies traction)
        y_coords = coords[:, 1]  # Y is second dimension
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Physical tolerance: 1mm (single voxel layer)
        tol_physical = 0.001  # meters
        
        # Identify top face points
        top_face_mask = np.abs(y_coords - y_max) < tol_physical
        top_indices = np.where(top_face_mask)[0]
        
        # Also check bottom for comparison
        bottom_face_mask = np.abs(y_coords - y_min) < tol_physical
        bottom_indices = np.where(bottom_face_mask)[0]
        
        print(f"  Y-range: [{y_min*1000:.1f}, {y_max*1000:.1f}] mm")
        print(f"  Top face (y={y_max*1000:.1f}mm): {len(top_indices)} points")
        print(f"  Bottom face (y={y_min*1000:.1f}mm): {len(bottom_indices)} points")
        
        # Compute average displacement at each face
        u_meas_np = u_meas.cpu().numpy().flatten()
        top_disp_mean = u_meas_np[top_indices].mean()
        bottom_disp_mean = u_meas_np[bottom_indices].mean()
        
        print(f"\n  Displacement analysis:")
        print(f"    Top face avg: {top_disp_mean:.6e}")
        print(f"    Bottom face avg: {bottom_disp_mean:.6e}")
        print(f"    Ratio (top/bottom): {top_disp_mean/bottom_disp_mean:.2f}")
        
        # Use top face as actuator (matches BIOQIC documentation)
        bc_indices_full = top_indices
        
        # Subsample to reduce constraint
        bc_indices_np = bc_indices_full[::subsample]
        
        bc_indices = torch.from_numpy(bc_indices_np).long().to(device)
        u_bc_vals = u_meas[bc_indices]
        
        print(f"\n  ✅ Using top Y-face as actuator boundary")
        print(f"     Total points: {len(bc_indices_full)}")
        print(f"     Subsampled: {len(bc_indices)} (every {subsample}th point)")
        print(f"     Percentage of data: {100*len(bc_indices)/len(u_meas):.2f}%")
        print(f"     BC value range: [{u_bc_vals.min():.3e}, {u_bc_vals.max():.3e}]")
        
        info = {
            'n_bc_points': len(bc_indices),
            'n_total_points': len(u_meas),
            'bc_percentage': 100 * len(bc_indices) / len(u_meas),
            'top_face_points': len(top_indices),
            'subsample_factor': subsample,
            'actuator_location': 'top_y_face',
            'y_max': y_max
        }
        
        return bc_indices, u_bc_vals, info
    
    def _detect_minimal(
        self,
        u_meas: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Minimal anchoring: Use just enough points to prevent rigid body motion.
        
        Appropriate for inverse problems where:
        - We have measurements everywhere (full-field data)
        - Physics constraints come from data fitting, not geometry
        - High data_weight will enforce displacement matching
        """
        print("\n📍 Minimal Anchoring Strategy")
        print("-" * 70)
        print("  Philosophy: MRE inverse is data-driven, not boundary-driven")
        print("  Approach: Use 2-3 anchor points for solver uniqueness only")
        
        # Use 3 well-separated points to anchor the solution
        N = len(u_meas)
        anchor_indices = [0, N//2, N-1]
        
        bc_indices = torch.tensor(anchor_indices, dtype=torch.long, device=device)
        u_bc_vals = u_meas[bc_indices]
        
        print(f"\n  ✅ Using {len(bc_indices)} anchor points")
        print(f"     Indices: {anchor_indices}")
        print(f"     Percentage of data: {100*len(bc_indices)/N:.3f}%")
        print(f"     BC value range: [{u_bc_vals.min():.3e}, {u_bc_vals.max():.3e}]")
        print(f"\n  ⚠️  Requires HIGH data_weight (50-100+) for stability")
        
        info = {
            'n_bc_points': len(bc_indices),
            'n_total_points': N,
            'bc_percentage': 100 * len(bc_indices) / N,
            'strategy': 'minimal_anchoring'
        }
        
        return bc_indices, u_bc_vals, info
    
    def _detect_weighted(
        self,
        coords: np.ndarray,
        n_points: int,
        device: torch.device
    ) -> Tuple[None, None, Dict]:
        """
        Interior weighting: No explicit BCs, return weights for data loss.
        
        Returns None for bc_indices/bc_vals, but provides weight tensor
        that should be used in data loss computation.
        
        Strategy:
        - Interior points: weight = 1.0
        - Boundary points: weight = 0.1 (de-emphasized)
        - Allows boundaries to "float" during inversion
        """
        print("\n📍 Interior Weighting Strategy")
        print("-" * 70)
        print("  Philosophy: Don't enforce hard boundary constraints")
        print("  Approach: Weight interior data higher than boundary data")
        
        # Identify geometric boundaries
        tol = 0.001  # 1mm tolerance
        
        is_boundary = (
            (np.abs(coords[:, 0] - coords[:, 0].min()) < tol) |
            (np.abs(coords[:, 0] - coords[:, 0].max()) < tol) |
            (np.abs(coords[:, 1] - coords[:, 1].min()) < tol) |
            (np.abs(coords[:, 1] - coords[:, 1].max()) < tol) |
            (np.abs(coords[:, 2] - coords[:, 2].min()) < tol) |
            (np.abs(coords[:, 2] - coords[:, 2].max()) < tol)
        )
        
        # Create weights: interior=1.0, boundary=0.1
        weights = np.ones(n_points, dtype=np.float32)
        weights[is_boundary] = 0.1
        
        weights_tensor = torch.from_numpy(weights).to(device)
        
        n_boundary = is_boundary.sum()
        n_interior = n_points - n_boundary
        
        print(f"\n  ✅ Computed data weights")
        print(f"     Interior points: {n_interior:,} (weight=1.0)")
        print(f"     Boundary points: {n_boundary:,} (weight=0.1)")
        print(f"     Boundary fraction: {100*n_boundary/n_points:.1f}%")
        print(f"\n  📝 Usage: loss_data = torch.mean(weights * (u_pred - u_meas)**2)")
        
        info = {
            'n_interior': int(n_interior),
            'n_boundary': int(n_boundary),
            'n_total_points': n_points,
            'boundary_fraction': float(n_boundary / n_points),
            'weights': weights_tensor,
            'strategy': 'interior_weighting'
        }
        
        # Return None for bc_indices/bc_vals since we're not using explicit BCs
        return None, None, info


def visualize_boundary_detection(
    coords: np.ndarray,
    bc_indices: Optional[np.ndarray],
    strategy: str,
    save_path: str = 'boundary_detection.png'
):
    """
    Visualize detected boundary points.
    
    Args:
        coords: (N, 3) physical coordinates
        bc_indices: Boundary point indices (or None)
        strategy: Strategy name for title
        save_path: Where to save figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot all points
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                c='lightgray', s=1, alpha=0.3, label='All points')
    
    # Plot boundary points if available
    if bc_indices is not None:
        bc_coords = coords[bc_indices]
        ax1.scatter(bc_coords[:, 0], bc_coords[:, 1], bc_coords[:, 2],
                    c='red', s=10, label='Boundary points')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D View\n({strategy})')
    ax1.legend()
    
    # X-Y projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.3)
    if bc_indices is not None:
        ax2.scatter(bc_coords[:, 0], bc_coords[:, 1], c='red', s=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Projection')
    ax2.grid(alpha=0.3)
    
    # Y-Z projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(coords[:, 1], coords[:, 2], c='lightgray', s=1, alpha=0.3)
    if bc_indices is not None:
        ax3.scatter(bc_coords[:, 1], bc_coords[:, 2], c='red', s=10)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Y-Z Projection')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Saved visualization: {save_path}")
    plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("BOUNDARY DETECTION MODULE - Testing")
    print("="*70)
    
    # Create synthetic test data
    print("\n🔧 Creating test data...")
    N = 1000
    coords = np.random.rand(N, 3) * 0.1  # 100mm cube
    coords_norm = coords / 0.1
    u_meas = torch.rand(N, 1) * 0.01
    device = torch.device('cpu')
    
    print(f"  Grid: {N} points")
    print(f"  Coords range: [{coords.min():.3f}, {coords.max():.3f}] m")
    
    # Test all three strategies
    strategies = ['actuator', 'minimal', 'weighted']
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        detector = BoundaryDetector(strategy=strategy)
        bc_indices, bc_vals, info = detector.detect(
            coords, coords_norm, u_meas, device, subsample=5
        )
        
        print(f"\nℹ️  Info returned:")
        for key, val in info.items():
            if not isinstance(val, torch.Tensor):
                print(f"    {key}: {val}")
        
        print("="*70)
    
    print("\n✅ All strategies tested successfully!")
