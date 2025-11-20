"""
Comprehensive exploration and visualization of BIOQIC phantom data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the preprocessed BIOQIC data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box'
    
    coords = np.load(data_dir / 'coordinates.npy')
    coords_norm = np.load(data_dir / 'coordinates_normalized.npy')
    displacement = np.load(data_dir / 'displacement.npy')
    stiffness = np.load(data_dir / 'stiffness_ground_truth.npy')
    params = np.load(data_dir / 'preprocessing_params.npy', allow_pickle=True).item()
    
    print("=" * 80)
    print("BIOQIC Phase 1 Box Phantom Data")
    print("=" * 80)
    print("\nLoaded data:")
    print(f"  Coordinates: {coords.shape} ({coords.dtype})")
    print(f"  Coordinates (normalized): {coords_norm.shape} ({coords_norm.dtype})")
    print(f"  Displacement: {displacement.shape} ({displacement.dtype})")
    print(f"  Stiffness: {stiffness.shape} ({stiffness.dtype})")
    
    return coords, coords_norm, displacement, stiffness, params

def explore_geometry(coords, params):
    """Explore the geometry and grid structure."""
    print("\n" + "=" * 80)
    print("GEOMETRY INFORMATION")
    print("=" * 80)
    
    print(f"\nGrid shape: {params['grid_shape']}")
    print(f"Voxel size: {params['voxel_size_m']*1000:.3f} mm")
    print(f"Frequency: {params['frequency_hz']} Hz")
    
    # Coordinate ranges
    x_range = [coords[:, 0].min(), coords[:, 0].max()]
    y_range = [coords[:, 1].min(), coords[:, 1].max()]
    z_range = [coords[:, 2].min(), coords[:, 2].max()]
    
    print(f"\nPhysical domain size:")
    print(f"  X: [{x_range[0]:.4f}, {x_range[1]:.4f}] m  (range: {x_range[1]-x_range[0]:.4f} m = {(x_range[1]-x_range[0])*1000:.1f} mm)")
    print(f"  Y: [{y_range[0]:.4f}, {y_range[1]:.4f}] m  (range: {y_range[1]-y_range[0]:.4f} m = {(y_range[1]-y_range[0])*1000:.1f} mm)")
    print(f"  Z: [{z_range[0]:.4f}, {z_range[1]:.4f}] m  (range: {z_range[1]-z_range[0]:.4f} m = {(z_range[1]-z_range[0])*1000:.1f} mm)")
    
    # Plot coordinate distribution
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(coords[::100, 0]*1000, coords[::100, 1]*1000, coords[::100, 2]*1000,
                         c=coords[::100, 2]*1000, cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Point Cloud (subsampled 1:100)')
    plt.colorbar(scatter, ax=ax1, label='Z (mm)', shrink=0.6)
    
    # XY plane
    ax2 = fig.add_subplot(132)
    ax2.scatter(coords[::10, 0]*1000, coords[::10, 1]*1000, s=0.5, alpha=0.3)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Plane View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # XZ plane
    ax3 = fig.add_subplot(133)
    ax3.scatter(coords[::10, 0]*1000, coords[::10, 2]*1000, s=0.5, alpha=0.3)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Plane View')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'data_geometry.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: data_exploration/outputs/data_geometry.png")
    plt.close()

def explore_displacement(coords, displacement):
    """Explore displacement field properties."""
    print("\n" + "=" * 80)
    print("DISPLACEMENT FIELD ANALYSIS")
    print("=" * 80)
    
    # Compute magnitude and phase
    u_mag = np.abs(displacement)
    u_phase = np.angle(displacement)
    
    # Statistics for each component
    print("\nDisplacement statistics (complex):")
    components = ['X', 'Y', 'Z']
    for i, comp in enumerate(components):
        mag = u_mag[:, i]
        phase = u_phase[:, i]
        print(f"\n  Component {comp}:")
        print(f"    Magnitude: [{mag.min():.6f}, {mag.max():.6f}] m = [{mag.min()*1e6:.2f}, {mag.max()*1e6:.2f}] μm")
        print(f"    Mean: {mag.mean():.6f} m = {mag.mean()*1e6:.2f} μm, Std: {mag.std():.6f} m")
        print(f"    Phase: [{phase.min():.3f}, {phase.max():.3f}] rad")
    
    # Total displacement magnitude
    u_total_mag = np.sqrt(np.sum(u_mag**2, axis=1))
    print(f"\nTotal displacement magnitude:")
    print(f"  Range: [{u_total_mag.min():.6f}, {u_total_mag.max():.6f}] m")
    print(f"  Range: [{u_total_mag.min()*1e6:.2f}, {u_total_mag.max()*1e6:.2f}] μm")
    print(f"  Mean: {u_total_mag.mean():.6f} m = {u_total_mag.mean()*1e6:.2f} μm")
    print(f"  Std: {u_total_mag.std():.6f} m = {u_total_mag.std()*1e6:.2f} μm")
    
    # Plot displacement fields
    fig = plt.figure(figsize=(18, 12))
    
    # Magnitude for each component
    for i, comp in enumerate(components):
        ax = fig.add_subplot(3, 4, i*4 + 1, projection='3d')
        scatter = ax.scatter(coords[::50, 0]*1000, coords[::50, 1]*1000, coords[::50, 2]*1000,
                           c=u_mag[::50, i]*1e6, cmap='hot', s=10, alpha=0.6)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{comp} Magnitude')
        plt.colorbar(scatter, ax=ax, label='|u| (μm)', shrink=0.6)
        
        # Phase
        ax = fig.add_subplot(3, 4, i*4 + 2, projection='3d')
        scatter = ax.scatter(coords[::50, 0]*1000, coords[::50, 1]*1000, coords[::50, 2]*1000,
                           c=u_phase[::50, i], cmap='twilight', s=10, alpha=0.6, vmin=-np.pi, vmax=np.pi)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{comp} Phase')
        plt.colorbar(scatter, ax=ax, label='φ (rad)', shrink=0.6)
        
        # Histogram of magnitude
        ax = fig.add_subplot(3, 4, i*4 + 3)
        ax.hist(u_mag[:, i]*1e6, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'|u_{comp}| (μm)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{comp} Magnitude Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Histogram of phase
        ax = fig.add_subplot(3, 4, i*4 + 4)
        ax.hist(u_phase[:, i], bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel(f'φ_{comp} (rad)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{comp} Phase Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / 'outputs'
    plt.savefig(output_dir / 'data_displacement_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_displacement_detailed.png")
    plt.close()
    
    # Additional: Component comparison on log scale
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, comp in enumerate(['X', 'Y', 'Z']):
        ax = axes[i]
        mag = u_mag[:, i] * 1e6  # Convert to μm
        
        # Create bins on log scale
        bins = np.logspace(np.log10(mag[mag > 0].min()), np.log10(mag.max()), 50)
        ax.hist(mag, bins=bins, alpha=0.7, edgecolor='black', color=['blue', 'green', 'red'][i])
        ax.set_xscale('log')
        ax.set_xlabel(f'|u_{comp}| (μm)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{comp}-component (Log Scale)')
        ax.grid(True, alpha=0.3, which='both')
        
        # Add statistics text
        stats_text = f'Mean: {mag.mean():.1f} μm\nMax: {mag.max():.1f} μm'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Displacement Component Distributions (Note: Z-component dominates)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_displacement_components_log.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_displacement_components_log.png")
    plt.close()
    
    # Plot total magnitude in 3D
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(coords[::50, 0]*1000, coords[::50, 1]*1000, coords[::50, 2]*1000,
                         c=u_total_mag[::50]*1e6, cmap='plasma', s=10, alpha=0.6)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Total Displacement Magnitude')
    plt.colorbar(scatter, ax=ax1, label='|u| (μm)', shrink=0.8)
    
    ax2 = fig.add_subplot(132)
    ax2.hist(u_total_mag*1e6, bins=100, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total |u| (μm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Total Magnitude Distribution')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(133)
    h = ax3.hist2d(coords[:, 2]*1000, u_total_mag*1e6, bins=50, cmap='hot')
    ax3.set_xlabel('Z position (mm)')
    ax3.set_ylabel('|u| (μm)')
    ax3.set_title('Displacement vs Z Position')
    plt.colorbar(h[3], ax=ax3, label='Count')
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / 'outputs'
    plt.savefig(output_dir / 'data_displacement_total.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_displacement_total.png")
    plt.close()

def explore_stiffness(coords, stiffness, params):
    """Explore ground truth stiffness distribution."""
    print("\n" + "=" * 80)
    print("STIFFNESS FIELD ANALYSIS")
    print("=" * 80)
    
    # Real part (storage modulus)
    mu_real = stiffness.real.flatten()
    # Imaginary part (loss modulus)
    mu_imag = stiffness.imag.flatten()
    # Magnitude
    mu_mag = np.abs(stiffness).flatten()
    
    # Identify distinct regions (targets)
    unique_stiff = np.unique(np.round(mu_real, 0))
    print(f"\nIdentified {len(unique_stiff)} distinct stiffness regions:")
    for i, val in enumerate(unique_stiff):
        count = np.sum(np.round(mu_real, 0) == val)
        pct = 100 * count / len(mu_real)
        print(f"  Region {i+1}: {val:.0f} Pa ({count:,} voxels, {pct:.1f}%)")
    
    print(f"\nReal part (μ' - storage modulus):")
    print(f"  Range: [{mu_real.min():.1f}, {mu_real.max():.1f}] Pa")
    print(f"  Mean: {mu_real.mean():.1f} Pa, Std: {mu_real.std():.1f} Pa")
    print(f"  Unique values: {len(np.unique(np.round(mu_real, -2)))}")
    
    print(f"\nImaginary part (μ'' - loss modulus):")
    print(f"  Range: [{mu_imag.min():.1f}, {mu_imag.max():.1f}] Pa")
    print(f"  Mean: {mu_imag.mean():.1f} Pa, Std: {mu_imag.std():.1f} Pa")
    
    print(f"\nMagnitude (|μ|):")
    print(f"  Range: [{mu_mag.min():.1f}, {mu_mag.max():.1f}] Pa")
    print(f"  Mean: {mu_mag.mean():.1f} Pa, Std: {mu_mag.std():.1f} Pa")
    
    # Loss tangent
    loss_tangent = mu_imag / (mu_real + 1e-10)
    print(f"\nLoss tangent (μ''/μ'):")
    print(f"  Range: [{loss_tangent.min():.4f}, {loss_tangent.max():.4f}]")
    print(f"  Mean: {loss_tangent.mean():.4f}, Std: {loss_tangent.std():.4f}")
    
    # Plot stiffness distribution
    fig = plt.figure(figsize=(18, 10))
    
    # 3D scatter of real part
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(coords[::50, 0]*1000, coords[::50, 1]*1000, coords[::50, 2]*1000,
                         c=mu_real[::50], cmap='coolwarm', s=10, alpha=0.6)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title("Storage Modulus μ' (Pa)")
    plt.colorbar(scatter, ax=ax1, label='μ\' (Pa)', shrink=0.8)
    
    # Histogram of real part
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(mu_real, bins=50, alpha=0.7, edgecolor='black', color='blue')
    ax2.set_xlabel('μ\' (Pa)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Storage Modulus Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 2D slice at mid-Z
    grid_shape = params['grid_shape']
    mid_z = grid_shape[2] // 2
    mu_grid = mu_real.reshape(grid_shape)
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(mu_grid[:, :, mid_z].T, origin='lower', cmap='coolwarm', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Y index')
    ax3.set_ylabel('X index')
    ax3.set_title(f'μ\' at Z = {mid_z} (mid-plane) - Showing 4 Targets')
    plt.colorbar(im, ax=ax3, label='μ\' (Pa)')
    
    # Add contours to show target boundaries
    levels = [mu_real.min() + 0.1*(mu_real.max()-mu_real.min())]
    ax3.contour(mu_grid[:, :, mid_z].T, levels=levels, colors='white', linewidths=1.5, alpha=0.7)
    
    # 3D scatter of imaginary part
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter = ax4.scatter(coords[::50, 0]*1000, coords[::50, 1]*1000, coords[::50, 2]*1000,
                         c=mu_imag[::50], cmap='viridis', s=10, alpha=0.6)
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    ax4.set_title("Loss Modulus μ'' (Pa)")
    plt.colorbar(scatter, ax=ax4, label='μ\'\' (Pa)', shrink=0.8)
    
    # Loss tangent distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(loss_tangent, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax5.set_xlabel('μ\'\'/μ\'')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Loss Tangent Distribution')
    ax5.grid(True, alpha=0.3)
    
    # Real vs Imaginary scatter
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(mu_real[::10], mu_imag[::10], s=1, alpha=0.3)
    ax6.set_xlabel('μ\' (Pa)')
    ax6.set_ylabel('μ\'\' (Pa)')
    ax6.set_title('Storage vs Loss Modulus')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / 'outputs'
    plt.savefig(output_dir / 'data_stiffness.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_stiffness.png")
    plt.close()
    
    # Additional plot: Multiple Z-slices to show 3D structure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, z_idx in enumerate(range(0, grid_shape[2], max(1, grid_shape[2]//10))):
        if idx < len(axes):
            ax = axes[idx]
            im = ax.imshow(mu_grid[:, :, z_idx].T, origin='lower', cmap='coolwarm', 
                          aspect='auto', interpolation='nearest',
                          vmin=mu_real.min(), vmax=mu_real.max())
            ax.set_title(f'Z = {z_idx} ({z_idx*params["voxel_size_m"]*1000:.1f} mm)')
            ax.set_xlabel('Y')
            ax.set_ylabel('X')
            ax.contour(mu_grid[:, :, z_idx].T, levels=levels, colors='white', linewidths=1, alpha=0.5)
    
    # Remove empty subplots
    for idx in range(len(range(0, grid_shape[2], max(1, grid_shape[2]//10))), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.colorbar(im, ax=axes, label='μ\' (Pa)', fraction=0.046, pad=0.04)
    plt.suptitle('Stiffness Distribution Across Z-Slices (4 Targets Visible)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_stiffness_slices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_stiffness_slices.png")
    plt.close()

def explore_wave_propagation(coords, displacement, params):
    """Analyze wave propagation characteristics."""
    print("\n" + "=" * 80)
    print("WAVE PROPAGATION ANALYSIS")
    print("=" * 80)
    
    freq = params['frequency_hz']
    omega = 2 * np.pi * freq
    
    print(f"\nFrequency: {freq} Hz")
    print(f"Angular frequency (ω): {omega:.2f} rad/s")
    
    # Analyze phase patterns
    u_phase = np.angle(displacement)
    
    # Dominant displacement component
    u_mag = np.abs(displacement)
    dominant_comp = np.argmax(u_mag, axis=1)
    
    print(f"\nDominant displacement component distribution:")
    for i, comp in enumerate(['X', 'Y', 'Z']):
        count = np.sum(dominant_comp == i)
        pct = 100 * count / len(dominant_comp)
        print(f"  {comp}: {count} points ({pct:.1f}%)")
    
    # Plot phase propagation
    fig = plt.figure(figsize=(18, 6))
    
    for i, comp in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        scatter = ax.scatter(coords[::50, 0]*1000, coords[::50, 1]*1000, coords[::50, 2]*1000,
                           c=u_phase[::50, i], cmap='hsv', s=10, alpha=0.6,
                           vmin=-np.pi, vmax=np.pi)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Phase Pattern - {comp} Component')
        plt.colorbar(scatter, ax=ax, label='Phase (rad)', shrink=0.8)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / 'outputs'
    plt.savefig(output_dir / 'data_wave_phase.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_wave_phase.png")
    plt.close()

def explore_spatial_variations(coords, displacement, stiffness):
    """Explore how displacement and stiffness vary spatially."""
    print("\n" + "=" * 80)
    print("SPATIAL VARIATION ANALYSIS")
    print("=" * 80)
    
    u_mag_total = np.sqrt(np.sum(np.abs(displacement)**2, axis=1))
    mu_real = stiffness.real.flatten()
    
    print("\nSpatial correlation analysis:")
    print(f"  Correlation (X vs |u|): {np.corrcoef(coords[:, 0], u_mag_total)[0, 1]:.4f}")
    print(f"  Correlation (Y vs |u|): {np.corrcoef(coords[:, 1], u_mag_total)[0, 1]:.4f}")
    print(f"  Correlation (Z vs |u|): {np.corrcoef(coords[:, 2], u_mag_total)[0, 1]:.4f}")
    print(f"  Correlation (|u| vs μ'): {np.corrcoef(u_mag_total, mu_real)[0, 1]:.4f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # X variation
    axes[0, 0].scatter(coords[::10, 0]*1000, u_mag_total[::10]*1e6, s=1, alpha=0.3)
    axes[0, 0].set_xlabel('X position (mm)')
    axes[0, 0].set_ylabel('|u| (μm)')
    axes[0, 0].set_title('Displacement vs X')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(coords[::10, 0]*1000, mu_real[::10], s=1, alpha=0.3, color='red')
    axes[1, 0].set_xlabel('X position (mm)')
    axes[1, 0].set_ylabel('μ\' (Pa)')
    axes[1, 0].set_title('Stiffness vs X')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Y variation
    axes[0, 1].scatter(coords[::10, 1]*1000, u_mag_total[::10]*1e6, s=1, alpha=0.3)
    axes[0, 1].set_xlabel('Y position (mm)')
    axes[0, 1].set_ylabel('|u| (μm)')
    axes[0, 1].set_title('Displacement vs Y')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(coords[::10, 1]*1000, mu_real[::10], s=1, alpha=0.3, color='red')
    axes[1, 1].set_xlabel('Y position (mm)')
    axes[1, 1].set_ylabel('μ\' (Pa)')
    axes[1, 1].set_title('Stiffness vs Y')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Z variation
    axes[0, 2].scatter(coords[::10, 2]*1000, u_mag_total[::10]*1e6, s=1, alpha=0.3)
    axes[0, 2].set_xlabel('Z position (mm)')
    axes[0, 2].set_ylabel('|u| (μm)')
    axes[0, 2].set_title('Displacement vs Z')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(coords[::10, 2]*1000, mu_real[::10], s=1, alpha=0.3, color='red')
    axes[1, 2].set_xlabel('Z position (mm)')
    axes[1, 2].set_ylabel('μ\' (Pa)')
    axes[1, 2].set_title('Stiffness vs Z')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / 'outputs'
    plt.savefig(output_dir / 'data_spatial_variations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_spatial_variations.png")
    plt.close()

def generate_summary_report(coords, displacement, stiffness, params):
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    u_total = np.sqrt(np.sum(np.abs(displacement)**2, axis=1))
    mu_real = stiffness.real.flatten()
    mu_imag = stiffness.imag.flatten()
    
    report = f"""
BIOQIC Phase 1 Box Phantom - Data Summary
==========================================

Dataset Properties:
-------------------
- Grid shape: {params['grid_shape']}
- Total points: {len(coords):,}
- Voxel size: {params['voxel_size_m']*1000:.3f} mm
- Frequency: {params['frequency_hz']} Hz
- Angular frequency: {2*np.pi*params['frequency_hz']:.2f} rad/s

Physical Domain:
----------------
- X range: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}] m = [{coords[:, 0].min()*1000:.1f}, {coords[:, 0].max()*1000:.1f}] mm
- Y range: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}] m = [{coords[:, 1].min()*1000:.1f}, {coords[:, 1].max()*1000:.1f}] mm
- Z range: [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}] m = [{coords[:, 2].min()*1000:.1f}, {coords[:, 2].max()*1000:.1f}] mm
- Volume: ~{(coords[:, 0].max()-coords[:, 0].min())*(coords[:, 1].max()-coords[:, 1].min())*(coords[:, 2].max()-coords[:, 2].min())*1e6:.2f} cm³

Displacement Field:
-------------------
- Total magnitude range: [{u_total.min():.6f}, {u_total.max():.6f}] m = [{u_total.min()*1e6:.2f}, {u_total.max()*1e6:.2f}] μm
- Mean total displacement: {u_total.mean():.6f} m = {u_total.mean()*1e6:.2f} μm
- Component X: [{np.abs(displacement[:, 0]).min():.6f}, {np.abs(displacement[:, 0]).max():.6f}] m = [{np.abs(displacement[:, 0]).min()*1e6:.2f}, {np.abs(displacement[:, 0]).max()*1e6:.2f}] μm
- Component Y: [{np.abs(displacement[:, 1]).min():.6f}, {np.abs(displacement[:, 1]).max():.6f}] m = [{np.abs(displacement[:, 1]).min()*1e6:.2f}, {np.abs(displacement[:, 1]).max()*1e6:.2f}] μm
- Component Z: [{np.abs(displacement[:, 2]).min():.6f}, {np.abs(displacement[:, 2]).max():.6f}] m = [{np.abs(displacement[:, 2]).min()*1e6:.2f}, {np.abs(displacement[:, 2]).max()*1e6:.2f}] μm

Stiffness Field (Ground Truth):
-------------------------------
- Storage modulus (μ'): [{mu_real.min():.1f}, {mu_real.max():.1f}] Pa
- Mean μ': {mu_real.mean():.1f} ± {mu_real.std():.1f} Pa
- Loss modulus (μ''): [{mu_imag.min():.1f}, {mu_imag.max():.1f}] Pa
- Mean μ'': {mu_imag.mean():.1f} ± {mu_imag.std():.1f} Pa
- Loss tangent: {(mu_imag.mean()/(mu_real.mean()+1e-10)):.4f}
- Stiffness contrast: {mu_real.max()/mu_real.min():.2f}× (soft to stiff)

Key Observations:
-----------------
1. Displacement is on the order of ~{u_total.mean()*1e6:.1f} μm (typical for MRE at {params['frequency_hz']} Hz)
2. Stiffness range is {mu_real.max()/mu_real.min():.1f}× (from {mu_real.min():.0f} to {mu_real.max():.0f} Pa)
3. Background stiffness: ~{mu_real.min():.0f} Pa (soft gel)
4. Inclusion stiffness: ~{mu_real.max():.0f} Pa (stiffer targets)
5. Loss tangent ~{(mu_imag.mean()/(mu_real.mean()+1e-10)):.3f} suggests {'low' if (mu_imag.mean()/(mu_real.mean()+1e-10)) < 0.1 else 'moderate'} viscoelasticity
6. Data appears to be clean FEM simulation (low noise, well-defined boundaries)

Correlation Analysis:
---------------------
- Displacement vs X: {np.corrcoef(coords[:, 0], u_total)[0, 1]:.4f}
- Displacement vs Y: {np.corrcoef(coords[:, 1], u_total)[0, 1]:.4f}
- Displacement vs Z: {np.corrcoef(coords[:, 2], u_total)[0, 1]:.4f}
- Displacement vs Stiffness: {np.corrcoef(u_total, mu_real)[0, 1]:.4f}

Recommended Next Steps:
-----------------------
1. **Use full 3-component complex displacement** for richer constraints
2. **Identify actual excitation boundaries** from displacement/phase patterns
3. **Test on 2D slice** (e.g., mid-Z plane) before full 3D
4. **Subsample intelligently** (keep high-gradient regions, boundaries)
5. **Consider multi-frequency** if additional frequencies available
6. **Use proper physics scaling** (ρω² with Laplacian rescaling)
7. **Regularize toward mean stiffness** rather than unconstrained
"""
    
    print(report)
    
    # Save to file with UTF-8 encoding to handle special characters
    output_dir = Path(__file__).parent / 'outputs'
    with open(output_dir / 'data_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n✓ Saved: data_exploration/outputs/data_summary_report.txt")
    
    # Create a comprehensive comparison figure
    fig = plt.figure(figsize=(18, 10))
    
    # Top row: Stiffness distribution
    ax1 = fig.add_subplot(2, 3, 1)
    grid_shape = params['grid_shape']
    mid_z = grid_shape[2] // 2
    mu_grid = stiffness.real.flatten().reshape(grid_shape)
    im1 = ax1.imshow(mu_grid[:, :, mid_z].T, origin='lower', cmap='coolwarm', aspect='auto')
    ax1.set_xlabel('Y index')
    ax1.set_ylabel('X index')
    ax1.set_title('Ground Truth Stiffness (4 Targets)')
    plt.colorbar(im1, ax=ax1, label='μ\' (Pa)')
    
    # Displacement components at same slice
    u_grid = displacement.reshape((grid_shape[0], grid_shape[1], grid_shape[2], 3))
    
    for i, comp in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(2, 3, i+2)
        u_slice = np.abs(u_grid[:, :, mid_z, i].T) * 1e6  # μm
        im = ax.imshow(u_slice, origin='lower', cmap='hot', aspect='auto')
        ax.set_xlabel('Y index')
        ax.set_ylabel('X index')
        ax.set_title(f'|u_{comp}| at Z={mid_z}')
        plt.colorbar(im, ax=ax, label='μm')
    
    # Bottom row: Statistics
    ax5 = fig.add_subplot(2, 3, 5)
    u_total = np.sqrt(np.sum(np.abs(displacement)**2, axis=1))
    mu_real_flat = stiffness.real.flatten()
    
    # Scatter: displacement vs stiffness
    sample_idx = np.random.choice(len(u_total), 5000, replace=False)
    ax5.scatter(mu_real_flat[sample_idx], u_total[sample_idx]*1e6, 
               s=1, alpha=0.3, c=coords[sample_idx, 2]*1000, cmap='viridis')
    ax5.set_xlabel('Stiffness μ\' (Pa)')
    ax5.set_ylabel('|u| (μm)')
    ax5.set_title('Displacement vs Stiffness')
    ax5.grid(True, alpha=0.3)
    cb = plt.colorbar(ax5.collections[0], ax=ax5, label='Z (mm)')
    
    # Component ratios
    ax6 = fig.add_subplot(2, 3, 6)
    u_mag = np.abs(displacement)
    ratios = u_mag / (u_total.reshape(-1, 1) + 1e-10)
    ax6.boxplot([ratios[:, 0], ratios[:, 1], ratios[:, 2]], 
                labels=['X', 'Y', 'Z'])
    ax6.set_ylabel('Component Fraction of Total')
    ax6.set_title('Displacement Component Contributions')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    plt.suptitle(f'BIOQIC Phase 1 Box Phantom Summary ({params["frequency_hz"]} Hz)', 
                fontsize=16, y=0.995)
    plt.tight_layout()
    
    output_dir = Path(__file__).parent / 'outputs'
    plt.savefig(output_dir / 'data_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_exploration/outputs/data_comprehensive_summary.png")
    plt.close()

def main():
    """Main exploration workflow."""
    # Load data
    coords, coords_norm, displacement, stiffness, params = load_data()
    
    # Run all explorations
    explore_geometry(coords, params)
    explore_displacement(coords, displacement)
    explore_stiffness(coords, stiffness, params)
    explore_wave_propagation(coords, displacement, params)
    explore_spatial_variations(coords, displacement, stiffness)
    generate_summary_report(coords, displacement, stiffness, params)
    
    print("\n" + "=" * 80)
    print("✅ Data exploration complete!")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  - data_exploration/outputs/data_geometry.png")
    print("  - data_exploration/outputs/data_displacement_detailed.png")
    print("  - data_exploration/outputs/data_displacement_components_log.png")
    print("  - data_exploration/outputs/data_displacement_total.png")
    print("  - data_exploration/outputs/data_stiffness.png")
    print("  - data_exploration/outputs/data_stiffness_slices.png")
    print("  - data_exploration/outputs/data_wave_phase.png")
    print("  - data_exploration/outputs/data_spatial_variations.png")
    print("  - data_exploration/outputs/data_comprehensive_summary.png")
    print("  - data_exploration/outputs/data_summary_report.txt")
    print("\n")

if __name__ == '__main__':
    main()
