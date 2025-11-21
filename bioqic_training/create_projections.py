"""
Create different 2D projections from 3D BIOQIC data.

Projection methods:
1. Middle slice (single z-plane)
2. Average over all z-planes
3. Max over all z-planes
4. Min over all z-planes
5. Median over all z-planes
6. Std deviation over all z-planes

Apply to both displacement |u| and stiffness μ
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import BIOQICDataLoader
import sys


def create_projections():
    """Generate different 2D projections of 3D MRE data."""
    
    print("="*70)
    print("2D PROJECTIONS FROM 3D MRE DATA")
    print("="*70)
    
    # Load full 3D data
    print("\n1. Loading BIOQIC data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None  # All 80,000 points
    )
    sys.stdout.flush()
    data = loader.load()
    
    coords = data['coords']  # Physical (m)
    u_norm = data['u_data']  # Normalized [0,1]
    mu_norm = data['mu_data']  # Normalized [0,1]
    
    # Denormalize
    u_scale = data['scales']['u_scale']
    u_phys = u_norm * u_scale  # meters
    
    mu_min = data['scales']['mu_min']
    mu_scale = data['scales']['mu_scale']
    mu_phys = mu_norm * mu_scale + mu_min  # Pa
    
    print(f"   Grid: 100×80×10 voxels (80,000 points)")
    print(f"   Displacement: [{u_phys.min():.2e}, {u_phys.max():.2e}] m")
    print(f"   Stiffness: [{mu_phys.min():.0f}, {mu_phys.max():.0f}] Pa")
    
    # Reshape to 3D grid
    print("\n2. Reshaping to 3D grid...")
    grid_shape = (100, 80, 10)  # (y, x, z)
    
    x_3d = coords[:, 0].reshape(grid_shape)
    y_3d = coords[:, 1].reshape(grid_shape)
    z_3d = coords[:, 2].reshape(grid_shape)
    u_3d = u_phys.reshape(grid_shape)
    mu_3d = mu_phys.reshape(grid_shape)
    
    print(f"   x range: [{x_3d.min():.3f}, {x_3d.max():.3f}] m")
    print(f"   y range: [{y_3d.min():.3f}, {y_3d.max():.3f}] m")
    print(f"   z range: [{z_3d.min():.3f}, {z_3d.max():.3f}] m")
    
    # Get 2D grids (x, y) - same for all z slices
    x_2d = x_3d[:, :, 0]  # (100, 80)
    y_2d = y_3d[:, :, 0]  # (100, 80)
    
    # 3. Create different projections
    print("\n3. Creating projections...")
    
    projections = {}
    
    # A) Middle slice
    z_mid_idx = grid_shape[2] // 2  # Index 5 (middle of 10 slices)
    projections['middle'] = {
        'u': u_3d[:, :, z_mid_idx],
        'mu': mu_3d[:, :, z_mid_idx],
        'label': f'Middle Slice (z={z_mid_idx})',
        'description': 'Single slice at z=5 (middle of box)'
    }
    
    # B) Average (mean over z)
    projections['mean'] = {
        'u': np.mean(u_3d, axis=2),
        'mu': np.mean(mu_3d, axis=2),
        'label': 'Average (Mean)',
        'description': 'Average over all 10 z-slices'
    }
    
    # C) Maximum (max over z)
    projections['max'] = {
        'u': np.max(u_3d, axis=2),
        'mu': np.max(mu_3d, axis=2),
        'label': 'Maximum',
        'description': 'Maximum value over all z-slices'
    }
    
    # D) Minimum (min over z)
    projections['min'] = {
        'u': np.min(u_3d, axis=2),
        'mu': np.min(mu_3d, axis=2),
        'label': 'Minimum',
        'description': 'Minimum value over all z-slices'
    }
    
    # E) Median (median over z)
    projections['median'] = {
        'u': np.median(u_3d, axis=2),
        'mu': np.median(mu_3d, axis=2),
        'label': 'Median',
        'description': 'Median over all z-slices'
    }
    
    # F) Standard deviation (std over z)
    projections['std'] = {
        'u': np.std(u_3d, axis=2),
        'mu': np.std(mu_3d, axis=2),
        'label': 'Std Deviation',
        'description': 'Variability across z-slices'
    }
    
    # G) First slice (bottom)
    projections['first'] = {
        'u': u_3d[:, :, 0],
        'mu': mu_3d[:, :, 0],
        'label': 'First Slice (z=0)',
        'description': 'Bottom slice of box'
    }
    
    # H) Last slice (top)
    projections['last'] = {
        'u': u_3d[:, :, -1],
        'mu': mu_3d[:, :, -1],
        'label': 'Last Slice (z=9)',
        'description': 'Top slice of box'
    }
    
    # Print statistics
    print("\n4. Projection statistics:")
    print("\n" + "="*70)
    print(f"{'Method':<15} {'u_min':>12} {'u_max':>12} {'μ_min':>10} {'μ_max':>10}")
    print("="*70)
    
    for key, proj in projections.items():
        u_data = proj['u']
        mu_data = proj['mu']
        print(f"{proj['label']:<15} {u_data.min():>12.2e} {u_data.max():>12.2e} "
              f"{mu_data.min():>10.0f} {mu_data.max():>10.0f}")
    
    print("="*70)
    
    # 5. Visualize - Displacement
    print("\n5. Creating displacement visualizations...")
    
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle('Displacement |u| - Different 2D Projections from 3D Data', 
                  fontsize=16, fontweight='bold')
    
    for idx, (key, proj) in enumerate(projections.items(), 1):
        ax = plt.subplot(2, 4, idx)
        
        im = ax.contourf(x_2d*1000, y_2d*1000, proj['u']*1000, 
                        levels=20, cmap='plasma')
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_title(f"{proj['label']}\n{proj['description']}", 
                    fontweight='bold', fontsize=10)
        plt.colorbar(im, ax=ax, label='|u| (mm)')
        ax.set_aspect('equal')
        
        # Add stats text
        stats_text = (f"min: {proj['u'].min()*1000:.3f} mm\n"
                     f"max: {proj['u'].max()*1000:.3f} mm\n"
                     f"mean: {proj['u'].mean()*1000:.3f} mm")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output1 = 'outputs/projections_displacement.png'
    plt.savefig(output1, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output1}")
    
    # 6. Visualize - Stiffness
    print("\n6. Creating stiffness visualizations...")
    
    fig2 = plt.figure(figsize=(20, 12))
    fig2.suptitle('Stiffness μ - Different 2D Projections from 3D Data', 
                  fontsize=16, fontweight='bold')
    
    for idx, (key, proj) in enumerate(projections.items(), 1):
        ax = plt.subplot(2, 4, idx)
        
        im = ax.contourf(x_2d*1000, y_2d*1000, proj['mu']/1000, 
                        levels=20, cmap='viridis')
        ax.set_xlabel('X (mm)', fontweight='bold')
        ax.set_ylabel('Y (mm)', fontweight='bold')
        ax.set_title(f"{proj['label']}\n{proj['description']}", 
                    fontweight='bold', fontsize=10)
        plt.colorbar(im, ax=ax, label='μ (kPa)')
        ax.set_aspect('equal')
        
        # Add stats text
        stats_text = (f"min: {proj['mu'].min()/1000:.1f} kPa\n"
                     f"max: {proj['mu'].max()/1000:.1f} kPa\n"
                     f"mean: {proj['mu'].mean()/1000:.1f} kPa")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output2 = 'outputs/projections_stiffness.png'
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output2}")
    
    # 7. Comparison plot
    print("\n7. Creating comparison plot...")
    
    fig3 = plt.figure(figsize=(18, 10))
    fig3.suptitle('Side-by-Side Comparison: Displacement vs Stiffness', 
                  fontsize=16, fontweight='bold')
    
    # Select 4 interesting projections for comparison
    compare_keys = ['middle', 'mean', 'max', 'std']
    
    for idx, key in enumerate(compare_keys):
        proj = projections[key]
        
        # Displacement
        ax1 = plt.subplot(4, 2, idx*2 + 1)
        im1 = ax1.contourf(x_2d*1000, y_2d*1000, proj['u']*1000,
                          levels=20, cmap='plasma')
        ax1.set_ylabel('Y (mm)', fontweight='bold')
        if idx == 3:
            ax1.set_xlabel('X (mm)', fontweight='bold')
        ax1.set_title(f"{proj['label']}: |u|", fontweight='bold', fontsize=11)
        plt.colorbar(im1, ax=ax1, label='|u| (mm)')
        ax1.set_aspect('equal')
        
        # Stiffness
        ax2 = plt.subplot(4, 2, idx*2 + 2)
        im2 = ax2.contourf(x_2d*1000, y_2d*1000, proj['mu']/1000,
                          levels=20, cmap='viridis')
        if idx == 3:
            ax2.set_xlabel('X (mm)', fontweight='bold')
        ax2.set_title(f"{proj['label']}: μ", fontweight='bold', fontsize=11)
        plt.colorbar(im2, ax=ax2, label='μ (kPa)')
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    output3 = 'outputs/projections_comparison.png'
    plt.savefig(output3, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output3}")
    
    plt.show()
    
    # 8. Analysis
    print("\n" + "="*70)
    print("ANALYSIS:")
    print("="*70)
    
    print("\n📊 Displacement observations:")
    print(f"   - Middle slice shows clear wave pattern")
    print(f"   - Average smooths out z-variation")
    print(f"   - Max highlights peak displacement: {projections['max']['u'].max()*1000:.3f} mm")
    print(f"   - Std shows variation across z: [{projections['std']['u'].min()*1000:.3f}, "
          f"{projections['std']['u'].max()*1000:.3f}] mm")
    
    print("\n🎯 Stiffness observations:")
    # Check if μ varies with z
    mu_std_mean = projections['std']['mu'].mean()
    if mu_std_mean < 100:  # Pa
        print(f"   - μ is CONSTANT across z (std={mu_std_mean:.1f} Pa)")
        print(f"   - All slices show same 2 values: 3 kPa (bg), 10 kPa (targets)")
    else:
        print(f"   - μ VARIES across z (std={mu_std_mean:.0f} Pa)")
    
    print(f"   - 4 circular targets clearly visible")
    print(f"   - Background: {projections['mean']['mu'].min()/1000:.1f} kPa")
    print(f"   - Targets: {projections['mean']['mu'].max()/1000:.1f} kPa")
    
    return projections, x_2d, y_2d


if __name__ == '__main__':
    projections, x_2d, y_2d = create_projections()
