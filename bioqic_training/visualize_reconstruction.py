"""
Visualize stiffness reconstruction from trained model.
Shows ground truth vs predicted stiffness in 2D slices.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from pathlib import Path
import sys

# Add approach folder to path for pielm_solver
sys.path.insert(0, str(Path(__file__).parent.parent / 'approach'))

from data_loader import BIOQICDataLoader
from stiffness_network import FlexibleStiffnessNetwork
from forward_model import ForwardMREModel


def load_trained_model(experiment_name='baseline'):
    """Load trained model and configuration."""
    output_dir = Path(__file__).parent / 'outputs' / experiment_name
    
    # Load config
    with open(output_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Create stiffness network
    mu_net = FlexibleStiffnessNetwork(
        input_dim=3,
        hidden_dim=64,
        n_layers=3,
        output_strategy=config['stiffness_strategy'],
        mu_min=config['mu_min'],
        mu_max=config['mu_max']
    )
    
    # Create forward model
    model = ForwardMREModel(
        mu_network=mu_net,
        n_wave_neurons=config['n_wave_neurons'],
        input_dim=3,
        physics_mode=config['physics_mode']
    )
    
    # Load weights
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def visualize_reconstruction_2d(experiment_name='baseline', z_slice_index=5):
    """
    Visualize stiffness reconstruction on a 2D slice.
    
    Args:
        experiment_name: Name of experiment
        z_slice_index: Which Z-slice to visualize (0-9)
    """
    print(f"\n{'='*80}")
    print(f"STIFFNESS RECONSTRUCTION VISUALIZATION")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Z-slice: {z_slice_index}/10")
    
    # Load trained model
    print("\n📦 Loading trained model...")
    model, config = load_trained_model(experiment_name)
    
    # Load full data
    print("\n📂 Loading full BIOQIC data...")
    loader = BIOQICDataLoader(
        data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
        subsample=None,  # Load all points
        displacement_mode='z_component',
        seed=42
    )
    
    raw_data = loader.load()
    
    # Extract the specified Z-slice
    coords = raw_data['coords']
    mu_true = raw_data['mu_data']
    
    # Get unique Z values
    z_unique = np.unique(coords[:, 2])
    z_value = z_unique[z_slice_index]
    
    print(f"   Z-slice value: {z_value*1000:.1f} mm")
    
    # Extract slice
    slice_mask = np.abs(coords[:, 2] - z_value) < 1e-6
    coords_slice = coords[slice_mask]
    mu_true_slice = mu_true[slice_mask]
    
    # Denormalize ground truth: denorm = normalized * scale + min
    mu_scale = raw_data['scales']['mu_scale']
    mu_min = raw_data['scales']['mu_min']
    mu_true_slice_pa = mu_true_slice * mu_scale + mu_min
    
    print(f"   Points in slice: {len(coords_slice)}")
    print(f"   Stiffness range (true): [{mu_true_slice_pa.min():.0f}, {mu_true_slice_pa.max():.0f}] Pa")
    
    # Convert to torch tensors
    x_slice = torch.FloatTensor(coords_slice)
    
    # Get predictions
    print("\n🔮 Computing predictions...")
    with torch.no_grad():
        mu_pred_slice = model.mu_network(x_slice).numpy()
    
    # Denormalize predictions: denorm = normalized * scale + min
    mu_pred_slice_pa = mu_pred_slice * mu_scale + mu_min
    
    print(f"   Stiffness range (pred): [{mu_pred_slice_pa.min():.0f}, {mu_pred_slice_pa.max():.0f}] Pa")
    
    # Reshape to grid
    grid_shape = (100, 80, 10)
    n_x, n_y = grid_shape[0], grid_shape[1]
    
    # Get X, Y coordinates
    x_coords = coords_slice[:, 0]
    y_coords = coords_slice[:, 1]
    
    # Create meshgrid
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    
    # Reshape to 2D arrays
    mu_true_2d = mu_true_slice_pa.reshape(n_x, n_y)
    mu_pred_2d = mu_pred_slice_pa.reshape(n_x, n_y)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Common settings
    extent = [y_unique[0]*1000, y_unique[-1]*1000, 
              x_unique[0]*1000, x_unique[-1]*1000]
    vmin, vmax = 2000, 11000  # Pa
    
    # Plot 1: Ground Truth
    ax = axes[0, 0]
    im1 = ax.imshow(mu_true_2d, extent=extent, origin='lower', 
                    cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('Ground Truth Stiffness', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('X (mm)')
    plt.colorbar(im1, ax=ax, label='μ (Pa)')
    
    # Plot 2: Predicted
    ax = axes[0, 1]
    im2 = ax.imshow(mu_pred_2d, extent=extent, origin='lower', 
                    cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('Predicted Stiffness', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('X (mm)')
    plt.colorbar(im2, ax=ax, label='μ (Pa)')
    
    # Plot 3: Absolute Error
    ax = axes[0, 2]
    error = np.abs(mu_pred_2d - mu_true_2d)
    im3 = ax.imshow(error, extent=extent, origin='lower', 
                    cmap='hot', aspect='auto')
    ax.set_title('Absolute Error', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('X (mm)')
    plt.colorbar(im3, ax=ax, label='|Error| (Pa)')
    
    # Plot 4: Ground Truth Histogram
    ax = axes[1, 0]
    ax.hist(mu_true_slice_pa.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(3000, color='red', linestyle='--', label='Background (3 kPa)')
    ax.axvline(10000, color='green', linestyle='--', label='Targets (10 kPa)')
    ax.set_xlabel('Stiffness (Pa)')
    ax.set_ylabel('Frequency')
    ax.set_title('Ground Truth Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 5: Predicted Histogram
    ax = axes[1, 1]
    ax.hist(mu_pred_slice_pa.flatten(), bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(3000, color='red', linestyle='--', label='Expected Background')
    ax.axvline(10000, color='green', linestyle='--', label='Expected Targets')
    ax.set_xlabel('Stiffness (Pa)')
    ax.set_ylabel('Frequency')
    ax.set_title('Predicted Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 6: Scatter Plot
    ax = axes[1, 2]
    ax.scatter(mu_true_slice_pa.flatten(), mu_pred_slice_pa.flatten(), 
               alpha=0.3, s=1, c='blue')
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', label='Perfect prediction')
    ax.set_xlabel('Ground Truth (Pa)')
    ax.set_ylabel('Predicted (Pa)')
    ax.set_title('Scatter: True vs Predicted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    
    # Add metrics
    mae = np.mean(np.abs(mu_pred_slice_pa.flatten() - mu_true_slice_pa.flatten()))
    rmse = np.sqrt(np.mean((mu_pred_slice_pa.flatten() - mu_true_slice_pa.flatten())**2))
    
    fig.suptitle(
        f'Stiffness Reconstruction - {experiment_name.upper()} - Z-slice {z_slice_index}\n'
        f'MAE: {mae:.0f} Pa | RMSE: {rmse:.0f} Pa',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_dir = Path(__file__).parent / 'outputs' / experiment_name
    save_path = output_dir / f'reconstruction_slice_{z_slice_index}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: {save_path}")
    
    plt.show()
    
    return mu_true_2d, mu_pred_2d


def visualize_all_slices(experiment_name='baseline'):
    """Visualize reconstruction across all Z-slices."""
    print(f"\n{'='*80}")
    print(f"MULTI-SLICE RECONSTRUCTION VISUALIZATION")
    print(f"{'='*80}")
    
    # Load model
    model, config = load_trained_model(experiment_name)
    
    # Load full data
    loader = BIOQICDataLoader(
        data_dir=Path(__file__).parent.parent / 'data' / 'processed' / 'phase1_box',
        subsample=None,
        displacement_mode='z_component',
        seed=42
    )
    raw_data = loader.load()
    
    coords = raw_data['coords']
    mu_true = raw_data['mu_data']
    mu_scale = raw_data['scales']['mu_scale']
    
    # Get unique Z values
    z_unique = np.unique(coords[:, 2])
    n_slices = len(z_unique)
    
    print(f"Total slices: {n_slices}")
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    vmin, vmax = 2000, 11000
    
    for i, z_val in enumerate(z_unique):
        slice_mask = np.abs(coords[:, 2] - z_val) < 1e-6
        coords_slice = coords[slice_mask]
        mu_true_slice = mu_true[slice_mask]
        
        # Predict
        x_slice = torch.FloatTensor(coords_slice)
        with torch.no_grad():
            mu_pred_slice = model.mu_network(x_slice).numpy()
        mu_pred_slice_pa = mu_pred_slice * mu_scale
        
        # Reshape
        mu_pred_2d = mu_pred_slice_pa.reshape(100, 80)
        
        # Plot
        ax = axes[i]
        x_unique = np.unique(coords_slice[:, 0])
        y_unique = np.unique(coords_slice[:, 1])
        extent = [y_unique[0]*1000, y_unique[-1]*1000,
                  x_unique[0]*1000, x_unique[-1]*1000]
        
        im = ax.imshow(mu_pred_2d, extent=extent, origin='lower',
                       cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Z = {z_val*1000:.0f} mm', fontsize=10)
        ax.set_xlabel('Y (mm)', fontsize=8)
        ax.set_ylabel('X (mm)', fontsize=8)
        
        if i == n_slices - 1:
            plt.colorbar(im, ax=ax, label='μ (Pa)')
    
    fig.suptitle(f'Predicted Stiffness - All Z-Slices - {experiment_name.upper()}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'outputs' / experiment_name
    save_path = output_dir / 'reconstruction_all_slices.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize stiffness reconstruction')
    parser.add_argument('--experiment', type=str, default='baseline',
                        help='Experiment name')
    parser.add_argument('--slice', type=int, default=5,
                        help='Z-slice index (0-9)')
    parser.add_argument('--all-slices', action='store_true',
                        help='Visualize all slices')
    
    args = parser.parse_args()
    
    if args.all_slices:
        visualize_all_slices(args.experiment)
    else:
        visualize_reconstruction_2d(args.experiment, args.slice)
