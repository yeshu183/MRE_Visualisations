"""
Diagnose CNN Stiffness Network μ Predictions
============================================
Visualize the spatial distribution and statistics of μ predicted by the CNN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_bioqic_data
from cnn_stiffness import CNNStiffnessNetwork

def diagnose_cnn_mu(checkpoint_path=None, subsample=2000, seed=0):
    """
    Load a trained CNN model (or initialize fresh) and visualize μ distribution.
    
    Args:
        checkpoint_path: Path to saved model checkpoint (optional)
        subsample: Number of points to sample
        seed: Random seed
    """
    device = torch.device('cpu')
    
    # Load data
    print("Loading BIOQIC data...")
    data_dict = load_bioqic_data(
        raw_dir=Path('data/raw'),
        subsample=subsample,
        seed=seed,
        displacement_mode='z_component'
    )
    
    grid_shape = data_dict['grid_shape']
    coords = data_dict['coords']
    mu_true = data_dict['mu']
    
    print(f"  Grid shape: {grid_shape}")
    print(f"  Points: {len(coords)}")
    print(f"  True μ range: [{mu_true.min():.0f}, {mu_true.max():.0f}] Pa")
    
    # Initialize CNN
    mu_min, mu_max = 3000.0, 10000.0
    cnn = CNNStiffnessNetwork(
        grid_shape=grid_shape,
        mu_min=mu_min,
        mu_max=mu_max,
        latent_dim=8,
        hidden_channels=16
    ).to(device)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(checkpoint['mu_net_state_dict'])
        print("  Checkpoint loaded")
    else:
        print("\nUsing freshly initialized CNN (no checkpoint)")
    
    # Get μ predictions
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        mu_pred = cnn(coords_tensor).cpu().numpy()
    
    mu_true_np = mu_true.cpu().numpy()
    
    # Statistics
    print("\n" + "="*60)
    print("μ Distribution Statistics")
    print("="*60)
    print(f"  Predicted μ:")
    print(f"    Range: [{mu_pred.min():.2f}, {mu_pred.max():.2f}] Pa")
    print(f"    Mean: {mu_pred.mean():.2f} Pa")
    print(f"    Std: {mu_pred.std():.2f} Pa")
    print(f"    Median: {np.median(mu_pred):.2f} Pa")
    print(f"    25th percentile: {np.percentile(mu_pred, 25):.2f} Pa")
    print(f"    75th percentile: {np.percentile(mu_pred, 75):.2f} Pa")
    
    print(f"\n  True μ:")
    print(f"    Range: [{mu_true_np.min():.2f}, {mu_true_np.max():.2f}] Pa")
    print(f"    Mean: {mu_true_np.mean():.2f} Pa")
    print(f"    Std: {mu_true_np.std():.2f} Pa")
    
    # Check for saturation at bounds
    tol = 0.01 * (mu_max - mu_min)  # 1% tolerance
    n_at_min = np.sum(mu_pred < (mu_min + tol))
    n_at_max = np.sum(mu_pred > (mu_max - tol))
    print(f"\n  Saturation at bounds:")
    print(f"    Near min ({mu_min:.0f} Pa): {n_at_min} / {len(mu_pred)} ({100*n_at_min/len(mu_pred):.1f}%)")
    print(f"    Near max ({mu_max:.0f} Pa): {n_at_max} / {len(mu_pred)} ({100*n_at_max/len(mu_pred):.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Histogram comparison
    ax = axes[0, 0]
    ax.hist(mu_true_np, bins=50, alpha=0.6, label='True', color='blue', density=True)
    ax.hist(mu_pred, bins=50, alpha=0.6, label='Predicted', color='orange', density=True)
    ax.axvline(mu_min, color='red', linestyle='--', alpha=0.5, label='Bounds')
    ax.axvline(mu_max, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('μ (Pa)')
    ax.set_ylabel('Density')
    ax.set_title('μ Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Scatter: true vs predicted
    ax = axes[0, 1]
    scatter = ax.scatter(mu_true_np, mu_pred, c=coords[:, 2], s=1, alpha=0.5, cmap='viridis')
    ax.plot([mu_min, mu_max], [mu_min, mu_max], 'r--', label='Perfect')
    ax.set_xlabel('True μ (Pa)')
    ax.set_ylabel('Predicted μ (Pa)')
    ax.set_title('μ: True vs Predicted')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='z (m)')
    
    # Spatial slice at z=0 (middle)
    z_mid_idx = np.argmin(np.abs(coords[:, 2] - np.median(coords[:, 2])))
    z_mid = coords[z_mid_idx, 2]
    z_tol = 0.01  # 1 cm tolerance
    mask_z = np.abs(coords[:, 2] - z_mid) < z_tol
    
    ax = axes[0, 2]
    scatter = ax.scatter(coords[mask_z, 0], coords[mask_z, 1], c=mu_pred[mask_z], 
                        s=10, cmap='viridis', vmin=mu_min, vmax=mu_max)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Predicted μ (z ≈ {z_mid:.3f} m)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='μ (Pa)')
    
    ax = axes[1, 0]
    scatter = ax.scatter(coords[mask_z, 0], coords[mask_z, 1], c=mu_true_np[mask_z], 
                        s=10, cmap='viridis', vmin=mu_min, vmax=mu_max)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'True μ (z ≈ {z_mid:.3f} m)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='μ (Pa)')
    
    # Error distribution
    error = mu_pred - mu_true_np
    ax = axes[1, 1]
    ax.hist(error, bins=50, color='red', alpha=0.6)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Error (Pa)')
    ax.set_ylabel('Count')
    ax.set_title(f'Prediction Error\nMSE: {np.mean(error**2):.2e} Pa²')
    ax.grid(alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1, 2]
    sorted_true = np.sort(mu_true_np)
    sorted_pred = np.sort(mu_pred)
    cdf = np.arange(1, len(sorted_true) + 1) / len(sorted_true)
    ax.plot(sorted_true, cdf, label='True', linewidth=2)
    ax.plot(sorted_pred, cdf, label='Predicted', linewidth=2)
    ax.set_xlabel('μ (Pa)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('bioqic_pielm/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'cnn_mu_diagnosis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    # Inspect CNN internal state
    print("\n" + "="*60)
    print("CNN Internal State")
    print("="*60)
    
    with torch.no_grad():
        # Get latent grid
        latent_grid = cnn.latent_grid
        print(f"  Latent grid shape: {latent_grid.shape}")
        print(f"  Latent grid range: [{latent_grid.min().item():.4f}, {latent_grid.max().item():.4f}]")
        print(f"  Latent grid mean: {latent_grid.mean().item():.4f}")
        print(f"  Latent grid std: {latent_grid.std().item():.4f}")
        
        # Decode latent to μ grid
        mu_grid = cnn.decoder(latent_grid).squeeze()  # [D, H, W]
        mu_grid_scaled = torch.sigmoid(mu_grid) * (mu_max - mu_min) + mu_min
        
        print(f"\n  Decoded μ grid shape: {mu_grid.shape}")
        print(f"  Decoded μ grid (pre-sigmoid) range: [{mu_grid.min().item():.4f}, {mu_grid.max().item():.4f}]")
        print(f"  Decoded μ grid (scaled) range: [{mu_grid_scaled.min().item():.2f}, {mu_grid_scaled.max().item():.2f}] Pa")
        print(f"  Decoded μ grid (scaled) mean: {mu_grid_scaled.mean().item():.2f} Pa")
        print(f"  Decoded μ grid (scaled) std: {mu_grid_scaled.std().item():.2f} Pa")
    
    return {
        'mu_pred': mu_pred,
        'mu_true': mu_true_np,
        'coords': coords,
        'cnn': cnn
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose CNN μ predictions')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--subsample', type=int, default=2000,
                       help='Number of points to sample')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    args = parser.parse_args()
    
    diagnose_cnn_mu(
        checkpoint_path=args.checkpoint,
        subsample=args.subsample,
        seed=args.seed
    )
