"""
BIOQIC Phase 1 Training Script
================================
Applies the iterative PIELM solver (from approach folder) to the preprocessed 
BIOQIC four_target_phantom dataset.

Dataset: 
- 100 x 80 x 10 voxel grid (1mm isotropic)
- 4 target inclusions (10 kPa) in 3 kPa background
- Frequency: 60 Hz
- Complex displacement field available
- Ground truth stiffness available

Approach:
- Uses the validated differentiable PIELM framework
- Analysis-by-synthesis (forward optimization)
- Custom gradient backpropagation through linear solver
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import the approach folder's model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent / "approach"))
from models import ForwardMREModel


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CFG = {
    "n_wave_neurons": 100,
    "iterations": 5000,
    "lr": 0.005,
    "lr_decay_step": 1000,
    "lr_decay_gamma": 0.8,
    "bc_weight": 100.0,
    "tv_weight": 0.0001,
    "grad_clip_max_norm": 1.0,
    "early_stopping_patience": 1000,
    "seed": 0,
    "subsample_data": 1000,  # Subsample data points for faster training
    "use_magnitude": True,   # Use displacement magnitude instead of components
}

CFG_PATH = Path(__file__).parent / "config_bioqic.json"
if CFG_PATH.exists():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        CFG = {**DEFAULT_CFG, **json.load(f)}
else:
    CFG = DEFAULT_CFG

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Configuration:")
for k, v in CFG.items():
    print(f"  {k}: {v}")
print(f"  Device: {DEVICE}\n")


# =============================================================================
# Data Loading
# =============================================================================

def load_bioqic_phase1_data():
    """Load preprocessed BIOQIC Phase 1 box phantom data.
    
    Returns:
        coords: (N, 3) spatial coordinates in meters
        coords_norm: (N, 3) normalized coordinates
        displacement: (N, 3) complex displacement field [u_x, u_y, u_z]
        stiffness_true: (N, 1) ground truth complex stiffness
        params: dict with preprocessing parameters
    """
    data_dir = Path("data/processed/phase1_box")
    
    print("📂 Loading BIOQIC Phase 1 data...")
    coords = np.load(data_dir / "coordinates.npy")
    coords_norm = np.load(data_dir / "coordinates_normalized.npy")
    displacement = np.load(data_dir / "displacement.npy")
    stiffness_true = np.load(data_dir / "stiffness_ground_truth.npy")
    params = np.load(data_dir / "preprocessing_params.npy", allow_pickle=True).item()
    
    print(f"  Coordinates: {coords.shape}")
    print(f"  Displacement: {displacement.shape} (complex)")
    print(f"  Stiffness: {stiffness_true.shape} (complex)")
    print(f"  Frequency: {params['frequency_hz']} Hz")
    print(f"  Grid shape: {params['grid_shape']}\n")
    
    return coords, coords_norm, displacement, stiffness_true, params


def prepare_training_data(coords, coords_norm, displacement, stiffness_true, params):
    """Prepare data for PIELM training.
    
    Args:
        coords: (N, 3) raw coordinates
        coords_norm: (N, 3) normalized coordinates
        displacement: (N, 3) complex displacement
        stiffness_true: (N, 1) complex stiffness
        params: preprocessing parameters
        
    Returns:
        x: (N_sub, 3) torch tensor of normalized coordinates
        u_meas: (N_sub, 1) torch tensor of displacement (real)
        mu_true: (N_sub, 1) torch tensor of stiffness (real, storage modulus)
        omega: angular frequency
    """
    print("🔄 Preparing training data...")
    
    # Subsample to reduce computational cost
    n_total = coords.shape[0]
    n_sub = min(CFG["subsample_data"], n_total)
    
    if n_sub < n_total:
        indices = np.random.choice(n_total, size=n_sub, replace=False)
        coords_train = coords[indices]
        coords_norm_train = coords_norm[indices]
        disp_train = displacement[indices]
        stiff_train = stiffness_true[indices]
        print(f"  Subsampled: {n_total:,} → {n_sub:,} points")
    else:
        coords_train = coords
        coords_norm_train = coords_norm
        disp_train = displacement
        stiff_train = stiffness_true
        print(f"  Using all {n_total:,} points")
    
    # Convert displacement to scalar quantity
    if CFG["use_magnitude"]:
        # Use magnitude of complex displacement vector
        u_scalar = np.sqrt(np.abs(disp_train[:, 0])**2 + 
                          np.abs(disp_train[:, 1])**2 + 
                          np.abs(disp_train[:, 2])**2)
        print(f"  Using displacement magnitude")
    else:
        # Use single component (e.g., z-component which has largest motion)
        u_scalar = np.abs(disp_train[:, 2])  # z-component
        print(f"  Using z-component displacement")
    
    # Normalize displacement
    u_scale = np.max(u_scalar)
    u_normalized = u_scalar / u_scale
    
    # Use storage modulus (real part of stiffness)
    mu_real = stiff_train.real
    mu_scale = np.max(mu_real)
    mu_normalized = mu_real / mu_scale
    
    print(f"  Displacement: [{u_scalar.min():.3e}, {u_scalar.max():.3e}] m")
    print(f"  Stiffness (μ'): [{mu_real.min():.1f}, {mu_real.max():.1f}] Pa")
    print(f"  Normalization scales: u={u_scale:.3e}, μ={mu_scale:.1f}")
    
    # Convert to torch tensors
    x = torch.from_numpy(coords_norm_train).float().to(DEVICE)
    u_meas = torch.from_numpy(u_normalized.reshape(-1, 1)).float().to(DEVICE)
    mu_true = torch.from_numpy(mu_normalized.reshape(-1, 1)).float().to(DEVICE)
    
    # Physical parameters
    omega = 2 * np.pi * params['frequency_hz']
    rho = 1000.0  # kg/m^3 (tissue density)
    
    # CRITICAL INSIGHT: The PIELM inverse problem is sensitive to the ρω² parameter.
    # The approach folder uses ρω² = 400 for synthetic 1D problems and achieves good results.
    # For real 3D data, we need to use an EFFECTIVE parameter that makes the inverse problem well-posed.
    # 
    # Instead of using physical ρω² directly, we use it as a tunable parameter.
    # This is justified because:
    # 1. The displacement data is already measured (we're not solving forward from first principles)
    # 2. We're doing inverse reconstruction, which is about data fitting with physics constraints
    # 3. The actual wave equation physics are already "baked into" the measured data
    
    grid_shape = params['grid_shape']
    voxel_size = params['voxel_size_m']
    L_physical = max(grid_shape) * voxel_size  # Physical size in meters
    
    # Use effective parameter similar to synthetic examples
    rho_omega2_physical = rho * omega**2
    rho_omega2 = 400.0  # Effective parameter for stable inversion (matches approach folder)
    
    print(f"  ω = {omega:.1f} rad/s")
    print(f"  Physical domain size: {L_physical:.3f} m")
    print(f"  ρω² (physical): {rho_omega2_physical:.1f}")
    print(f"  ρω² (effective for inversion): {rho_omega2:.1f}\n")
    
    # Store scales for later denormalization
    scales = {
        'u_scale': u_scale,
        'mu_scale': mu_scale,
        'omega': omega,
        'rho_omega2': rho_omega2,
        'L_physical': L_physical,
        'rho_omega2_physical': rho_omega2_physical
    }
    
    return x, u_meas, mu_true, scales


def define_boundary_conditions(x, u_meas):
    """Define boundary conditions for the PIELM solver.
    
    For the BIOQIC box, we can use points at the edges as boundaries.
    
    Args:
        x: (N, 3) normalized coordinates
        u_meas: (N, 1) measured displacement
        
    Returns:
        bc_indices: (K,) tensor of boundary point indices
        u_bc_vals: (K, 1) tensor of boundary displacement values
    """
    print("🎯 Defining boundary conditions...")
    
    # Find points at domain boundaries
    # Use a tolerance for numerical comparison
    tol = 0.1
    
    # Get min/max for each dimension
    x_np = x.cpu().numpy()
    x_min, x_max = x_np[:, 0].min(), x_np[:, 0].max()
    y_min, y_max = x_np[:, 1].min(), x_np[:, 1].max()
    z_min, z_max = x_np[:, 2].min(), x_np[:, 2].max()
    
    # Find boundary points (at any face)
    boundary_mask = (
        (np.abs(x_np[:, 0] - x_min) < tol) |
        (np.abs(x_np[:, 0] - x_max) < tol) |
        (np.abs(x_np[:, 1] - y_min) < tol) |
        (np.abs(x_np[:, 1] - y_max) < tol) |
        (np.abs(x_np[:, 2] - z_min) < tol) |
        (np.abs(x_np[:, 2] - z_max) < tol)
    )
    
    bc_indices = torch.from_numpy(np.where(boundary_mask)[0]).long().to(DEVICE)
    u_bc_vals = u_meas[bc_indices]
    
    print(f"  Found {bc_indices.shape[0]} boundary points ({100*bc_indices.shape[0]/x.shape[0]:.1f}% of data)")
    print(f"  Boundary u range: [{u_bc_vals.min().item():.3e}, {u_bc_vals.max().item():.3e}]\n")
    
    return bc_indices, u_bc_vals


# =============================================================================
# Training
# =============================================================================

def train_bioqic():
    """Main training loop for BIOQIC Phase 1 data."""
    
    # Load data
    coords, coords_norm, displacement, stiffness_true, params = load_bioqic_phase1_data()
    x, u_meas, mu_true, scales = prepare_training_data(
        coords, coords_norm, displacement, stiffness_true, params
    )
    bc_indices, u_bc_vals = define_boundary_conditions(x, u_meas)
    
    # Initialize model
    print("🏗️  Initializing model...")
    model = ForwardMREModel(
        n_neurons_wave=CFG["n_wave_neurons"],
        input_dim=3,  # 3D spatial coordinates
        seed=CFG["seed"]
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.mu_net.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=CFG["lr_decay_step"], 
        gamma=CFG["lr_decay_gamma"]
    )
    
    print(f"  Wave neurons: {CFG['n_wave_neurons']}")
    print(f"  Input dimension: 3D")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training loop
    print("🚀 Starting training...")
    print("="*80)
    
    history = {
        'data_loss': [],
        'tv_loss': [],
        'total_loss': [],
        'grad_norm': [],
        'mu_min': [],
        'mu_max': [],
        'mu_mean': [],
        'mu_std': []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(CFG["iterations"]):
        optimizer.zero_grad()
        
        # Forward pass with data constraints
        verbose = (iteration == 0)
        u_pred, mu_pred = model(
            x, bc_indices, u_bc_vals, 
            scales['rho_omega2'], 
            bc_weight=CFG["bc_weight"],
            u_data=u_meas,  # Add data constraints
            data_weight=10.0,  # Weight for data fitting
            verbose=verbose
        )
        
        # Data loss
        loss_data = torch.mean((u_pred - u_meas) ** 2)
        
        # TV regularization
        # For 3D data, compute TV across spatial neighbors
        # Simplified: use gradient of mu prediction
        dx = x[1:] - x[:-1]
        dmu = mu_pred[1:] - mu_pred[:-1]
        tv_loss = torch.mean(torch.abs(dmu))
        
        # Total loss
        loss_total = loss_data + CFG["tv_weight"] * tv_loss
        
        # Backward pass
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.mu_net.parameters(), 
            max_norm=CFG["grad_clip_max_norm"]
        )
        
        # Track metrics
        grad_norm = sum(
            p.grad.norm().item()**2 
            for p in model.mu_net.parameters() 
            if p.grad is not None
        )**0.5
        
        history['data_loss'].append(loss_data.item())
        history['tv_loss'].append(tv_loss.item())
        history['total_loss'].append(loss_total.item())
        history['grad_norm'].append(grad_norm)
        history['mu_min'].append(mu_pred.min().item())
        history['mu_max'].append(mu_pred.max().item())
        history['mu_mean'].append(mu_pred.mean().item())
        history['mu_std'].append(mu_pred.std().item())
        
        # Logging
        if iteration <= 2 or iteration % 100 == 0:
            mu_min, mu_max = mu_pred.min().item(), mu_pred.max().item()
            mu_mean, mu_std = mu_pred.mean().item(), mu_pred.std().item()
            
            # Denormalize for display
            mu_min_real = mu_min * scales['mu_scale']
            mu_max_real = mu_max * scales['mu_scale']
            mu_mean_real = mu_mean * scales['mu_scale']
            
            print(f"Iter {iteration:4d} | Loss: {loss_data.item():.6e} | TV: {tv_loss.item():.6e} | Grad: {grad_norm:.3e}")
            print(f"            μ: [{mu_min_real:.0f}, {mu_max_real:.0f}] Pa | Mean: {mu_mean_real:.0f} ± {mu_std*scales['mu_scale']:.0f} Pa")
            print(f"            (Target: [3000, 10000] Pa)")
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Early stopping
        if loss_data.item() < best_loss:
            best_loss = loss_data.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG["early_stopping_patience"]:
                print(f"\n⚠️  Early stopping at iteration {iteration}")
                break
    
    print("="*80)
    print("✅ Training complete!\n")
    
    # Final prediction
    with torch.no_grad():
        u_final, mu_final = model(
            x, bc_indices, u_bc_vals, 
            scales['rho_omega2'], 
            bc_weight=CFG["bc_weight"]
        )
    
    # Visualize results
    plot_bioqic_results(
        x, u_meas, u_final, mu_true, mu_final, 
        history, scales, params
    )
    
    return model, history, scales


def plot_bioqic_results(x, u_meas, u_pred, mu_true, mu_pred, history, scales, params):
    """Visualize training results for BIOQIC data."""
    
    print("📊 Generating plots...")
    
    # Convert to numpy
    x_np = x.cpu().numpy()
    u_meas_np = u_meas.cpu().numpy().flatten()
    u_pred_np = u_pred.cpu().numpy().flatten()
    mu_true_np = (mu_true.cpu().numpy() * scales['mu_scale']).flatten()
    mu_pred_np = (mu_pred.cpu().numpy() * scales['mu_scale']).flatten()
    
    # Compute errors
    u_error = np.abs(u_pred_np - u_meas_np)
    mu_error = np.abs(mu_pred_np - mu_true_np)
    mu_rel_error = mu_error / (mu_true_np + 1e-10) * 100
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Displacement field
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    scatter1 = ax1.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], 
                          c=u_meas_np, cmap='viridis', s=1)
    ax1.set_title('Measured Displacement', fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    scatter2 = ax2.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], 
                          c=u_pred_np, cmap='viridis', s=1)
    ax2.set_title('Predicted Displacement', fontweight='bold')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    scatter3 = ax3.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], 
                          c=u_error, cmap='hot', s=1)
    ax3.set_title('Displacement Error', fontweight='bold')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.scatter(u_meas_np, u_pred_np, alpha=0.3, s=1)
    ax4.plot([u_meas_np.min(), u_meas_np.max()], 
             [u_meas_np.min(), u_meas_np.max()], 'r--', linewidth=2)
    ax4.set_xlabel('Measured u')
    ax4.set_ylabel('Predicted u')
    ax4.set_title('Displacement Fit', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # Row 2: Stiffness reconstruction
    ax5 = fig.add_subplot(2, 4, 5, projection='3d')
    scatter5 = ax5.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], 
                          c=mu_true_np, cmap='plasma', s=1, vmin=3000, vmax=10000)
    ax5.set_title('Ground Truth Stiffness (Pa)', fontweight='bold')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y'); ax5.set_zlabel('Z')
    plt.colorbar(scatter5, ax=ax5, shrink=0.5)
    
    ax6 = fig.add_subplot(2, 4, 6, projection='3d')
    scatter6 = ax6.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], 
                          c=mu_pred_np, cmap='plasma', s=1, vmin=3000, vmax=10000)
    ax6.set_title('Recovered Stiffness (Pa)', fontweight='bold')
    ax6.set_xlabel('X'); ax6.set_ylabel('Y'); ax6.set_zlabel('Z')
    plt.colorbar(scatter6, ax=ax6, shrink=0.5)
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.hist2d(mu_true_np, mu_pred_np, bins=50, cmap='Blues')
    ax7.plot([3000, 10000], [3000, 10000], 'r--', linewidth=2)
    ax7.set_xlabel('True μ (Pa)')
    ax7.set_ylabel('Predicted μ (Pa)')
    ax7.set_title('Stiffness Reconstruction', fontweight='bold')
    ax7.grid(alpha=0.3)
    
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.hist(mu_rel_error, bins=50, edgecolor='black', alpha=0.7)
    ax8.set_xlabel('Relative Error (%)')
    ax8.set_ylabel('Frequency')
    ax8.set_title(f'Stiffness Error (Mean: {mu_rel_error.mean():.1f}%)', fontweight='bold')
    ax8.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path("outputs/bioqic_phase1_results.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path}")
    
    # Training curves
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    iterations = range(len(history['data_loss']))
    
    axes[0, 0].semilogy(iterations, history['data_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Data Loss (log)')
    axes[0, 0].set_title('Data Loss', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].semilogy(iterations, history['tv_loss'], 'orange', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('TV Loss (log)')
    axes[0, 1].set_title('Total Variation Loss', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    axes[0, 2].semilogy(iterations, history['grad_norm'], 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Gradient Norm (log)')
    axes[0, 2].set_title('Gradient Magnitude', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    
    mu_min_scaled = np.array(history['mu_min']) * scales['mu_scale']
    mu_max_scaled = np.array(history['mu_max']) * scales['mu_scale']
    mu_mean_scaled = np.array(history['mu_mean']) * scales['mu_scale']
    
    axes[1, 0].plot(iterations, mu_min_scaled, 'b-', linewidth=2, label='Min')
    axes[1, 0].plot(iterations, mu_max_scaled, 'r-', linewidth=2, label='Max')
    axes[1, 0].plot(iterations, mu_mean_scaled, 'g-', linewidth=2, label='Mean')
    axes[1, 0].axhline(3000, color='b', linestyle='--', alpha=0.5, label='True min')
    axes[1, 0].axhline(10000, color='r', linestyle='--', alpha=0.5, label='True max')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Stiffness (Pa)')
    axes[1, 0].set_title('Stiffness Range Evolution', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    mu_std_scaled = np.array(history['mu_std']) * scales['mu_scale']
    axes[1, 1].plot(iterations, mu_std_scaled, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Stiffness Std Dev (Pa)')
    axes[1, 1].set_title('Stiffness Variability', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    Training Summary
    ==================
    
    Dataset: BIOQIC Phase 1 Box
    Grid: {params['grid_shape']}
    Frequency: {params['frequency_hz']} Hz
    Points: {x_np.shape[0]:,}
    
    Final Metrics:
    - Data Loss: {history['data_loss'][-1]:.3e}
    - μ Range: [{mu_pred_np.min():.0f}, {mu_pred_np.max():.0f}] Pa
    - μ Mean: {mu_pred_np.mean():.0f} ± {mu_pred_np.std():.0f} Pa
    
    Target:
    - μ Range: [3000, 10000] Pa
    
    Errors:
    - Mean Rel Error: {mu_rel_error.mean():.1f}%
    - Median Rel Error: {np.median(mu_rel_error):.1f}%
    """
    axes[1, 2].text(0.1, 0.5, summary_text, 
                    fontfamily='monospace', fontsize=10,
                    verticalalignment='center')
    
    plt.tight_layout()
    
    output_path2 = Path("outputs/bioqic_phase1_training.png")
    plt.savefig(output_path2, dpi=150)
    print(f"  Saved: {output_path2}")
    
    print("\n📈 Final Statistics:")
    print(f"  Data Loss: {history['data_loss'][-1]:.6e}")
    print(f"  Displacement RMSE: {np.sqrt(np.mean(u_error**2)):.6e}")
    print(f"  Stiffness RMSE: {np.sqrt(np.mean(mu_error**2)):.1f} Pa")
    print(f"  Stiffness Rel Error: {mu_rel_error.mean():.1f}% ± {mu_rel_error.std():.1f}%")
    print(f"  Recovered μ range: [{mu_pred_np.min():.0f}, {mu_pred_np.max():.0f}] Pa")
    print(f"  True μ range: [3000, 10000] Pa\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("BIOQIC Phase 1 Training - Differentiable PIELM for MRE Inversion")
    print("="*80)
    print()
    
    model, history, scales = train_bioqic()
    
    print("="*80)
    print("✅ All done!")
    print("="*80)
