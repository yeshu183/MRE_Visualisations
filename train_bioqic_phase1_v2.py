"""
BIOQIC Phase 1 Training - Modified version with custom stiffness network.

This version addresses the issue that the approach folder's StiffnessGenerator
has hardcoded bounds [0.5, 5.0] which don't match the bioqic data range [0.3, 1.0].
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add approach folder to path
sys.path.insert(0, str(Path(__file__).parent / 'approach'))
from pielm_solver import pielm_solve

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CFG = {
    "n_wave_neurons": 100,
    "iterations": 5000,
    "lr": 0.01,
    "lr_decay_step": 1000,
    "lr_decay_gamma": 0.8,
    "bc_weight": 50.0,
    "data_weight": 50.0,  # Strong data constraints
    "tv_weight": 0.001,   # Stronger TV regularization
    "grad_clip_max_norm": 1.0,
    "early_stopping_patience": 1000,
    "seed": 0,
    "subsample_data": 1000,
    "use_magnitude": True,
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

# ============================================================================
# Custom Stiffness Network with Flexible Bounds
# ============================================================================

class FlexibleStiffnessGenerator(nn.Module):
    """Stiffness network with configurable output bounds."""
    
    def __init__(self, input_dim=3, hidden_dim=64, n_fourier=10, 
                 min_val=0.1, max_val=5.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
        # Random Fourier features
        self.register_buffer('B_fourier', torch.randn(input_dim, n_fourier) * 5.0)
        
        feature_dim = input_dim + 2 * n_fourier
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Center output
        with torch.no_grad():
            self.net[-1].bias.fill_(0.0)
    
    def forward(self, x):
        """Forward pass with flexible bounds."""
        # Fourier features
        z = x @ self.B_fourier
        fourier_features = torch.cat([torch.sin(2 * 3.14159 * z), 
                                     torch.cos(2 * 3.14159 * z)], dim=1)
        features = torch.cat([x, fourier_features], dim=1)
        
        # Network output
        raw = self.net(features)
        
        # Sigmoid to [0, 1], then scale to [min_val, max_val]
        normalized = torch.sigmoid(raw)
        mu = self.min_val + (self.max_val - self.min_val) * normalized
        
        return mu


class CustomForwardMREModel(nn.Module):
    """Forward model with custom stiffness network."""
    
    def __init__(self, n_neurons_wave, input_dim=3, omega_basis=15.0, 
                 seed=None, mu_min=0.1, mu_max=5.0):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        self.mu_net = FlexibleStiffnessGenerator(
            input_dim=input_dim, 
            min_val=mu_min, 
            max_val=mu_max
        )
        
        # Wave basis
        self.register_buffer(
            'omega_basis',
            torch.randn(input_dim, n_neurons_wave) * omega_basis
        )
        self.n_neurons_wave = n_neurons_wave
    
    def get_basis_and_laplacian(self, x):
        """Compute basis functions and their Laplacians."""
        # phi(x) = sin(omega * x)
        z = x @ self.omega_basis  # (N, n_neurons)
        phi = torch.sin(z)
        
        # Laplacian: ∇²phi = -||omega||² * phi
        omega_sq = torch.sum(self.omega_basis ** 2, dim=0)  # (n_neurons,)
        phi_lap = -omega_sq.unsqueeze(0) * phi  # (N, n_neurons)
        
        return phi, phi_lap
    
    def build_system(self, mu, phi, phi_lap, rho_omega2,
                     bc_indices, u_bc_vals, bc_weight,
                     u_data=None, data_weight=0.0):
        """Build PIELM system H and b."""
        N, M = phi.shape
        
        # PDE residual: A = -mu * Lap(phi) - rho_omega2 * phi
        A = -mu * phi_lap - rho_omega2 * phi  # (N, M)
        
        # System matrix H = A^T A
        H = A.t() @ A  # (M, M)
        b = torch.zeros(M, 1, device=phi.device)
        
        # Boundary conditions
        if bc_indices is not None and len(bc_indices) > 0:
            phi_bc = phi[bc_indices]  # (K, M)
            H += bc_weight * (phi_bc.t() @ phi_bc)
            b += bc_weight * (phi_bc.t() @ u_bc_vals)
        
        # Data constraints
        if u_data is not None and data_weight > 0:
            H += data_weight * (phi.t() @ phi)
            b += data_weight * (phi.t() @ u_data)
        
        return H, b
    
    def forward(self, x_col, bc_indices, u_bc_vals, rho_omega2,
                bc_weight=1.0, u_data=None, data_weight=0.0, verbose=False):
        """Forward pass."""
        mu_pred = self.mu_net(x_col)
        phi, phi_lap = self.get_basis_and_laplacian(x_col)
        H, b = self.build_system(
            mu_pred, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, bc_weight,
            u_data, data_weight
        )
        C_u = pielm_solve(H, b, verbose=verbose)
        u_pred = phi @ C_u
        return u_pred, mu_pred


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_bioqic_data():
    """Load processed BIOQIC Phase 1 data."""
    data_dir = Path("data/processed/phase1_box")
    
    coords = np.load(data_dir / "coordinates.npy")
    coords_norm = np.load(data_dir / "coordinates_normalized.npy")
    disp = np.load(data_dir / "displacement.npy")
    stiff = np.load(data_dir / "stiffness_ground_truth.npy")
    params = np.load(data_dir / "preprocessing_params.npy", allow_pickle=True).item()
    
    return coords, coords_norm, disp, stiff, params


def prepare_training_data(coords, coords_norm, disp, stiff, params):
    """Prepare training data with subsampling and normalization."""
    print("🔄 Preparing training data...")
    
    # Subsample
    n_total = len(coords)
    n_train = min(CFG["subsample_data"], n_total)
    indices = np.random.choice(n_total, n_train, replace=False)
    
    coords_train = coords[indices]
    coords_norm_train = coords_norm[indices]
    disp_train = disp[indices]
    stiff_train = stiff[indices]
    
    print(f"  Subsampled: {n_total:,} → {n_train:,} points")
    
    # Process displacement
    if CFG["use_magnitude"]:
        print("  Using displacement magnitude")
        # Compute magnitude across all 3 components
        u_scalar = np.linalg.norm(np.abs(disp_train), axis=1, keepdims=True)  # (n, 1)
    else:
        # Use magnitude of first component only
        u_scalar = np.abs(disp_train[:, 0:1])  # (n, 1)
    
    # Normalize displacement
    u_scale = np.max(u_scalar)
    u_normalized = u_scalar / u_scale
    
    # Use storage modulus (real part)
    mu_real = stiff_train.real
    mu_scale = np.max(mu_real)
    mu_normalized = mu_real / mu_scale
    
    print(f"  Displacement: [{u_scalar.min():.3e}, {u_scalar.max():.3e}] m")
    print(f"  Stiffness (μ'): [{mu_real.min():.1f}, {mu_real.max():.1f}] Pa")
    print(f"  Normalization scales: u={u_scale:.3e}, μ={mu_scale:.1f}")
    print(f"  Normalized μ range: [{mu_normalized.min():.3f}, {mu_normalized.max():.3f}]")
    
    # Convert to tensors
    x = torch.from_numpy(coords_norm_train).float().to(DEVICE)
    u_meas = torch.from_numpy(u_normalized.reshape(-1, 1)).float().to(DEVICE)
    mu_true = torch.from_numpy(mu_normalized.reshape(-1, 1)).float().to(DEVICE)
    
    # Physical parameters (use effective value)
    omega = 2 * np.pi * params['frequency_hz']
    rho = 1000.0
    rho_omega2 = 400.0  # Effective parameter
    
    print(f"  ω = {omega:.1f} rad/s")
    print(f"  ρω² (effective): {rho_omega2:.1f}\n")
    
    scales = {
        'u_scale': u_scale,
        'mu_scale': mu_scale,
        'omega': omega,
        'rho_omega2': rho_omega2,
        'mu_min': mu_normalized.min(),
        'mu_max': mu_normalized.max()
    }
    
    return x, u_meas, mu_true, scales


def define_boundary_conditions(x, u_meas):
    """Define boundary conditions.
    
    For MRE inverse problems, we use ALL measurement points as soft constraints
    rather than traditional hard boundary conditions. This is because:
    1. We have displacement measurements everywhere
    2. MRE doesn't have fixed boundaries - it's an interior problem
    3. The data constraints guide the inversion
    """
    print("🎯 Defining soft boundary conditions...")
    
    # Use a small subset of edge points as "anchor" boundaries
    # This stabilizes the solver without over-constraining
    tol = 0.02  # Much tighter tolerance
    x_np = x.cpu().numpy()
    
    is_boundary = (
        (x_np[:, 0] < tol) | (x_np[:, 0] > 1 - tol) |
        (x_np[:, 1] < tol) | (x_np[:, 1] > 1 - tol) |
        (x_np[:, 2] < tol) | (x_np[:, 2] > 1 - tol)
    )
    
    bc_indices = torch.from_numpy(np.where(is_boundary)[0]).long().to(DEVICE)
    
    if len(bc_indices) == 0:
        # Fallback: use corners or a few points
        print("  Warning: No boundary points found, using corner points")
        bc_indices = torch.tensor([0, len(x)-1], dtype=torch.long, device=DEVICE)
    
    u_bc_vals = u_meas[bc_indices]
    
    print(f"  Found {len(bc_indices)} boundary points ({100*len(bc_indices)/len(x):.1f}% of data)")
    print(f"  Boundary u range: [{u_bc_vals.min():.3e}, {u_bc_vals.max():.3e}]\n")
    
    return bc_indices, u_bc_vals


# ============================================================================
# Training
# ============================================================================

def train_bioqic():
    """Main training function."""
    print("🔧 Configuration:")
    for k, v in CFG.items():
        print(f"  {k}: {v}")
    print(f"  Device: {DEVICE}\n")
    
    print("="*80)
    print("BIOQIC Phase 1 Training - Custom Stiffness Network")
    print("="*80)
    print()
    
    # Load data
    print("📂 Loading BIOQIC Phase 1 data...")
    coords, coords_norm, disp, stiff, params = load_bioqic_data()
    print(f"  Coordinates: {coords.shape}")
    print(f"  Displacement: {disp.shape} (complex)")
    print(f"  Stiffness: {stiff.shape} (complex)")
    print(f"  Frequency: {params['frequency_hz']} Hz")
    print(f"  Grid shape: {params['grid_shape']}\n")
    
    # Prepare training data
    x, u_meas, mu_true, scales = prepare_training_data(
        coords, coords_norm, disp, stiff, params
    )
    
    # Boundary conditions
    bc_indices, u_bc_vals = define_boundary_conditions(x, u_meas)
    
    # Initialize model with appropriate bounds
    print("🏗️  Initializing model...")
    model = CustomForwardMREModel(
        n_neurons_wave=CFG["n_wave_neurons"],
        input_dim=3,
        seed=CFG["seed"],
        mu_min=0.2,  # Allow slightly below true minimum for flexibility
        mu_max=1.5   # Allow slightly above true maximum
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.mu_net.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=CFG["lr_decay_step"], 
        gamma=CFG["lr_decay_gamma"]
    )
    
    print(f"  Wave neurons: {CFG['n_wave_neurons']}")
    print(f"  Stiffness bounds: [{0.2:.1f}, {1.5:.1f}] (normalized)")
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
            u_data=u_meas,
            data_weight=CFG["data_weight"],
            verbose=verbose
        )
        
        # Data loss
        loss_data = torch.mean((u_pred - u_meas) ** 2)
        
        # TV regularization
        dmu = mu_pred[1:] - mu_pred[:-1]
        tv_loss = torch.mean(torch.abs(dmu))
        
        # Total loss
        loss_total = loss_data + CFG["tv_weight"] * tv_loss
        
        # Backward
        loss_total.backward()
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
            
            # Denormalize
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
            bc_weight=CFG["bc_weight"],
            u_data=u_meas,
            data_weight=CFG["data_weight"]
        )
    
    return model, history, scales, x, u_meas, u_final, mu_true, mu_final


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    model, history, scales, x, u_meas, u_final, mu_true, mu_final = train_bioqic()
    
    # Print final statistics
    print("📈 Final Statistics:")
    loss_data = history['data_loss'][-1]
    u_rmse = np.sqrt(loss_data)
    
    mu_final_np = mu_final.detach().cpu().numpy()
    mu_true_np = mu_true.cpu().numpy()
    mu_rmse = np.sqrt(np.mean((mu_final_np - mu_true_np) ** 2)) * scales['mu_scale']
    mu_rel_error = np.abs((mu_final_np - mu_true_np) / mu_true_np) * 100
    
    print(f"  Data Loss: {loss_data:.6e}")
    print(f"  Displacement RMSE: {u_rmse:.6e}")
    print(f"  Stiffness RMSE: {mu_rmse:.1f} Pa")
    print(f"  Stiffness Rel Error: {np.mean(mu_rel_error):.1f}% ± {np.std(mu_rel_error):.1f}%")
    print(f"  Recovered μ range: [{mu_final.min().item()*scales['mu_scale']:.0f}, {mu_final.max().item()*scales['mu_scale']:.0f}] Pa")
    print(f"  True μ range: [{mu_true.min().item()*scales['mu_scale']:.0f}, {mu_true.max().item()*scales['mu_scale']:.0f}] Pa")
    
    print("\n" + "="*80)
    print("✅ All done!")
    print("="*80)
