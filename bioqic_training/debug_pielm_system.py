"""
Debug PIELM system construction.

Check:
1. What boundary conditions are being used?
2. How are the rows stacked in H and b?
3. Is b becoming all zeros?
4. Are PDE rows dominating and forcing u → 0?
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from pielm_polynomial import PIELMPolyModel
import sys


class DebugMuNetwork:
    """Return ground truth μ for debugging."""
    def __init__(self, mu_true):
        self.mu_true = mu_true
    def __call__(self, x):
        return self.mu_true


def debug_pielm_system():
    """Debug PIELM system construction step by step."""
    
    print("="*70)
    print("DEBUGGING PIELM SYSTEM CONSTRUCTION")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    loader = BIOQICDataLoader(
        data_dir='../data/processed/phase1_box',
        displacement_mode='magnitude',
        subsample=None
    )
    sys.stdout.flush()
    data = loader.load()
    
    # Small subsample for debugging
    subsample = 200  # Small for detailed inspection
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    mu_true = torch.tensor(data['mu_data'][indices], dtype=torch.float32)
    
    # Physics
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    print(f"   Subsample: {subsample} points")
    print(f"   u_meas: range [{u_meas.min():.4f}, {u_meas.max():.4f}], mean {u_meas.mean():.4f}")
    print(f"   μ_true: range [{mu_true.min():.4f}, {mu_true.max():.4f}]")
    print(f"   ρω²: {rho_omega2:.1f}")
    
    # Test different BC strategies
    print("\n2. Testing different boundary condition strategies...")
    print("="*70)
    
    bc_strategies = [
        {
            'name': 'Random 10%',
            'n_bc': int(0.1 * subsample),
            'bc_selection': 'random',
            'bc_weight': 100.0,
            'data_weight': 0.1
        },
        {
            'name': 'Random 25%',
            'n_bc': int(0.25 * subsample),
            'bc_selection': 'random',
            'bc_weight': 100.0,
            'data_weight': 0.1
        },
        {
            'name': 'Boundary edges (geometric)',
            'n_bc': None,  # Will compute based on geometry
            'bc_selection': 'edges',
            'bc_weight': 100.0,
            'data_weight': 0.1
        },
        {
            'name': 'High data weight',
            'n_bc': int(0.1 * subsample),
            'bc_selection': 'random',
            'bc_weight': 10.0,  # Lower BC weight
            'data_weight': 10.0  # Much higher data weight
        },
    ]
    
    for strategy in bc_strategies:
        print(f"\n--- Strategy: {strategy['name']} ---")
        
        # Get BC indices
        if strategy['bc_selection'] == 'random':
            bc_indices = torch.randperm(subsample)[:strategy['n_bc']]
        elif strategy['bc_selection'] == 'edges':
            # Find points on domain boundary
            coords_np = x.numpy()
            # Points where any coordinate is close to 0 or 1
            is_edge = (
                (coords_np[:, 0] < 0.05) | (coords_np[:, 0] > 0.95) |
                (coords_np[:, 1] < 0.05) | (coords_np[:, 1] > 0.95) |
                (coords_np[:, 2] < 0.05) | (coords_np[:, 2] > 0.95)
            )
            bc_indices = torch.tensor(np.where(is_edge)[0])
            print(f"  Found {len(bc_indices)} edge points")
        
        if len(bc_indices) == 0:
            print("  ⚠️ No BC points, skipping")
            continue
        
        u_bc_vals = u_meas[bc_indices]
        
        print(f"  BC points: {len(bc_indices)}")
        print(f"  BC values: range [{u_bc_vals.min():.4f}, {u_bc_vals.max():.4f}], mean {u_bc_vals.mean():.4f}")
        print(f"  BC weight: {strategy['bc_weight']:.1f}")
        print(f"  Data weight: {strategy['data_weight']:.3f}")
        
        # Create model
        mu_network = DebugMuNetwork(mu_true)
        model = PIELMPolyModel(mu_network=mu_network, poly_degree=4, seed=42)
        
        # Get basis
        phi, phi_lap = model.get_basis(x)
        
        # Build system WITH DETAILED INSPECTION
        print(f"\n  Building system...")
        H, b = model.build_system(
            mu_true, phi, phi_lap, rho_omega2,
            bc_indices, u_bc_vals, strategy['bc_weight'],
            u_meas, strategy['data_weight'], verbose=True
        )
        
        # Analyze the system
        print(f"\n  System analysis:")
        print(f"    H shape: {H.shape}")
        print(f"    b shape: {b.shape}")
        print(f"    H norm: {torch.norm(H).item():.2e}")
        print(f"    b norm: {torch.norm(b).item():.2e}")
        print(f"    b min/max: [{b.min().item():.4f}, {b.max().item():.4f}]")
        print(f"    b mean: {b.mean().item():.4f}")
        
        # Check if b has structure
        n_pde = subsample
        n_bc = len(bc_indices)
        n_data = subsample
        
        b_pde = b[:n_pde]
        b_bc = b[n_pde:n_pde+n_bc]
        b_data = b[n_pde+n_bc:]
        
        print(f"\n  Right-hand side breakdown:")
        print(f"    PDE rows:  norm={torch.norm(b_pde).item():.2e}, should be ~0")
        print(f"    BC rows:   norm={torch.norm(b_bc).item():.2e}, mean={b_bc.mean().item():.3f}")
        print(f"    Data rows: norm={torch.norm(b_data).item():.2e}, mean={b_data.mean().item():.3f}")
        
        # Solve
        from pielm_solver import pielm_solve
        c = pielm_solve(H, b, verbose=False)
        u_pred = phi @ c
        
        mse = torch.mean((u_pred - u_meas)**2).item()
        
        print(f"\n  Solution:")
        print(f"    u_pred: range [{u_pred.min():.4f}, {u_pred.max():.4f}], mean {u_pred.mean():.4f}")
        print(f"    MSE: {mse:.6f}")
        
        # Check if solution satisfies constraints
        pde_residual = (mu_true * phi_lap + rho_omega2 * phi) @ c
        pde_residual_norm = torch.norm(pde_residual).item()
        bc_error = torch.norm(u_pred[bc_indices] - u_bc_vals).item()
        
        print(f"    PDE residual norm: {pde_residual_norm:.2e}")
        print(f"    BC error: {bc_error:.4f}")
        
        if mse < 0.05:
            print(f"  ✅ Good fit!")
        elif u_pred.abs().max() < 0.01:
            print(f"  ❌ Solution collapsed to ~zero!")
        else:
            print(f"  ⚠️  Poor fit")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    print("If u_pred ≈ 0 consistently:")
    print("  → PDE rows are dominating the system")
    print("  → PDE: μ∇²u + ρω²u = 0 forces u → 0")
    print("  → This happens because BIOQIC data doesn't satisfy elastic PDE")
    print("\nPossible fixes:")
    print("  1. Much higher BC/data weights (1000+)")
    print("  2. Normalize PDE term by ρω² to balance magnitudes")
    print("  3. Implement complex-valued PDE")


if __name__ == '__main__':
    debug_pielm_system()
