"""
Detailed analysis of boundary condition rows in PIELM system.

Focus on:
1. How BC rows are constructed
2. Magnitude comparison: PDE rows vs BC rows vs Data rows
3. Condition number analysis
4. Why solution collapses to zero
"""

import torch
import numpy as np
from data_loader import BIOQICDataLoader
from pielm_polynomial import PIELMPolyModel
import sys


class ConstantMuNetwork:
    """Helper: Return constant ground truth μ."""
    def __init__(self, mu_true):
        self.mu_true = mu_true
    def __call__(self, x):
        return self.mu_true


class DetailedSystemAnalyzer:
    """Analyze PIELM system construction in detail."""
    
    @staticmethod
    def analyze_rows(H, b, n_pde, n_bc, n_data):
        """
        Detailed analysis of each row type in the system.
        
        Args:
            H: (N_total, M) system matrix
            b: (N_total, 1) right-hand side
            n_pde: Number of PDE rows
            n_bc: Number of BC rows
            n_data: Number of data rows
        """
        print("\n" + "="*70)
        print("DETAILED ROW ANALYSIS")
        print("="*70)
        
        # Extract different row types
        H_pde = H[:n_pde]
        b_pde = b[:n_pde]
        
        H_bc = H[n_pde:n_pde+n_bc]
        b_bc = b[n_pde:n_pde+n_bc]
        
        H_data = H[n_pde+n_bc:]
        b_data = b[n_pde+n_bc:]
        
        print(f"\n1. PDE ROWS (Physics constraint)")
        print(f"   Shape: H_pde {H_pde.shape}, b_pde {b_pde.shape}")
        print(f"   H_pde:")
        print(f"     Norm per row: min={torch.norm(H_pde, dim=1).min():.2e}, "
              f"max={torch.norm(H_pde, dim=1).max():.2e}, "
              f"mean={torch.norm(H_pde, dim=1).mean():.2e}")
        print(f"     Matrix norm: {torch.norm(H_pde):.2e}")
        print(f"     First row sample: {H_pde[0, :5]}")
        print(f"   b_pde:")
        print(f"     Values: min={b_pde.min():.2e}, max={b_pde.max():.2e}, mean={b_pde.mean():.2e}")
        print(f"     Norm: {torch.norm(b_pde):.2e}")
        print(f"     Should be: ~0 (PDE residual = 0)")
        
        print(f"\n2. BOUNDARY CONDITION ROWS")
        print(f"   Shape: H_bc {H_bc.shape}, b_bc {b_bc.shape}")
        print(f"   H_bc:")
        print(f"     Norm per row: min={torch.norm(H_bc, dim=1).min():.2e}, "
              f"max={torch.norm(H_bc, dim=1).max():.2e}, "
              f"mean={torch.norm(H_bc, dim=1).mean():.2e}")
        print(f"     Matrix norm: {torch.norm(H_bc):.2e}")
        print(f"     First row sample: {H_bc[0, :5]}")
        print(f"   b_bc:")
        print(f"     Values: min={b_bc.min():.2e}, max={b_bc.max():.2e}, mean={b_bc.mean():.2e}")
        print(f"     Norm: {torch.norm(b_bc):.2e}")
        
        print(f"\n3. DATA ROWS (Regularization)")
        print(f"   Shape: H_data {H_data.shape}, b_data {b_data.shape}")
        print(f"   H_data:")
        print(f"     Norm per row: min={torch.norm(H_data, dim=1).min():.2e}, "
              f"max={torch.norm(H_data, dim=1).max():.2e}, "
              f"mean={torch.norm(H_data, dim=1).mean():.2e}")
        print(f"     Matrix norm: {torch.norm(H_data):.2e}")
        print(f"     First row sample: {H_data[0, :5]}")
        print(f"   b_data:")
        print(f"     Values: min={b_data.min():.2e}, max={b_data.max():.2e}, mean={b_data.mean():.2e}")
        print(f"     Norm: {torch.norm(b_data):.2e}")
        
        print(f"\n4. MAGNITUDE COMPARISON")
        pde_row_norm = torch.norm(H_pde, dim=1).mean()
        bc_row_norm = torch.norm(H_bc, dim=1).mean()
        data_row_norm = torch.norm(H_data, dim=1).mean()
        
        print(f"   Average row norms:")
        print(f"     PDE:  {pde_row_norm:.2e}")
        print(f"     BC:   {bc_row_norm:.2e}  (ratio to PDE: {bc_row_norm/pde_row_norm:.2e})")
        print(f"     Data: {data_row_norm:.2e}  (ratio to PDE: {data_row_norm/pde_row_norm:.2e})")
        
        print(f"\n   Right-hand side norms:")
        print(f"     b_pde:  {torch.norm(b_pde):.2e}")
        print(f"     b_bc:   {torch.norm(b_bc):.2e}")
        print(f"     b_data: {torch.norm(b_data):.2e}")
        
        print(f"\n5. EFFECTIVE INFLUENCE ANALYSIS")
        # Effective influence = (row_norm / total_norm) * weight
        total_H_norm = torch.norm(H)
        pde_influence = (torch.norm(H_pde) / total_H_norm) * 1.0  # PDE weight = 1
        bc_influence = (torch.norm(H_bc) / total_H_norm) * 1.0   # Already weighted
        data_influence = (torch.norm(H_data) / total_H_norm) * 1.0  # Already weighted
        
        total_influence = pde_influence + bc_influence + data_influence
        
        print(f"   Relative influence (H matrix):")
        print(f"     PDE:  {pde_influence/total_influence*100:.1f}%")
        print(f"     BC:   {bc_influence/total_influence*100:.1f}%")
        print(f"     Data: {data_influence/total_influence*100:.1f}%")
        
        print(f"\n6. SYSTEM CONDITIONING")
        HtH = H.t() @ H
        cond = torch.linalg.cond(HtH)
        print(f"   Condition number (H^T H): {cond:.2e}")
        if cond > 1e10:
            print(f"   ⚠️  VERY ILL-CONDITIONED! (> 1e10)")
        elif cond > 1e6:
            print(f"   ⚠️  Ill-conditioned (> 1e6)")
        else:
            print(f"   ✓ Reasonably conditioned")
        
        return {
            'pde_row_norm': pde_row_norm.item(),
            'bc_row_norm': bc_row_norm.item(),
            'data_row_norm': data_row_norm.item(),
            'pde_influence': pde_influence.item(),
            'bc_influence': bc_influence.item(),
            'data_influence': data_influence.item(),
            'condition_number': cond.item()
        }


def detailed_bc_analysis():
    """Detailed boundary condition analysis."""
    
    print("="*70)
    print("DETAILED BOUNDARY CONDITION ANALYSIS")
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
    
    # Small subsample for detailed analysis
    subsample = 200
    np.random.seed(42)
    indices = np.random.choice(len(data['coords_norm']), subsample, replace=False)
    
    x = torch.tensor(data['coords_norm'][indices], dtype=torch.float32)
    u_meas = torch.tensor(data['u_data'][indices], dtype=torch.float32)
    mu_true = torch.tensor(data['mu_data'][indices], dtype=torch.float32)
    
    rho = 1000.0
    omega = data['scales']['omega']
    rho_omega2 = rho * omega**2
    
    print(f"   Subsample: {subsample} points")
    print(f"   ρω²: {rho_omega2:.2e}")
    
    # Create model
    mu_network = ConstantMuNetwork(mu_true)
    model = PIELMPolyModel(mu_network=mu_network, poly_degree=4, seed=42)
    
    # BC: random 10%
    n_bc = 20
    bc_indices = torch.randperm(subsample)[:n_bc]
    u_bc_vals = u_meas[bc_indices]
    
    print(f"\n2. Building system with DETAILED analysis...")
    print(f"   BC: {n_bc} points")
    print(f"   BC values: range [{u_bc_vals.min():.4f}, {u_bc_vals.max():.4f}]")
    
    # Get stiffness and basis
    mu_pred = model.mu_network(x)
    phi, phi_lap = model.get_basis(x)
    
    # Build system WITHOUT calling model.build_system (to inspect intermediate values)
    N, M = phi.shape
    print(f"   N points: {N}, M basis: {M}")
    
    print(f"\n3. ANALYZING INDIVIDUAL COMPONENTS BEFORE STACKING:")
    
    # PDE rows: A = μ·∇²φ + ρω²·φ
    print(f"\n   A) PDE term: μ·∇²φ")
    mu_phi_lap = mu_pred * phi_lap
    print(f"      μ range: [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
    print(f"      ∇²φ range: [{phi_lap.min():.2e}, {phi_lap.max():.2e}]")
    print(f"      μ·∇²φ range: [{mu_phi_lap.min():.2e}, {mu_phi_lap.max():.2e}]")
    print(f"      μ·∇²φ norm: {torch.norm(mu_phi_lap):.2e}")
    
    print(f"\n   B) PDE term: ρω²·φ")
    rho_omega2_phi = rho_omega2 * phi
    print(f"      ρω²: {rho_omega2:.2e}")
    print(f"      φ range: [{phi.min():.3f}, {phi.max():.3f}]")
    print(f"      ρω²·φ range: [{rho_omega2_phi.min():.2e}, {rho_omega2_phi.max():.2e}]")
    print(f"      ρω²·φ norm: {torch.norm(rho_omega2_phi):.2e}")
    
    print(f"\n   C) Full PDE: A = μ·∇²φ + ρω²·φ")
    A_pde = mu_phi_lap + rho_omega2_phi
    print(f"      A_pde range: [{A_pde.min():.2e}, {A_pde.max():.2e}]")
    print(f"      A_pde norm: {torch.norm(A_pde):.2e}")
    print(f"      ⚠️  HUGE! This is {torch.norm(A_pde)/torch.norm(mu_phi_lap):.1f}× larger than μ·∇²φ")
    
    print(f"\n   D) BC rows: w·φ(x_bc)")
    bc_weight = 100.0
    phi_bc = phi[bc_indices]
    H_bc = bc_weight * phi_bc
    b_bc = bc_weight * u_bc_vals
    print(f"      BC weight: {bc_weight}")
    print(f"      φ_bc range: [{phi_bc.min():.3f}, {phi_bc.max():.3f}]")
    print(f"      H_bc range: [{H_bc.min():.2e}, {H_bc.max():.2e}]")
    print(f"      H_bc norm: {torch.norm(H_bc):.2e}")
    print(f"      b_bc range: [{b_bc.min():.2e}, {b_bc.max():.2e}]")
    print(f"      b_bc norm: {torch.norm(b_bc):.2e}")
    
    print(f"\n   E) Data rows: w·φ")
    data_weight = 0.1
    H_data = data_weight * phi
    b_data = data_weight * u_meas
    print(f"      Data weight: {data_weight}")
    print(f"      H_data range: [{H_data.min():.2e}, {H_data.max():.2e}]")
    print(f"      H_data norm: {torch.norm(H_data):.2e}")
    print(f"      b_data norm: {torch.norm(b_data):.2e}")
    
    print(f"\n4. MAGNITUDE COMPARISON (BEFORE STACKING):")
    print(f"   ‖A_pde‖:  {torch.norm(A_pde):.2e}  ← PDE rows")
    print(f"   ‖H_bc‖:   {torch.norm(H_bc):.2e}   ← BC rows")
    print(f"   ‖H_data‖: {torch.norm(H_data):.2e}  ← Data rows")
    print(f"\n   Ratio PDE/BC:   {torch.norm(A_pde)/torch.norm(H_bc):.2e}")
    print(f"   Ratio PDE/Data: {torch.norm(A_pde)/torch.norm(H_data):.2e}")
    
    print(f"\n   ⚠️  PDE ROWS ARE DOMINATING BY ~10^8!")
    print(f"      This forces the solution to minimize PDE residual")
    print(f"      → Tries to solve: A·c = 0")
    print(f"      → Best solution: c ≈ 0 → u ≈ 0")
    
    # Now stack and analyze
    print(f"\n5. STACKED SYSTEM:")
    H = torch.cat([A_pde, H_bc, H_data], dim=0)
    b = torch.cat([torch.zeros(N, 1), b_bc, b_data], dim=0)
    
    analyzer = DetailedSystemAnalyzer()
    stats = analyzer.analyze_rows(H, b, N, n_bc, N)
    
    print(f"\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    print(f"\n🔴 ROOT CAUSE:")
    print(f"   ρω² = {rho_omega2:.2e} is ENORMOUS!")
    print(f"   It makes PDE rows ~10^8 times larger than BC/data rows")
    print(f"   System becomes: solve A·c ≈ 0 subject to tiny BC/data constraints")
    print(f"   → Solution: c ≈ 0 → u ≈ 0")
    
    print(f"\n💡 SOLUTIONS:")
    print(f"   1. NORMALIZE PDE term by ρω²:")
    print(f"      A_normalized = (μ·∇²φ + ρω²·φ) / ρω²")
    print(f"      = μ/(ρω²)·∇²φ + φ")
    print(f"      This brings PDE rows to O(1)")
    
    print(f"\n   2. INCREASE BC weight dramatically:")
    print(f"      Need: bc_weight ≈ {rho_omega2/100:.0e} (not 100!)")
    print(f"      To match PDE magnitude")
    
    print(f"\n   3. REFORMULATE PDE:")
    print(f"      Current: μ∇²u + ρω²u = 0")
    print(f"      Normalized: μ/(ρω²)∇²u + u = 0")
    print(f"      Or: ∇²u + (ρω²/μ)u = 0")


if __name__ == '__main__':
    detailed_bc_analysis()
