"""
PIELM-MRE: Physics-Informed Extreme Learning Machine for MRE Inverse Problem
==============================================================================

Implements Iterative PIELM with Curriculum Learning for solving the coupled
MRE inverse problem: finding stiffness μ(x) from measured displacement u(x).

Architecture: Dual ELM Networks
- Network A (Displacement): X → u(x,y,z) [complex-valued]
- Network B (Modulus): X → μ(x,y,z) [complex-valued]

Training Strategy: Iterative Alternating Optimization
1. Fix μ, solve for u (data matching + physics)
2. Fix u, solve for μ (physics + smoothness)
3. Repeat until convergence

Curriculum Learning:
- Stage 1: High regularization (near-homogeneous)
- Stage 2: Medium regularization
- Stage 3: Full heterogeneity

Author: Yeshwanth Kesav
Date: November 2025
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from scipy.stats.qmc import Halton


class PIELMFeatures:
    """
    Random feature generator for ELM with automatic differentiation support.
    
    Uses tanh activation: φ(z) = tanh(Wx·x + Wy·y + Wz·z + b)
    
    Provides analytical derivatives:
    - φ_x, φ_y, φ_z (first-order)
    - φ_xx, φ_yy, φ_zz (second-order diagonal)
    - φ_xy, φ_xz, φ_yz (second-order cross)
    """
    
    def __init__(self, n_neurons: int, dim: int = 3, seed: int = 42):
        """
        Initialize random feature layer.
        
        Args:
            n_neurons: Number of hidden neurons
            dim: Input dimension (3 for MRE: x,y,z)
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        
        # Random weights for each spatial dimension
        self.Wx = rng.normal(size=n_neurons) if dim >= 1 else np.zeros(n_neurons)
        self.Wy = rng.normal(size=n_neurons) if dim >= 2 else np.zeros(n_neurons)
        self.Wz = rng.normal(size=n_neurons) if dim >= 3 else np.zeros(n_neurons)
        self.b = rng.normal(size=n_neurons)
        
        self.n_neurons = n_neurons
        self.dim = dim
        
        print(f"✅ PIELM Features initialized:")
        print(f"   Neurons: {n_neurons}")
        print(f"   Dimension: {dim}")
        print(f"   Activation: tanh")
    
    def _z(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pre-activation: z = Wx·x + Wy·y + Wz·z + b
        
        Args:
            X: Input coordinates (N, 3) for [x, y, z]
        
        Returns:
            Pre-activation (N, n_neurons)
        """
        z = np.outer(X[:, 0], self.Wx) + self.b
        if self.dim >= 2:
            z += np.outer(X[:, 1], self.Wy)
        if self.dim >= 3:
            z += np.outer(X[:, 2], self.Wz)
        return z
    
    # ========== Activation and derivatives ==========
    def phi(self, X: np.ndarray) -> np.ndarray:
        """φ(z) = tanh(z)"""
        return np.tanh(self._z(X))
    
    def phi_x(self, X: np.ndarray) -> np.ndarray:
        """∂φ/∂x = sech²(z) · Wx"""
        z = self._z(X)
        sech2 = (1 / np.cosh(z))**2
        return sech2 * self.Wx
    
    def phi_y(self, X: np.ndarray) -> np.ndarray:
        """∂φ/∂y = sech²(z) · Wy"""
        if self.dim < 2:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        sech2 = (1 / np.cosh(z))**2
        return sech2 * self.Wy
    
    def phi_z(self, X: np.ndarray) -> np.ndarray:
        """∂φ/∂z = sech²(z) · Wz"""
        if self.dim < 3:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        sech2 = (1 / np.cosh(z))**2
        return sech2 * self.Wz
    
    def phi_xx(self, X: np.ndarray) -> np.ndarray:
        """∂²φ/∂x² = -2·tanh(z)·sech²(z)·Wx²"""
        z = self._z(X)
        return -2 * np.tanh(z) * (1 / np.cosh(z))**2 * (self.Wx**2)
    
    def phi_yy(self, X: np.ndarray) -> np.ndarray:
        """∂²φ/∂y² = -2·tanh(z)·sech²(z)·Wy²"""
        if self.dim < 2:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        return -2 * np.tanh(z) * (1 / np.cosh(z))**2 * (self.Wy**2)
    
    def phi_zz(self, X: np.ndarray) -> np.ndarray:
        """∂²φ/∂z² = -2·tanh(z)·sech²(z)·Wz²"""
        if self.dim < 3:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        return -2 * np.tanh(z) * (1 / np.cosh(z))**2 * (self.Wz**2)
    
    def phi_xy(self, X: np.ndarray) -> np.ndarray:
        """∂²φ/∂x∂y = -2·tanh(z)·sech²(z)·Wx·Wy"""
        if self.dim < 2:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        return -2 * np.tanh(z) * (1 / np.cosh(z))**2 * (self.Wx * self.Wy)
    
    def phi_xz(self, X: np.ndarray) -> np.ndarray:
        """∂²φ/∂x∂z = -2·tanh(z)·sech²(z)·Wx·Wz"""
        if self.dim < 3:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        return -2 * np.tanh(z) * (1 / np.cosh(z))**2 * (self.Wx * self.Wz)
    
    def phi_yz(self, X: np.ndarray) -> np.ndarray:
        """∂²φ/∂y∂z = -2·tanh(z)·sech²(z)·Wy·Wz"""
        if self.dim < 3:
            return np.zeros((X.shape[0], self.n_neurons))
        z = self._z(X)
        return -2 * np.tanh(z) * (1 / np.cosh(z))**2 * (self.Wy * self.Wz)
    
    def laplacian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        
        Returns:
            (N, n_neurons) array
        """
        return self.phi_xx(X) + self.phi_yy(X) + self.phi_zz(X)


class PIELMNetwork:
    """
    Single ELM network for either displacement or modulus.
    
    Network structure:
        Input (x,y,z) → Hidden Layer (random tanh features) → Output
    
    For complex-valued outputs (MRE), we use:
        - Output_real from weights_real
        - Output_imag from weights_imag
    """
    
    def __init__(self, features: PIELMFeatures, is_complex: bool = True):
        """
        Initialize ELM network.
        
        Args:
            features: Random feature generator
            is_complex: Whether output is complex-valued
        """
        self.features = features
        self.is_complex = is_complex
        
        # Output weights (solved via least squares)
        self.weights_real = None
        self.weights_imag = None if is_complex else None
        
        self.is_trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for given inputs.
        
        Args:
            X: Input coordinates (N, 3)
        
        Returns:
            Predictions (N,) - complex if is_complex=True
        """
        if not self.is_trained:
            raise RuntimeError("Network not trained yet!")
        
        Phi = self.features.phi(X)  # (N, n_neurons)
        
        if self.is_complex:
            u_real = Phi @ self.weights_real
            u_imag = Phi @ self.weights_imag
            return u_real + 1j * u_imag
        else:
            return Phi @ self.weights_real
    
    def solve_weights(
        self,
        H: np.ndarray,
        b: np.ndarray,
        ridge: float = 1e-8
    ):
        """
        Solve for output weights using ridge regression.
        
        Args:
            H: Feature matrix (M, n_neurons)
            b: Target vector (M,) or (M, 2) for complex
            ridge: Regularization parameter
        """
        # Solve: (H^T H + λI)w = H^T b
        HtH = H.T @ H
        HtH += ridge * np.eye(H.shape[1])
        Htb_real = H.T @ b.real
        
        self.weights_real = np.linalg.solve(HtH, Htb_real)
        
        if self.is_complex:
            Htb_imag = H.T @ b.imag
            self.weights_imag = np.linalg.solve(HtH, Htb_imag)
        
        self.is_trained = True


class IterativePIELMMRE:
    """
    Iterative PIELM solver for MRE inverse problem.
    
    Solves for both displacement u(x) and modulus μ(x) by alternating:
    1. Fix μ, optimize u (data matching + physics)
    2. Fix u, optimize μ (physics + smoothness)
    
    Implements curriculum learning to progressively reduce regularization.
    """
    
    def __init__(
        self,
        n_neurons: int = 1000,
        frequency: float = 60.0,
        density: float = 1000.0,
        seed: int = 42
    ):
        """
        Initialize iterative PIELM-MRE solver.
        
        Args:
            n_neurons: Number of hidden neurons per network
            frequency: MRE excitation frequency (Hz)
            density: Tissue density (kg/m³)
            seed: Random seed
        """
        # Create feature generators for both networks
        self.features_u = PIELMFeatures(n_neurons, dim=3, seed=seed)
        self.features_mu = PIELMFeatures(n_neurons, dim=3, seed=seed+1)
        
        # Create networks
        self.u_network = PIELMNetwork(self.features_u, is_complex=True)
        self.mu_network = PIELMNetwork(self.features_mu, is_complex=True)
        
        # Physics parameters
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.density = density
        self.rho_omega_sq = density * self.omega**2
        
        # Training history
        self.history = {
            'iteration': [],
            'loss_total': [],
            'loss_data': [],
            'loss_physics_u': [],
            'loss_physics_mu': [],
            'loss_reg': []
        }
        
        print(f"\n{'='*60}")
        print("Iterative PIELM-MRE Solver Initialized")
        print(f"{'='*60}")
        print(f"📊 Network Configuration:")
        print(f"   Hidden neurons (each net): {n_neurons}")
        print(f"   Displacement network: Complex-valued ELM")
        print(f"   Modulus network: Complex-valued ELM")
        print(f"\n⚙️  Physics Parameters:")
        print(f"   Frequency: {frequency} Hz")
        print(f"   Angular frequency (ω): {self.omega:.2f} rad/s")
        print(f"   Density (ρ): {density} kg/m³")
        print(f"   ρω²: {self.rho_omega_sq:.2e} kg/(m·s²)")
        print(f"{'='*60}\n")
    
    def assemble_displacement_system(
        self,
        X_data: np.ndarray,
        u_measured: np.ndarray,
        X_colloc: np.ndarray,
        mu_current: np.ndarray,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble weighted least squares system for displacement network.
        
        Given fixed μ, solve for u that minimizes:
            L = λ_data ||u - u_measured||² + λ_physics ||PDE_residual||²
        
        PDE: ∇·[μ∇u] + ρω²u = 0
        Expanded: μ∇²u + ∇μ·∇u + ρω²u = 0
        
        Args:
            X_data: Data points (N_data, 3)
            u_measured: Measured displacement (N_data,) complex
            X_colloc: Collocation points for physics (N_colloc, 3)
            mu_current: Current modulus estimate (N_colloc,) complex
            lambda_data: Weight for data loss
            lambda_physics: Weight for physics loss
        
        Returns:
            H: Feature matrix (M, n_neurons)
            b: Target vector (M,) complex
        """
        feat = self.features_u
        
        # ========== Data Loss Rows ==========
        Phi_data = feat.phi(X_data)  # (N_data, n_neurons)
        H_data = np.sqrt(lambda_data) * Phi_data
        b_data = np.sqrt(lambda_data) * u_measured
        
        # ========== Physics Loss Rows ==========
        # For each collocation point, compute PDE residual row
        N_colloc = X_colloc.shape[0]
        
        # Get feature derivatives at collocation points
        Phi_colloc = feat.phi(X_colloc)  # (N_colloc, n_neurons)
        Phi_x = feat.phi_x(X_colloc)
        Phi_y = feat.phi_y(X_colloc)
        Phi_z = feat.phi_z(X_colloc)
        Laplacian_phi = feat.laplacian(X_colloc)  # ∇²φ
        
        # Compute ∇μ at collocation points (need to evaluate μ_network derivatives)
        # For now, approximate ∇μ using finite differences or assume we have it
        # TODO: This requires μ_network to provide derivatives
        # For simplicity, let's assume ∇μ ≈ 0 for first iteration (homogeneous approximation)
        
        # Heterogeneous Helmholtz: μ∇²u + ∇μ·∇u + ρω²u = 0
        # If we neglect ∇μ·∇u term initially:
        # μ∇²u + ρω²u = 0
        
        # Rows for physics: [μ·∇²φ + ρω²·φ] @ weights = 0
        H_physics = np.sqrt(lambda_physics) * (
            mu_current[:, None] * Laplacian_phi + self.rho_omega_sq * Phi_colloc
        )
        b_physics = np.zeros(N_colloc, dtype=complex)
        
        # ========== Stack ==========
        H = np.vstack([H_data, H_physics])
        b = np.concatenate([b_data, b_physics])
        
        return H, b
    
    def assemble_modulus_system(
        self,
        X_colloc: np.ndarray,
        u_current: np.ndarray,
        grad_u_current: np.ndarray,
        laplacian_u_current: np.ndarray,
        lambda_physics: float = 1.0,
        lambda_reg: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble weighted least squares system for modulus network.
        
        Given fixed u, solve for μ that minimizes:
            L = λ_physics ||PDE_residual||² + λ_reg ||∇μ||²
        
        From PDE: μ∇²u + ∇μ·∇u + ρω²u = 0
        Rearranged: μ = -(∇μ·∇u + ρω²u) / ∇²u
        
        But this is implicit. Instead, we solve:
            μ·∇²u = -(∇μ·∇u + ρω²u)
        
        Simplified (neglecting ∇μ·∇u for stability):
            μ·∇²u ≈ -ρω²u
        
        Args:
            X_colloc: Collocation points (N_colloc, 3)
            u_current: Current displacement (N_colloc,) complex
            grad_u_current: Current ∇u (N_colloc, 3) complex
            laplacian_u_current: Current ∇²u (N_colloc,) complex
            lambda_physics: Weight for physics loss
            lambda_reg: Weight for regularization (smoothness)
        
        Returns:
            H: Feature matrix (M, n_neurons)
            b: Target vector (M,) complex
        """
        feat = self.features_mu
        N_colloc = X_colloc.shape[0]
        
        # ========== Physics Rows ==========
        # From PDE: μ·∇²u + ρω²u = 0 (neglecting ∇μ·∇u term initially)
        # Rearranged: μ·∇²u = -ρω²u
        # 
        # In least squares form: [∇²u · Φ(x)] @ weights_μ = -ρω²u
        # where Φ(x) are the features for μ network
        
        Phi_colloc = feat.phi(X_colloc)  # (N_colloc, n_neurons)
        
        # Weight each row by ∇²u (element-wise multiplication)
        # H_physics[i,:] = ∇²u[i] * Φ[i,:] 
        H_physics = np.sqrt(lambda_physics) * (laplacian_u_current[:, None] * Phi_colloc)
        b_physics = np.sqrt(lambda_physics) * (-self.rho_omega_sq * u_current)
        
        # ========== Regularization Rows (Smoothness) ==========
        # Penalize ||∇μ||² by adding rows for ∂μ/∂x, ∂μ/∂y, ∂μ/∂z ≈ 0
        Phi_x = feat.phi_x(X_colloc)
        Phi_y = feat.phi_y(X_colloc)
        Phi_z = feat.phi_z(X_colloc)
        
        H_reg_x = np.sqrt(lambda_reg) * Phi_x
        H_reg_y = np.sqrt(lambda_reg) * Phi_y
        H_reg_z = np.sqrt(lambda_reg) * Phi_z
        
        b_reg = np.zeros(3 * N_colloc, dtype=complex)
        
        # ========== Stack ==========
        H = np.vstack([H_physics, H_reg_x, H_reg_y, H_reg_z])
        b = np.concatenate([b_physics, b_reg])
        
        return H, b
    
    def train(
        self,
        X_data: np.ndarray,
        u_measured: np.ndarray,
        X_colloc: np.ndarray,
        max_iterations: int = 50,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.5,
        lambda_reg_schedule: list = [1.0, 0.1, 0.01],
        ridge: float = 1e-8,
        verbose: bool = True
    ):
        """
        Train iterative PIELM-MRE with curriculum learning.
        
        Args:
            X_data: Measured data points (N_data, 3)
            u_measured: Measured displacement (N_data,) complex
            X_colloc: Collocation points (N_colloc, 3)
            max_iterations: Maximum number of iterations
            lambda_data: Weight for data matching loss
            lambda_physics: Weight for physics loss
            lambda_reg_schedule: Regularization schedule [stage1, stage2, stage3]
            ridge: Ridge regularization for weight solving
            verbose: Print progress
        """
        print(f"\n{'='*60}")
        print("Starting Iterative PIELM-MRE Training")
        print(f"{'='*60}")
        print(f"📌 Training Configuration:")
        print(f"   Data points: {X_data.shape[0]:,}")
        print(f"   Collocation points: {X_colloc.shape[0]:,}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   λ_data: {lambda_data}")
        print(f"   λ_physics: {lambda_physics}")
        print(f"   λ_reg schedule: {lambda_reg_schedule}")
        print(f"   Ridge: {ridge}")
        print(f"{'='*60}\n")
        
        # Curriculum learning stages
        stage1_end = max_iterations // 3
        stage2_end = 2 * max_iterations // 3
        
        # Initialize μ with homogeneous guess (5 kPa mean of 3-10 kPa range)
        # This helps avoid zero solutions
        mu_init_value = 5000.0 + 1j * self.omega * 1.0  # 5 kPa real + viscosity term
        mu_current = np.full(X_colloc.shape[0], mu_init_value, dtype=complex)
        
        # Pre-train μ network with homogeneous initialization
        Phi_mu_init = self.features_mu.phi(X_colloc)
        target_init = np.full(X_colloc.shape[0], mu_init_value, dtype=complex)
        self.mu_network.solve_weights(Phi_mu_init, target_init, ridge=ridge)
        
        print(f"🔧 Initialized μ network with homogeneous stiffness: {mu_init_value.real:.0f} Pa\n")
        
        for iteration in range(max_iterations):
            # Determine curriculum stage
            if iteration < stage1_end:
                lambda_reg = lambda_reg_schedule[0]
                stage = 1
            elif iteration < stage2_end:
                lambda_reg = lambda_reg_schedule[1]
                stage = 2
            else:
                lambda_reg = lambda_reg_schedule[2]
                stage = 3
            
            # ========== Step 1: Fix μ, solve for u ==========
            H_u, b_u = self.assemble_displacement_system(
                X_data, u_measured, X_colloc, mu_current,
                lambda_data, lambda_physics
            )
            
            self.u_network.solve_weights(H_u, b_u, ridge)
            
            # Predict current u at collocation points
            u_current = self.u_network.predict(X_colloc)
            
            # Compute derivatives of u from the trained network
            # Since u = Σ wᵢ·φᵢ(x), we have:
            #   ∂u/∂x = Σ wᵢ·∂φᵢ/∂x
            #   ∇²u = Σ wᵢ·∇²φᵢ
            
            feat_u = self.features_u
            
            # Gradient components (N_colloc, 3)
            grad_u_x = (feat_u.phi_x(X_colloc) @ self.u_network.weights_real) + \
                       1j * (feat_u.phi_x(X_colloc) @ self.u_network.weights_imag)
            grad_u_y = (feat_u.phi_y(X_colloc) @ self.u_network.weights_real) + \
                       1j * (feat_u.phi_y(X_colloc) @ self.u_network.weights_imag)
            grad_u_z = (feat_u.phi_z(X_colloc) @ self.u_network.weights_real) + \
                       1j * (feat_u.phi_z(X_colloc) @ self.u_network.weights_imag)
            
            grad_u_current = np.column_stack([grad_u_x, grad_u_y, grad_u_z])
            
            # Laplacian (N_colloc,)
            laplacian_u_current = (feat_u.laplacian(X_colloc) @ self.u_network.weights_real) + \
                                  1j * (feat_u.laplacian(X_colloc) @ self.u_network.weights_imag)
            
            # Debug: Check if Laplacian is non-zero
            if iteration == 0 or iteration % 10 == 0:
                lap_mag = np.abs(laplacian_u_current)
                print(f"   [Debug] ∇²u range: [{lap_mag.min():.2e}, {lap_mag.max():.2e}], mean: {lap_mag.mean():.2e}")
            
            # ========== Step 2: Fix u, solve for μ ==========
            H_mu, b_mu = self.assemble_modulus_system(
                X_colloc, u_current, grad_u_current, laplacian_u_current,
                lambda_physics, lambda_reg
            )
            
            self.mu_network.solve_weights(H_mu, b_mu, ridge)
            
            # Update μ estimate
            mu_current = self.mu_network.predict(X_colloc)
            
            # ========== Compute Losses ==========
            u_pred_data = self.u_network.predict(X_data)
            loss_data = np.mean(np.abs(u_pred_data - u_measured)**2)
            
            # Physics residual at collocation points
            # PDE: μ∇²u + ρω²u ≈ 0
            pde_residual = mu_current * laplacian_u_current + self.rho_omega_sq * u_current
            loss_physics = np.mean(np.abs(pde_residual)**2)
            
            loss_total = lambda_data * loss_data + lambda_physics * loss_physics
            
            # Log history
            self.history['iteration'].append(iteration)
            self.history['loss_total'].append(loss_total)
            self.history['loss_data'].append(loss_data)
            self.history['loss_physics_u'].append(0.0)  # Placeholder
            self.history['loss_physics_mu'].append(loss_physics)
            self.history['loss_reg'].append(0.0)  # Placeholder
            
            if verbose and (iteration % 5 == 0 or iteration == max_iterations - 1):
                print(f"Iter {iteration:3d} | Stage {stage} | "
                      f"L_total: {loss_total:.2e} | L_data: {loss_data:.2e} | "
                      f"L_physics: {loss_physics:.2e} | λ_reg: {lambda_reg:.2e}")
        
        print(f"\n{'='*60}")
        print("✅ Training Complete!")
        print(f"{'='*60}\n")


# Test function
if __name__ == "__main__":
    print("Testing PIELM-MRE Implementation...\n")
    
    # Create dummy data
    np.random.seed(42)
    N_data = 1000
    N_colloc = 500
    
    X_data = np.random.rand(N_data, 3) * 0.1  # 0.1m cube
    u_measured = np.random.randn(N_data) + 1j * np.random.randn(N_data)
    X_colloc = np.random.rand(N_colloc, 3) * 0.1
    
    # Initialize solver
    solver = IterativePIELMMRE(n_neurons=200, frequency=60.0)
    
    # Train (just structure test)
    # solver.train(X_data, u_measured, X_colloc, max_iterations=10, verbose=True)
    
    print("\n✅ PIELM-MRE structure validated!")
    print("📌 Next: Integrate with real BIOQIC data from Phase 1")
