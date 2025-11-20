"""
PIELM Solver for Helmholtz Equation in MRE

Pure PIELM implementation following the formulation in pielm_formulation.md:
- Dual networks: u-network (displacement, complex) and μ-network (stiffness, real)
- Bernstein polynomial basis functions (well-conditioned)
- Direct linear solve: C = (H^T H)^(-1) H^T b
- No iterative gradient descent, only alternating linear solves

Mathematical Foundation:
-----------------------
u(x) = C_u^T φ_u(x)  (complex-valued displacement)
μ(x) = C_μ^T φ_μ(x)  (real-valued stiffness)

Helmholtz PDE:
μ(x)∇²u(x) + ρω²u(x) = 0

Matrix System (following PDF formulation):
H = [Ψ_data   ]  (Data fitting: φ(X_data))
    [Ψ_physics]  (PDE residual: L[φ](X_colloc))
    
b = [u_measured]  (Complex displacement measurements)
    [0         ]  (PDE should be satisfied)

Solve: C = H† b (pseudoinverse or ridge regression)
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings
from bernstein_basis import BernsteinBasis3D


class PIELMHelmholtzSolver:
    """
    Dual-network PIELM solver for MRE Helmholtz equation.
    
    Uses Bernstein polynomial basis for well-conditioned systems.
    Following the pure PIELM formulation:
    1. u-network: C_u^T φ_u(x) for displacement (complex)
    2. μ-network: C_μ^T φ_μ(x) for stiffness (real)
    3. Direct linear solve (no gradient descent)
    4. Alternating optimization: fix μ solve u, fix u solve μ
    """
    
    def __init__(self,
                 degrees_u: Tuple[int, int, int] = (8, 10, 6),
                 degrees_mu: Tuple[int, int, int] = (6, 8, 5),
                 domain: Tuple[Tuple[float, float], ...] = None,
                 omega: float = 2 * np.pi * 60,
                 rho: float = 1000.0):
        """
        Initialize dual PIELM networks with Bernstein basis.
        
        Parameters
        ----------
        degrees_u : tuple of int
            Polynomial degrees for u-network (nx, ny, nz)
            Features = (nx+1) * (ny+1) * (nz+1)
        degrees_mu : tuple of int
            Polynomial degrees for μ-network
        domain : tuple of tuples
            Physical domain bounds
        omega : float
            Angular frequency (rad/s), default 60 Hz
        rho : float
            Density (kg/m³)
        """
    """
    Random basis functions with analytical derivatives for PIELM.
    
    φ(x) = tanh(Wx + b)
    
    where W and b are randomly initialized and FIXED (never trained).
    """
    
    def __init__(self, 
                 n_features: int,
                 input_dim: int = 3,
                 domain: Tuple[Tuple[float, float], ...] = None,
                 seed: Optional[int] = None):
        """
        Initialize random basis functions.
        
        Parameters
        ----------
        n_features : int
            Number of hidden neurons (basis functions)
        input_dim : int
            Input dimension (3 for 3D space)
        domain : tuple of tuples
            Physical domain bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            Used for proper derivative scaling
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_features = n_features
        self.input_dim = input_dim
        self.domain = domain
        
        # Random initialization (FIXED, never updated)
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and biases
        # Use Xavier/Glorot initialization for stability
        scale = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(n_features, input_dim) * scale  # (n_features, 3)
        self.b = np.random.randn(n_features, 1) * 0.5  # (n_features, 1)
        
        # Domain scaling factors for derivatives
        if domain is not None:
            self.scale_x = 1.0 / (domain[0][1] - domain[0][0])
            self.scale_y = 1.0 / (domain[1][1] - domain[1][0])
            self.scale_z = 1.0 / (domain[2][1] - domain[2][0])
        else:
            self.scale_x = self.scale_y = self.scale_z = 1.0
        
        print(f"PIELMBasis initialized:")
        print(f"  Features: {n_features}")
        print(f"  Input dim: {input_dim}")
        print(f"  W shape: {self.W.shape}")
        print(f"  Domain scaling: ({self.scale_x:.3e}, {self.scale_y:.3e}, {self.scale_z:.3e})")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions φ(x) = tanh(Wx + b).
        
        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Input coordinates
            
        Returns
        -------
        phi : np.ndarray, shape (N, n_features)
            Basis function values
        """
        # X: (N, 3), W: (n_features, 3), b: (n_features, 1)
        z = X @ self.W.T + self.b.T  # (N, n_features)
        return np.tanh(z)
    
    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇φ(x) with proper domain scaling.
        
        For φ(x) = tanh(Wx + b), the gradient is:
        ∂φ/∂x_i = ∂φ/∂z · ∂z/∂x_i = sech²(z) · W_i
        
        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Input coordinates
            
        Returns
        -------
        grad_phi : np.ndarray, shape (N, n_features, 3)
            Gradient of each basis function
            grad_phi[i, j, k] = ∂φ_j/∂x_k at point i
        """
        N = X.shape[0]
        
        # Compute z and tanh(z)
        z = X @ self.W.T + self.b.T  # (N, n_features)
        tanh_z = np.tanh(z)  # (N, n_features)
        sech2_z = 1.0 - tanh_z**2  # sech²(z) = 1 - tanh²(z)
        
        # Apply chain rule: ∂φ/∂x = sech²(z) · W
        grad_phi = np.zeros((N, self.n_features, 3))
        
        # sech2_z: (N, n_features), W: (n_features, 3)
        # Need: (N, n_features, 3)
        grad_phi[:, :, 0] = sech2_z * self.W[:, 0]  # (N, n_features)
        grad_phi[:, :, 1] = sech2_z * self.W[:, 1]
        grad_phi[:, :, 2] = sech2_z * self.W[:, 2]
        
        return grad_phi
    
    def laplacian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²φ(x) = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z².
        
        For φ(x) = tanh(z) where z = Wx + b:
        ∂²φ/∂x²= ∂/∂x[sech²(z)·W_x] = ∂(sech²)/∂z · W_x · W_x
                = -2·tanh(z)·sech²(z)·W_x²
        
        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Input coordinates
            
        Returns
        -------
        lap_phi : np.ndarray, shape (N, n_features)
            Laplacian of each basis function
        """
        # Compute z and derivatives
        z = X @ self.W.T + self.b.T  # (N, n_features)
        tanh_z = np.tanh(z)
        sech2_z = 1.0 - tanh_z**2
        
        # Second derivative: d²(tanh(z))/dz² = -2·tanh(z)·sech²(z)
        d2_tanh_dz2 = -2.0 * tanh_z * sech2_z
        
        # Apply chain rule for each dimension
        # ∂²φ/∂x² = d²(tanh)/dz² · W_x²
        lap_phi = (d2_tanh_dz2 * self.W[:, 0]**2 + 
                   d2_tanh_dz2 * self.W[:, 1]**2 + 
                   d2_tanh_dz2 * self.W[:, 2]**2)
        
        return lap_phi


class PIELMHelmholtzSolver:
    """
    Dual-network PIELM solver for MRE Helmholtz equation.
    
    Following the pure PIELM formulation:
    1. u-network: C_u^T φ_u(x) for displacement (complex)
    2. μ-network: C_μ^T φ_μ(x) for stiffness (real)
    3. Direct linear solve (no gradient descent)
    4. Alternating optimization: fix μ solve u, fix u solve μ
    """
    
    def __init__(self,
                 n_features_u: int = 1000,
                 n_features_mu: int = 500,
                 domain: Tuple[Tuple[float, float], ...] = None,
                 omega: float = 2 * np.pi * 60,
                 rho: float = 1000.0,
                 seed: Optional[int] = None):
        """
        Initialize dual PIELM networks.
        
        Parameters
        ----------
        n_features_u : int
            Number of basis functions for u-network
        n_features_mu : int
            Number of basis functions for μ-network
        domain : tuple of tuples
            Physical domain bounds
        omega : float
            Angular frequency (rad/s), default 60 Hz
        rho : float
            Density (kg/m³)
        seed : int, optional
            Random seed
        """
        self.omega = omega
        self.rho = rho
        self.domain = domain
        
        print("\n" + "="*70)
        print("PIELM Helmholtz Solver Initialization")
        print("="*70)
        print(f"Physics parameters:")
        print(f"  Frequency: {omega / (2*np.pi):.1f} Hz")
        print(f"  Density: {rho} kg/m³")
        print(f"  ρω²: {rho * omega**2:.3e}")
        
        # Initialize basis functions (random, fixed)
        print("\nInitializing u-network basis:")
        self.basis_u = PIELMBasis(n_features_u, input_dim=3, domain=domain, seed=seed)
        
        print("\nInitializing μ-network basis:")
        seed_mu = seed + 1 if seed is not None else None
        self.basis_mu = PIELMBasis(n_features_mu, input_dim=3, domain=domain, seed=seed_mu)
        
        # Output weights (to be solved)
        self.C_u_real = None  # (n_features_u,) for real part
        self.C_u_imag = None  # (n_features_u,) for imag part
        self.C_mu = None      # (n_features_mu,) for stiffness
        
        # Training history
        self.history = {
            'loss_data': [],
            'loss_physics': [],
            'loss_total': [],
            'laplacian_magnitude': [],
            'mu_range': []
        }
    
    def solve_u_network(self,
                       X_data: np.ndarray,
                       u_data: np.ndarray,
                       X_colloc: np.ndarray,
                       mu_current: np.ndarray,
                       lambda_data: float = 1.0,
                       lambda_physics: float = 1.0,
                       ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for u-network weights given current μ field.
        
        Following PDF formulation:
        H = [√λ_data · φ(X_data)                    ]
            [√λ_physics · (μ∇²φ + ρω²φ)(X_colloc)  ]
            
        b = [√λ_data · u_data]
            [0               ]
        
        Solve: C = (H^T H + ridge·I)^(-1) H^T b
        
        Parameters
        ----------
        X_data : np.ndarray, shape (N_data, 3)
            Measurement locations
        u_data : np.ndarray, shape (N_data,), complex
            Complex displacement measurements
        X_colloc : np.ndarray, shape (N_colloc, 3)
            Collocation points for PDE
        mu_current : np.ndarray, shape (N_colloc,)
            Current stiffness field at collocation points
        lambda_data : float
            Weight for data fitting term
        lambda_physics : float
            Weight for physics term
        ridge : float
            Ridge regression parameter
            
        Returns
        -------
        C_u_real : np.ndarray, shape (n_features_u,)
            Output weights for real part
        C_u_imag : np.ndarray, shape (n_features_u,)
            Output weights for imaginary part
        """
        N_data = X_data.shape[0]
        N_colloc = X_colloc.shape[0]
        
        # --- Build H matrix ---
        
        # 1. Data fitting term: √λ_data · φ(X_data)
        phi_data = self.basis_u.forward(X_data)  # (N_data, n_features)
        H_data = np.sqrt(lambda_data) * phi_data
        
        # 2. Physics term: √λ_physics · (μ∇²φ + ρω²φ)(X_colloc)
        phi_colloc = self.basis_u.forward(X_colloc)  # (N_colloc, n_features)
        lap_phi_colloc = self.basis_u.laplacian(X_colloc)  # (N_colloc, n_features)
        
        # Helmholtz operator: L[φ] = μ∇²φ + ρω²φ
        # μ: (N_colloc,), lap_phi: (N_colloc, n_features)
        L_phi = mu_current[:, None] * lap_phi_colloc + self.rho * self.omega**2 * phi_colloc
        H_physics = np.sqrt(lambda_physics) * L_phi
        
        # Stack H matrix
        H = np.vstack([H_data, H_physics])  # (N_data + N_colloc, n_features)
        
        # --- Build b vector ---
        
        # Split complex data into real and imaginary
        b_data_real = np.sqrt(lambda_data) * np.real(u_data).reshape(-1, 1)
        b_data_imag = np.sqrt(lambda_data) * np.imag(u_data).reshape(-1, 1)
        
        b_physics = np.zeros((N_colloc, 1))  # PDE residual should be zero
        
        b_real = np.vstack([b_data_real, b_physics])  # (N_data + N_colloc, 1)
        b_imag = np.vstack([b_data_imag, b_physics])
        
        # --- Solve linear system: C = (H^T H + ridge·I)^(-1) H^T b ---
        
        HTH = H.T @ H  # (n_features, n_features)
        HTH_ridge = HTH + ridge * np.eye(self.basis_u.n_features)
        
        # Check condition number
        cond = np.linalg.cond(HTH_ridge)
        if cond > 1e10:
            warnings.warn(f"Ill-conditioned system for u: cond={cond:.2e}. Increase ridge parameter.")
        
        # Solve for real and imaginary parts separately
        HTb_real = H.T @ b_real
        HTb_imag = H.T @ b_imag
        
        try:
            # Use Cholesky for symmetric positive definite (faster)
            L = np.linalg.cholesky(HTH_ridge)
            C_u_real = np.linalg.solve(L.T, np.linalg.solve(L, HTb_real)).flatten()
            C_u_imag = np.linalg.solve(L.T, np.linalg.solve(L, HTb_imag)).flatten()
        except np.linalg.LinAlgError:
            # Fall back to general solver
            C_u_real = np.linalg.solve(HTH_ridge, HTb_real).flatten()
            C_u_imag = np.linalg.solve(HTH_ridge, HTb_imag).flatten()
        
        return C_u_real, C_u_imag
    
    def solve_mu_network(self,
                        X_colloc: np.ndarray,
                        u_current: np.ndarray,
                        lap_u_current: np.ndarray,
                        lambda_physics: float = 1.0,
                        lambda_reg: float = 0.1,
                        ridge: float = 1e-8) -> np.ndarray:
        """
        Solve for μ-network weights given current u field.
        
        From Helmholtz equation: μ∇²u + ρω²u = 0
        Rearrange: μ = -ρω²u / ∇²u
        
        But to avoid division, formulate as weighted least squares:
        H = [√λ_physics · ∇²u · φ_μ(X_colloc)]
            [√λ_reg · φ_μ(X_colloc)          ]  (regularization)
            
        b = [√λ_physics · (-ρω²u)]
            [√λ_reg · μ_prior     ]  (smooth prior)
        
        Parameters
        ----------
        X_colloc : np.ndarray, shape (N_colloc, 3)
            Collocation points
        u_current : np.ndarray, shape (N_colloc,), complex
            Current displacement field
        lap_u_current : np.ndarray, shape (N_colloc,), complex
            Current Laplacian of displacement
        lambda_physics : float
            Weight for physics constraint
        lambda_reg : float
            Weight for regularization
        ridge : float
            Ridge parameter
            
        Returns
        -------
        C_mu : np.ndarray, shape (n_features_mu,)
            Output weights for μ-network
        """
        N_colloc = X_colloc.shape[0]
        
        # Evaluate μ-network basis
        phi_mu = self.basis_mu.forward(X_colloc)  # (N_colloc, n_features_mu)
        
        # --- Build H matrix ---
        
        # 1. Physics constraint: ∇²u · φ_μ = -ρω²u
        #    Weight by |∇²u| to give more importance where Laplacian is large
        weight_physics = np.abs(lap_u_current)
        
        # For complex u, use magnitude for the system
        # μ·∇²u_real = -ρω²·u_real and μ·∇²u_imag = -ρω²·u_imag
        # Combine both constraints
        
        H_physics_real = np.sqrt(lambda_physics) * np.real(lap_u_current)[:, None] * phi_mu
        H_physics_imag = np.sqrt(lambda_physics) * np.imag(lap_u_current)[:, None] * phi_mu
        
        # 2. Regularization: prefer smooth μ (prior)
        H_reg = np.sqrt(lambda_reg) * phi_mu
        
        # Stack
        H = np.vstack([H_physics_real, H_physics_imag, H_reg])
        
        # --- Build b vector ---
        
        b_physics_real = np.sqrt(lambda_physics) * (-self.rho * self.omega**2 * np.real(u_current)).reshape(-1, 1)
        b_physics_imag = np.sqrt(lambda_physics) * (-self.rho * self.omega**2 * np.imag(u_current)).reshape(-1, 1)
        
        # Regularization target: use mean stiffness as prior (will be updated each iteration)
        if self.C_mu is not None:
            mu_prior_mean = np.mean(self.predict_mu(X_colloc))
        else:
            mu_prior_mean = 5000.0  # Initial guess: 5 kPa
        
        b_reg = np.sqrt(lambda_reg) * np.full((N_colloc, 1), mu_prior_mean)
        
        b = np.vstack([b_physics_real, b_physics_imag, b_reg])
        
        # --- Solve linear system ---
        
        HTH = H.T @ H
        HTH_ridge = HTH + ridge * np.eye(self.basis_mu.n_features)
        
        cond = np.linalg.cond(HTH_ridge)
        if cond > 1e10:
            warnings.warn(f"Ill-conditioned system for μ: cond={cond:.2e}")
        
        HTb = H.T @ b
        
        try:
            L = np.linalg.cholesky(HTH_ridge)
            C_mu = np.linalg.solve(L.T, np.linalg.solve(L, HTb)).flatten()
        except np.linalg.LinAlgError:
            C_mu = np.linalg.solve(HTH_ridge, HTb).flatten()
        
        # Ensure μ is positive (physical constraint)
        # Don't enforce this in weights, but will clip predictions
        
        return C_mu
    
    def predict_u(self, X: np.ndarray) -> np.ndarray:
        """Predict displacement u(x) = C_u^T φ_u(x)."""
        if self.C_u_real is None or self.C_u_imag is None:
            raise ValueError("u-network not trained yet. Call train() first.")
        
        phi = self.basis_u.forward(X)
        u_real = phi @ self.C_u_real
        u_imag = phi @ self.C_u_imag
        return u_real + 1j * u_imag
    
    def predict_mu(self, X: np.ndarray) -> np.ndarray:
        """Predict stiffness μ(x) = C_μ^T φ_μ(x)."""
        if self.C_mu is None:
            raise ValueError("μ-network not trained yet. Call train() first.")
        
        phi = self.basis_mu.forward(X)
        mu = phi @ self.C_mu
        
        # Enforce physical constraint: μ > 0
        mu = np.maximum(mu, 100.0)  # Minimum 100 Pa
        
        return mu
    
    def compute_laplacian_u(self, X: np.ndarray) -> np.ndarray:
        """Compute ∇²u at given points."""
        if self.C_u_real is None or self.C_u_imag is None:
            raise ValueError("u-network not trained yet.")
        
        lap_phi = self.basis_u.laplacian(X)
        lap_u_real = lap_phi @ self.C_u_real
        lap_u_imag = lap_phi @ self.C_u_imag
        return lap_u_real + 1j * lap_u_imag
    
    def train(self,
             X_data: np.ndarray,
             u_data: np.ndarray,
             X_colloc: np.ndarray,
             n_iterations: int = 20,
             lambda_data: float = 1.0,
             lambda_physics_schedule: list = None,
             lambda_reg_schedule: list = None,
             ridge: float = 1e-8,
             verbose: bool = True):
        """
        Train dual PIELM networks with alternating optimization.
        
        Parameters
        ----------
        X_data : np.ndarray, shape (N_data, 3)
            Measurement locations
        u_data : np.ndarray, shape (N_data,), complex
            Displacement measurements
        X_colloc : np.ndarray, shape (N_colloc, 3)
            Collocation points for PDE
        n_iterations : int
            Number of alternating iterations
        lambda_data : float
            Data fitting weight (constant)
        lambda_physics_schedule : list
            Physics weights for curriculum learning
        lambda_reg_schedule : list
            Regularization weights
        ridge : float
            Ridge regression parameter
        verbose : bool
            Print progress
        """
        if lambda_physics_schedule is None:
            lambda_physics_schedule = [0.1] * n_iterations
        if lambda_reg_schedule is None:
            lambda_reg_schedule = [1.0] * n_iterations
        
        if verbose:
            print("\n" + "="*70)
            print("Training Dual PIELM Networks")
            print("="*70)
            print(f"Data points: {X_data.shape[0]}")
            print(f"Collocation points: {X_colloc.shape[0]}")
            print(f"Iterations: {n_iterations}")
            print(f"λ_data: {lambda_data}")
            print(f"λ_physics: {lambda_physics_schedule}")
            print(f"λ_reg: {lambda_reg_schedule}")
        
        # Initialize μ with homogeneous guess
        mu_init = 5000.0 * np.ones(X_colloc.shape[0])  # 5 kPa
        
        for iter_idx in range(n_iterations):
            lambda_physics = lambda_physics_schedule[min(iter_idx, len(lambda_physics_schedule)-1)]
            lambda_reg = lambda_reg_schedule[min(iter_idx, len(lambda_reg_schedule)-1)]
            
            # --- Step 1: Fix μ, solve for u ---
            if iter_idx == 0:
                mu_current = mu_init
            else:
                mu_current = self.predict_mu(X_colloc)
            
            self.C_u_real, self.C_u_imag = self.solve_u_network(
                X_data, u_data, X_colloc, mu_current,
                lambda_data=lambda_data,
                lambda_physics=lambda_physics,
                ridge=ridge
            )
            
            # --- Step 2: Fix u, solve for μ ---
            u_current = self.predict_u(X_colloc)
            lap_u_current = self.compute_laplacian_u(X_colloc)
            
            self.C_mu = self.solve_mu_network(
                X_colloc, u_current, lap_u_current,
                lambda_physics=lambda_physics,
                lambda_reg=lambda_reg,
                ridge=ridge
            )
            
            # --- Evaluate metrics ---
            if verbose and iter_idx % 5 == 0:
                # Data loss
                u_pred = self.predict_u(X_data)
                loss_data = np.mean(np.abs(u_pred - u_data)**2)
                
                # Physics loss
                mu_pred = self.predict_mu(X_colloc)
                residual = mu_pred * lap_u_current + self.rho * self.omega**2 * u_current
                loss_physics = np.mean(np.abs(residual)**2)
                
                # Laplacian magnitude
                lap_mag = np.abs(lap_u_current)
                
                # μ range
                mu_range = [np.min(mu_pred), np.max(mu_pred)]
                
                print(f"\nIteration {iter_idx:3d}:")
                print(f"  L_data:    {loss_data:.3e}")
                print(f"  L_physics: {loss_physics:.3e}")
                print(f"  |∇²u|:     [{lap_mag.min():.3e}, {lap_mag.max():.3e}] mean: {lap_mag.mean():.3e}")
                print(f"  μ range:   [{mu_range[0]:.1f}, {mu_range[1]:.1f}] Pa")
                
                # Store history
                self.history['loss_data'].append(loss_data)
                self.history['loss_physics'].append(loss_physics)
                self.history['loss_total'].append(loss_data + loss_physics)
                self.history['laplacian_magnitude'].append(lap_mag.mean())
                self.history['mu_range'].append(mu_range)
        
        if verbose:
            print("\n" + "="*70)
            print("Training Complete!")
            print("="*70)
    
    def evaluate(self, 
                X_test: np.ndarray,
                u_true: np.ndarray = None,
                mu_true: np.ndarray = None) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test coordinates
        u_true : np.ndarray, optional
            True displacement
        mu_true : np.ndarray, optional
            True stiffness
            
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        metrics = {}
        
        # Predict
        u_pred = self.predict_u(X_test)
        mu_pred = self.predict_mu(X_test)
        
        if u_true is not None:
            mse_u = np.mean(np.abs(u_pred - u_true)**2)
            rel_err_u = np.sqrt(mse_u) / (np.std(np.abs(u_true)) + 1e-10)
            metrics['u_mse'] = mse_u
            metrics['u_rel_error'] = rel_err_u * 100
        
        if mu_true is not None:
            mae_mu = np.mean(np.abs(mu_pred - mu_true))
            rel_err_mu = mae_mu / (np.mean(mu_true) + 1e-10)
            metrics['mu_mae'] = mae_mu
            metrics['mu_rel_error'] = rel_err_mu * 100
            metrics['mu_pred_range'] = [np.min(mu_pred), np.max(mu_pred)]
            metrics['mu_true_range'] = [np.min(mu_true), np.max(mu_true)]
        
        return metrics
