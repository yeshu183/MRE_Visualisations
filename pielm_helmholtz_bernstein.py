"""
PIELM Solver for Helmholtz Equation - Using Bernstein Basis

Pure PIELM implementation with Bernstein polynomials for MRE inverse problem.
Well-conditioned systems (κ ~ 1e2-1e4) vs tanh features (κ ~ 1e20).
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings
from bernstein_basis import BernsteinBasis3D


class PIELMHelmholtzSolver:
    """
    Dual-network PIELM solver using Bernstein polynomial basis.
    
    Architecture:
    - u-network: C_u^T φ_u(x) for displacement (complex-valued)
    - μ-network: C_μ^T φ_μ(x) for stiffness (real-valued)
    - Direct linear solve per iteration (no gradient descent)
    - Alternating optimization between networks
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
            Total features = (nx+1) * (ny+1) * (nz+1)
        degrees_mu : tuple of int
            Polynomial degrees for μ-network
        domain : tuple of tuples
            Physical domain ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        omega : float
            Angular frequency (rad/s)
        rho : float
            Density (kg/m³)
        """
        self.omega = omega
        self.rho = rho
        self.domain = domain
        
        print("\n" + "="*70)
        print("PIELM Helmholtz Solver - Bernstein Basis")
        print("="*70)
        print(f"Physics parameters:")
        print(f"  Frequency: {omega / (2*np.pi):.1f} Hz")
        print(f"  Density: {rho} kg/m³")
        print(f"  ρω²: {rho * omega**2:.3e}")
        
        # Initialize Bernstein basis functions
        print("\nInitializing u-network (Bernstein basis):")
        self.basis_u = BernsteinBasis3D(degrees=degrees_u, domain=domain)
        
        print("\nInitializing μ-network (Bernstein basis):")
        self.basis_mu = BernsteinBasis3D(degrees=degrees_mu, domain=domain)
        
        # Output weights (to be solved)
        self.C_u_real = None
        self.C_u_imag = None
        self.C_mu = None
        
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
                       ridge: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for u-network weights given current μ field.
        
        System:
        H = [√λ_data · φ(X_data)                    ]
            [√λ_physics · (μ∇²φ + ρω²φ)(X_colloc)  ]
        
        b = [√λ_data · u_data]
            [0               ]
        
        Parameters
        ----------
        X_data : np.ndarray, shape (N_data, 3)
            Measurement locations
        u_data : np.ndarray, shape (N_data,), complex
            Complex displacement measurements
        X_colloc : np.ndarray, shape (N_colloc, 3)
            Collocation points for PDE
        mu_current : np.ndarray, shape (N_colloc,)
            Current stiffness field
        lambda_data : float
            Data fitting weight
        lambda_physics : float
            Physics constraint weight
        ridge : float
            Ridge parameter (Bernstein allows 1e-10 to 1e-12)
            
        Returns
        -------
        C_u_real, C_u_imag : np.ndarray
            Output weights for real and imaginary parts
        """
        N_data = X_data.shape[0]
        N_colloc = X_colloc.shape[0]
        
        # Build H matrix
        # 1. Data term
        phi_data = self.basis_u.forward(X_data)
        H_data = np.sqrt(lambda_data) * phi_data
        
        # 2. Physics term: Helmholtz operator L[φ] = μ∇²φ + ρω²φ
        phi_colloc = self.basis_u.forward(X_colloc)
        lap_phi_colloc = self.basis_u.laplacian(X_colloc)
        
        L_phi = mu_current[:, None] * lap_phi_colloc + self.rho * self.omega**2 * phi_colloc
        H_physics = np.sqrt(lambda_physics) * L_phi
        
        # Stack
        H = np.vstack([H_data, H_physics])
        
        # Build b vector
        b_data_real = np.sqrt(lambda_data) * np.real(u_data).reshape(-1, 1)
        b_data_imag = np.sqrt(lambda_data) * np.imag(u_data).reshape(-1, 1)
        b_physics = np.zeros((N_colloc, 1))
        
        b_real = np.vstack([b_data_real, b_physics])
        b_imag = np.vstack([b_data_imag, b_physics])
        
        # Solve: C = (H^T H + ridge·I)^{-1} H^T b
        HTH = H.T @ H
        HTH_ridge = HTH + ridge * np.eye(self.basis_u.n_features)
        
        # Check conditioning
        cond = np.linalg.cond(HTH_ridge)
        if cond > 1e10:
            warnings.warn(f"Ill-conditioned u-system: cond={cond:.2e}")
        
        HTb_real = H.T @ b_real
        HTb_imag = H.T @ b_imag
        
        try:
            # Cholesky (fastest for SPD)
            L = np.linalg.cholesky(HTH_ridge)
            C_u_real = np.linalg.solve(L.T, np.linalg.solve(L, HTb_real)).flatten()
            C_u_imag = np.linalg.solve(L.T, np.linalg.solve(L, HTb_imag)).flatten()
        except np.linalg.LinAlgError:
            # Fallback
            C_u_real = np.linalg.solve(HTH_ridge, HTb_real).flatten()
            C_u_imag = np.linalg.solve(HTH_ridge, HTb_imag).flatten()
        
        return C_u_real, C_u_imag
    
    def solve_mu_network(self,
                        X_colloc: np.ndarray,
                        u_current: np.ndarray,
                        lap_u_current: np.ndarray,
                        lambda_physics: float = 1.0,
                        lambda_reg: float = 0.1,
                        ridge: float = 1e-10) -> np.ndarray:
        """
        Solve for μ-network weights given current u field.
        
        From Helmholtz: μ∇²u + ρω²u = 0
        Formulate as: ∇²u · φ_μ · μ = -ρω²u
        
        System:
        H = [√λ_physics · ∇²u_real · φ_μ]
            [√λ_physics · ∇²u_imag · φ_μ]
            [√λ_reg · φ_μ             ]
        
        b = [√λ_physics · (-ρω²u_real)]
            [√λ_physics · (-ρω²u_imag)]
            [√λ_reg · μ_prior        ]
        
        Parameters
        ----------
        X_colloc : np.ndarray
            Collocation points
        u_current : np.ndarray, complex
            Current displacement field
        lap_u_current : np.ndarray, complex
            Laplacian of displacement
        lambda_physics : float
            Physics weight
        lambda_reg : float
            Regularization weight
        ridge : float
            Ridge parameter
            
        Returns
        -------
        C_mu : np.ndarray
            Output weights for μ-network
        """
        N_colloc = X_colloc.shape[0]
        
        # Evaluate μ basis
        phi_mu = self.basis_mu.forward(X_colloc)
        
        # Build H matrix
        # Split complex into real and imaginary constraints
        H_physics_real = np.sqrt(lambda_physics) * np.real(lap_u_current)[:, None] * phi_mu
        H_physics_imag = np.sqrt(lambda_physics) * np.imag(lap_u_current)[:, None] * phi_mu
        H_reg = np.sqrt(lambda_reg) * phi_mu
        
        H = np.vstack([H_physics_real, H_physics_imag, H_reg])
        
        # Build b vector
        b_physics_real = np.sqrt(lambda_physics) * (-self.rho * self.omega**2 * np.real(u_current)).reshape(-1, 1)
        b_physics_imag = np.sqrt(lambda_physics) * (-self.rho * self.omega**2 * np.imag(u_current)).reshape(-1, 1)
        
        # Prior: smooth field around current mean
        if self.C_mu is not None:
            mu_prior = np.mean(self.predict_mu(X_colloc))
        else:
            mu_prior = 5000.0  # Initial: 5 kPa
        
        b_reg = np.sqrt(lambda_reg) * np.full((N_colloc, 1), mu_prior)
        
        b = np.vstack([b_physics_real, b_physics_imag, b_reg])
        
        # Solve
        HTH = H.T @ H
        HTH_ridge = HTH + ridge * np.eye(self.basis_mu.n_features)
        
        cond = np.linalg.cond(HTH_ridge)
        if cond > 1e10:
            warnings.warn(f"Ill-conditioned μ-system: cond={cond:.2e}")
        
        HTb = H.T @ b
        
        try:
            L = np.linalg.cholesky(HTH_ridge)
            C_mu = np.linalg.solve(L.T, np.linalg.solve(L, HTb)).flatten()
        except np.linalg.LinAlgError:
            C_mu = np.linalg.solve(HTH_ridge, HTb).flatten()
        
        return C_mu
    
    def predict_u(self, X: np.ndarray) -> np.ndarray:
        """Predict displacement u(x) = C_u^T φ_u(x)."""
        if self.C_u_real is None:
            raise ValueError("u-network not trained")
        
        phi = self.basis_u.forward(X)
        u_real = phi @ self.C_u_real
        u_imag = phi @ self.C_u_imag
        return u_real + 1j * u_imag
    
    def predict_mu(self, X: np.ndarray) -> np.ndarray:
        """Predict stiffness μ(x) = C_μ^T φ_μ(x)."""
        if self.C_mu is None:
            raise ValueError("μ-network not trained")
        
        phi = self.basis_mu.forward(X)
        mu = phi @ self.C_mu
        
        # Physical constraint: μ > 0
        mu = np.maximum(mu, 100.0)
        
        return mu
    
    def compute_laplacian_u(self, X: np.ndarray) -> np.ndarray:
        """Compute ∇²u at given points."""
        if self.C_u_real is None:
            raise ValueError("u-network not trained")
        
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
             ridge: float = 1e-10,
             verbose: bool = True):
        """
        Train dual networks with alternating optimization.
        
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
            Data weight
        lambda_physics_schedule : list
            Physics weights (curriculum learning)
        lambda_reg_schedule : list
            Regularization weights
        ridge : float
            Ridge parameter (1e-10 to 1e-12 for Bernstein)
        verbose : bool
            Print progress
        """
        if lambda_physics_schedule is None:
            lambda_physics_schedule = [0.1, 0.5, 1.0, 5.0, 10.0] + [10.0] * (n_iterations - 5)
        if lambda_reg_schedule is None:
            lambda_reg_schedule = [10.0, 5.0, 1.0, 0.5, 0.1] + [0.1] * (n_iterations - 5)
        
        if verbose:
            print("\n" + "="*70)
            print("Training Dual PIELM Networks")
            print("="*70)
            print(f"Data points: {X_data.shape[0]}")
            print(f"Collocation points: {X_colloc.shape[0]}")
            print(f"Iterations: {n_iterations}")
            print(f"Ridge parameter: {ridge:.2e}")
        
        # Initialize μ
        mu_init = 5000.0 * np.ones(X_colloc.shape[0])
        
        for iter_idx in range(n_iterations):
            lambda_physics = lambda_physics_schedule[min(iter_idx, len(lambda_physics_schedule)-1)]
            lambda_reg = lambda_reg_schedule[min(iter_idx, len(lambda_reg_schedule)-1)]
            
            # Step 1: Fix μ, solve u
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
            
            # Step 2: Fix u, solve μ
            u_current = self.predict_u(X_colloc)
            lap_u_current = self.compute_laplacian_u(X_colloc)
            
            self.C_mu = self.solve_mu_network(
                X_colloc, u_current, lap_u_current,
                lambda_physics=lambda_physics,
                lambda_reg=lambda_reg,
                ridge=ridge
            )
            
            # Metrics
            if verbose and iter_idx % 5 == 0:
                u_pred = self.predict_u(X_data)
                loss_data = np.mean(np.abs(u_pred - u_data)**2)
                
                mu_pred = self.predict_mu(X_colloc)
                residual = mu_pred * lap_u_current + self.rho * self.omega**2 * u_current
                loss_physics = np.mean(np.abs(residual)**2)
                
                lap_mag = np.abs(lap_u_current)
                mu_range = [np.min(mu_pred), np.max(mu_pred)]
                
                print(f"\nIteration {iter_idx:3d}:")
                print(f"  L_data:    {loss_data:.3e}")
                print(f"  L_physics: {loss_physics:.3e}")
                print(f"  |∇²u|:     [{lap_mag.min():.3e}, {lap_mag.max():.3e}] mean: {lap_mag.mean():.3e}")
                print(f"  μ range:   [{mu_range[0]:.1f}, {mu_range[1]:.1f}] Pa")
                
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
        
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        metrics = {}
        
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
