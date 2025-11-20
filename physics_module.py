"""
Physics Module for PIELM-MRE
=============================

Implements the physics equations for MRE inverse problem:
- Helmholtz equation (homogeneous and heterogeneous forms)
- Automatic differentiation for spatial derivatives
- PDE residual calculators
- Boundary conditions

Author: Yeshwanth Kesav
Date: November 2025
"""

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from typing import Tuple, Callable
import numpy as np


class HelmholtzPhysics:
    """
    Physics-informed operators for MRE inverse problem.
    
    The governing equation is the Helmholtz equation for elastic wave propagation:
        ∇·[μ(x)∇u(x)] + ρω²u(x) = 0
    
    where:
        u(x): displacement field (complex)
        μ(x): complex shear modulus (unknown, to be estimated)
        ρ: tissue density (assumed constant ≈ 1000 kg/m³)
        ω: angular frequency (2πf)
    """
    
    def __init__(self, frequency: float = 60.0, density: float = 1000.0):
        """
        Initialize physics module.
        
        Args:
            frequency: Excitation frequency in Hz (default: 60 Hz)
            density: Tissue density in kg/m³ (default: 1000 kg/m³ for soft tissue)
        """
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency  # Angular frequency
        self.density = density
        self.rho_omega_sq = density * self.omega**2
        
        print(f"Physics Module Initialized:")
        print(f"  Frequency: {frequency} Hz")
        print(f"  Angular frequency (ω): {self.omega:.2f} rad/s")
        print(f"  Density (ρ): {density} kg/m³")
        print(f"  ρω²: {self.rho_omega_sq:.2e} kg/(m·s²)")
    
    @staticmethod
    @jit
    def gradient_3d(u_net: Callable, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute spatial gradient ∇u = [∂u/∂x, ∂u/∂y, ∂u/∂z].
        
        Args:
            u_net: Neural network function u(x,y,z) -> u
            x: Input coordinates (N x 3)
        
        Returns:
            Gradient tensor (N x 3) for [∂u/∂x, ∂u/∂y, ∂u/∂z]
        """
        grad_fn = vmap(grad(lambda xi: u_net(xi).sum()))
        return grad_fn(x)
    
    @staticmethod
    @jit
    def laplacian_3d(u_net: Callable, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z².
        
        Args:
            u_net: Neural network function u(x,y,z) -> u
            x: Input coordinates (N x 3)
        
        Returns:
            Laplacian values (N,)
        """
        def second_derivative(xi):
            # Compute trace of Hessian (sum of diagonal elements)
            hessian = jax.hessian(lambda x_: u_net(x_).sum())(xi)
            return jnp.trace(hessian)
        
        laplacian_fn = vmap(second_derivative)
        return laplacian_fn(x)
    
    def helmholtz_residual_homogeneous(
        self,
        u_net: Callable,
        x: jnp.ndarray,
        mu_constant: float = 3000.0
    ) -> jnp.ndarray:
        """
        Compute PDE residual for HOMOGENEOUS Helmholtz equation:
            μ∇²u + ρω²u = 0
        
        This assumes constant stiffness μ across the domain (simplified case for Phase 1).
        
        Args:
            u_net: Displacement network u(x,y,z) -> u
            x: Collocation points (N x 3)
            mu_constant: Constant shear modulus value (Pa)
        
        Returns:
            PDE residual (N,) - should be close to zero
        """
        # Get displacement values
        u = vmap(lambda xi: u_net(xi))(x)
        
        # Compute Laplacian
        laplacian_u = self.laplacian_3d(u_net, x)
        
        # Helmholtz equation residual: μ∇²u + ρω²u
        residual = mu_constant * laplacian_u + self.rho_omega_sq * u
        
        return residual
    
    def helmholtz_residual_heterogeneous(
        self,
        u_net: Callable,
        mu_net: Callable,
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute PDE residual for HETEROGENEOUS Helmholtz equation:
            ∇·[μ(x)∇u(x)] + ρω²u(x) = 0
        
        Expanded form:
            μ∇²u + ∇μ·∇u + ρω²u = 0
        
        Args:
            u_net: Displacement network u(x,y,z) -> u
            mu_net: Modulus network μ(x,y,z) -> μ
            x: Collocation points (N x 3)
        
        Returns:
            PDE residual (N,) - should be close to zero
        """
        # Get displacement and modulus values
        u = vmap(lambda xi: u_net(xi))(x)
        mu = vmap(lambda xi: mu_net(xi))(x)
        
        # Compute spatial derivatives
        grad_u = self.gradient_3d(u_net, x)  # ∇u (N x 3)
        grad_mu = self.gradient_3d(mu_net, x)  # ∇μ (N x 3)
        laplacian_u = self.laplacian_3d(u_net, x)  # ∇²u (N,)
        
        # Compute ∇μ·∇u (dot product)
        grad_mu_dot_grad_u = jnp.sum(grad_mu * grad_u, axis=1)  # (N,)
        
        # Heterogeneous Helmholtz residual: μ∇²u + ∇μ·∇u + ρω²u
        residual = mu * laplacian_u + grad_mu_dot_grad_u + self.rho_omega_sq * u
        
        return residual
    
    @staticmethod
    def stress_free_boundary_condition(
        u_net: Callable,
        mu_net: Callable,
        x_boundary: jnp.ndarray,
        normals: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute stress-free boundary condition residual:
            σ·n = μ(∇u + ∇u^T)·n = 0
        
        For shear waves in MRE, boundaries are often assumed stress-free.
        
        Args:
            u_net: Displacement network
            mu_net: Modulus network
            x_boundary: Boundary points (N x 3)
            normals: Outward normal vectors (N x 3)
        
        Returns:
            Boundary condition residual (N,)
        """
        # Get modulus at boundary
        mu = vmap(lambda xi: mu_net(xi))(x_boundary)
        
        # Compute displacement gradient
        grad_u = HelmholtzPhysics.gradient_3d(u_net, x_boundary)  # (N x 3)
        
        # Stress: σ = μ(∇u + ∇u^T), simplified to σ ≈ μ∇u for scalar case
        # Traction: t = σ·n ≈ μ(∇u·n)
        traction = mu[:, None] * jnp.sum(grad_u * normals, axis=1, keepdims=True)
        
        return traction.squeeze()
    
    def compute_wave_number(self, mu: float) -> float:
        """
        Compute wave number k = ω√(ρ/μ) for given stiffness.
        
        Args:
            mu: Shear modulus (Pa)
        
        Returns:
            Wave number (1/m)
        """
        k = self.omega * np.sqrt(self.density / mu)
        return k
    
    def compute_wavelength(self, mu: float) -> float:
        """
        Compute wavelength λ = 2π/k = 2π√(μ/ρ)/ω.
        
        Args:
            mu: Shear modulus (Pa)
        
        Returns:
            Wavelength (m)
        """
        k = self.compute_wave_number(mu)
        wavelength = 2 * np.pi / k
        return wavelength


class PhysicsLoss:
    """
    Loss functions for physics-informed training.
    """
    
    def __init__(self, physics: HelmholtzPhysics):
        """
        Initialize physics loss calculator.
        
        Args:
            physics: HelmholtzPhysics instance
        """
        self.physics = physics
    
    def data_loss(
        self,
        u_pred: jnp.ndarray,
        u_measured: jnp.ndarray,
        weights: jnp.ndarray = None
    ) -> float:
        """
        Data matching loss: ||u_pred - u_measured||²
        
        Args:
            u_pred: Predicted displacement (N,)
            u_measured: Measured displacement from MRI (N,)
            weights: Optional weights for each point (N,)
        
        Returns:
            Mean squared error
        """
        diff = u_pred - u_measured
        
        if weights is not None:
            loss = jnp.mean(weights * jnp.abs(diff)**2)
        else:
            loss = jnp.mean(jnp.abs(diff)**2)
        
        return loss
    
    def physics_loss_homogeneous(
        self,
        u_net: Callable,
        x_colloc: jnp.ndarray,
        mu_constant: float = 3000.0
    ) -> float:
        """
        Physics loss for homogeneous case: ||PDE residual||²
        
        Args:
            u_net: Displacement network
            x_colloc: Collocation points (N x 3)
            mu_constant: Constant stiffness value
        
        Returns:
            Mean squared PDE residual
        """
        residual = self.physics.helmholtz_residual_homogeneous(
            u_net, x_colloc, mu_constant
        )
        loss = jnp.mean(jnp.abs(residual)**2)
        return loss
    
    def physics_loss_heterogeneous(
        self,
        u_net: Callable,
        mu_net: Callable,
        x_colloc: jnp.ndarray
    ) -> float:
        """
        Physics loss for heterogeneous case: ||PDE residual||²
        
        Args:
            u_net: Displacement network
            mu_net: Modulus network
            x_colloc: Collocation points (N x 3)
        
        Returns:
            Mean squared PDE residual
        """
        residual = self.physics.helmholtz_residual_heterogeneous(
            u_net, mu_net, x_colloc
        )
        loss = jnp.mean(jnp.abs(residual)**2)
        return loss
    
    @staticmethod
    def regularization_loss(
        mu_net: Callable,
        x: jnp.ndarray,
        regularization_type: str = 'smoothness'
    ) -> float:
        """
        Regularization loss for physically plausible modulus.
        
        Args:
            mu_net: Modulus network
            x: Points for regularization (N x 3)
            regularization_type: 'smoothness' (penalize ∇μ) or 'magnitude' (penalize |μ|)
        
        Returns:
            Regularization loss
        """
        if regularization_type == 'smoothness':
            # Penalize large gradients: ||∇μ||²
            grad_mu = HelmholtzPhysics.gradient_3d(mu_net, x)
            loss = jnp.mean(jnp.sum(grad_mu**2, axis=1))
        elif regularization_type == 'magnitude':
            # Penalize extreme values: ||μ - μ_mean||²
            mu = vmap(lambda xi: mu_net(xi))(x)
            mu_mean = jnp.mean(mu)
            loss = jnp.mean((mu - mu_mean)**2)
        else:
            raise ValueError(f"Unknown regularization type: {regularization_type}")
        
        return loss
    
    def combined_loss(
        self,
        u_net: Callable,
        mu_net: Callable,
        x_data: jnp.ndarray,
        u_measured: jnp.ndarray,
        x_colloc: jnp.ndarray,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.5,
        lambda_reg: float = 0.1,
        heterogeneous: bool = True,
        mu_constant: float = 3000.0
    ) -> Tuple[float, dict]:
        """
        Combined loss function for PIELM training.
        
        Args:
            u_net: Displacement network
            mu_net: Modulus network
            x_data: Data points (N x 3)
            u_measured: Measured displacement (N,)
            x_colloc: Collocation points for physics (M x 3)
            lambda_data: Weight for data loss
            lambda_physics: Weight for physics loss
            lambda_reg: Weight for regularization loss
            heterogeneous: Use heterogeneous (True) or homogeneous (False) Helmholtz
            mu_constant: Constant μ for homogeneous case
        
        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Dictionary with individual loss components
        """
        # Data loss
        u_pred = vmap(lambda xi: u_net(xi))(x_data)
        L_data = self.data_loss(u_pred, u_measured)
        
        # Physics loss
        if heterogeneous:
            L_physics = self.physics_loss_heterogeneous(u_net, mu_net, x_colloc)
        else:
            L_physics = self.physics_loss_homogeneous(u_net, x_colloc, mu_constant)
        
        # Regularization loss
        L_reg = self.regularization_loss(mu_net, x_colloc, regularization_type='smoothness')
        
        # Total loss
        total_loss = lambda_data * L_data + lambda_physics * L_physics + lambda_reg * L_reg
        
        loss_dict = {
            'total': float(total_loss),
            'data': float(L_data),
            'physics': float(L_physics),
            'regularization': float(L_reg)
        }
        
        return total_loss, loss_dict


# Test functions
if __name__ == "__main__":
    print("Testing Physics Module...")
    
    # Initialize physics
    physics = HelmholtzPhysics(frequency=60.0, density=1000.0)
    
    # Test wave properties
    mu_liver = 3000.0  # Typical liver stiffness (Pa)
    wavelength = physics.compute_wavelength(mu_liver)
    print(f"\nFor liver tissue (μ = {mu_liver} Pa):")
    print(f"  Wavelength: {wavelength*1000:.2f} mm")
    print(f"  Wave number: {physics.compute_wave_number(mu_liver):.2f} rad/m")
    
    print("\n✅ Physics module ready for training!")
