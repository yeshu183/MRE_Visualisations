"""
Bernstein Polynomial Basis Functions for 3D PIELM

Based on mre_eigpielm repository implementation.
Provides well-conditioned basis with analytical derivatives.

Mathematical Foundation:
-----------------------
1D Bernstein polynomial of degree n:
    B_{i,n}(t) = C(n,i) * t^i * (1-t)^{n-i}  for t ∈ [0,1]

3D tensor product:
    Φ_{i,j,k}(x,y,z) = B_i(x) * B_j(y) * B_k(z)

Derivatives (recursive):
    dB_{i,n}/dt = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))
    d²B_{i,n}/dt² = n(n-1) * (B_{i-2,n-2} - 2*B_{i-1,n-2} + B_{i,n-2})
"""

import numpy as np
from scipy.special import comb
from typing import Tuple, Optional
import warnings


class BernsteinBasis3D:
    """
    3D Tensor product Bernstein polynomial basis for PIELM.
    
    Provides excellent numerical conditioning (κ ~ 1e2-1e4) compared to
    random tanh features (κ ~ 1e20).
    """
    
    def __init__(self,
                 degrees: Tuple[int, int, int],
                 domain: Tuple[Tuple[float, float], ...],
                 cache_binomials: bool = True):
        """
        Initialize Bernstein basis.
        
        Parameters
        ----------
        degrees : tuple of int
            Polynomial degrees (nx, ny, nz) for each dimension
            Total features = (nx+1) * (ny+1) * (nz+1)
        domain : tuple of tuples
            Physical domain ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        cache_binomials : bool
            Precompute binomial coefficients for speed
        """
        self.nx, self.ny, self.nz = degrees
        self.degrees = degrees
        self.n_features = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        
        # Domain bounds
        (self.x_min, self.x_max), \
        (self.y_min, self.y_max), \
        (self.z_min, self.z_max) = domain
        self.domain = domain
        
        # Precompute binomial coefficients for numerical stability
        if cache_binomials:
            self.binom_x = self._compute_binomials(self.nx)
            self.binom_y = self._compute_binomials(self.ny)
            self.binom_z = self._compute_binomials(self.nz)
        
        print(f"BernsteinBasis3D initialized:")
        print(f"  Degrees: nx={self.nx}, ny={self.ny}, nz={self.nz}")
        print(f"  Total features: {self.n_features}")
        print(f"  Domain: x ∈ [{self.x_min:.4f}, {self.x_max:.4f}]")
        print(f"          y ∈ [{self.y_min:.4f}, {self.y_max:.4f}]")
        print(f"          z ∈ [{self.z_min:.4f}, {self.z_max:.4f}]")
    
    def _compute_binomials(self, n: int) -> np.ndarray:
        """Compute binomial coefficients C(n, i) for i=0..n."""
        return np.array([comb(n, i, exact=True) for i in range(n + 1)])
    
    def _normalize_coords(self, X: np.ndarray) -> np.ndarray:
        """Map physical coordinates to [0, 1]³."""
        t = np.zeros_like(X)
        t[:, 0] = (X[:, 0] - self.x_min) / (self.x_max - self.x_min)
        t[:, 1] = (X[:, 1] - self.y_min) / (self.y_max - self.y_min)
        t[:, 2] = (X[:, 2] - self.z_min) / (self.z_max - self.z_min)
        # Clamp to [0, 1] for numerical safety
        return np.clip(t, 0.0, 1.0)
    
    def _bernstein_1d(self, t: np.ndarray, i: int, n: int, 
                     binom: np.ndarray) -> np.ndarray:
        """
        Evaluate single 1D Bernstein polynomial B_{i,n}(t).
        
        B_{i,n}(t) = C(n,i) * t^i * (1-t)^{n-i}
        """
        # Handle edge cases for numerical stability
        if i == 0:
            return (1.0 - t) ** n
        elif i == n:
            return t ** n
        else:
            # General case
            return binom[i] * (t ** i) * ((1.0 - t) ** (n - i))
    
    def _bernstein_1d_deriv(self, t: np.ndarray, i: int, n: int,
                           binom: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Evaluate derivative of 1D Bernstein polynomial.
        
        First derivative:
            dB_{i,n}/dt = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))
        
        Second derivative:
            d²B_{i,n}/dt² = n(n-1) * (B_{i-2,n-2} - 2*B_{i-1,n-2} + B_{i,n-2})
        """
        if order == 1:
            # First derivative
            if n == 0:
                return np.zeros_like(t)
            
            term1 = np.zeros_like(t)
            term2 = np.zeros_like(t)
            
            if i > 0:
                # Need B_{i-1, n-1}
                binom_lower = self._compute_binomials(n - 1)
                term1 = self._bernstein_1d(t, i - 1, n - 1, binom_lower)
            
            if i < n:
                # Need B_{i, n-1}
                binom_lower = self._compute_binomials(n - 1)
                term2 = self._bernstein_1d(t, i, n - 1, binom_lower)
            
            return n * (term1 - term2)
        
        elif order == 2:
            # Second derivative
            if n <= 1:
                return np.zeros_like(t)
            
            term1 = np.zeros_like(t)
            term2 = np.zeros_like(t)
            term3 = np.zeros_like(t)
            
            binom_lower2 = self._compute_binomials(n - 2)
            
            if i >= 2:
                term1 = self._bernstein_1d(t, i - 2, n - 2, binom_lower2)
            if i >= 1 and i <= n - 1:
                term2 = self._bernstein_1d(t, i - 1, n - 2, binom_lower2)
            if i <= n - 2:
                term3 = self._bernstein_1d(t, i, n - 2, binom_lower2)
            
            return n * (n - 1) * (term1 - 2 * term2 + term3)
        
        else:
            raise ValueError(f"Derivative order {order} not supported")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at given points.
        
        Φ_{i,j,k}(x,y,z) = B_i(x) * B_j(y) * B_k(z)
        
        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Physical coordinates
            
        Returns
        -------
        phi : np.ndarray, shape (N, n_features)
            Basis function values
        """
        N = X.shape[0]
        
        # Normalize to [0, 1]³
        t = self._normalize_coords(X)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        
        # Evaluate all 1D Bernstein polynomials
        B_x = np.zeros((N, self.nx + 1))
        B_y = np.zeros((N, self.ny + 1))
        B_z = np.zeros((N, self.nz + 1))
        
        for i in range(self.nx + 1):
            B_x[:, i] = self._bernstein_1d(tx, i, self.nx, self.binom_x)
        
        for j in range(self.ny + 1):
            B_y[:, j] = self._bernstein_1d(ty, j, self.ny, self.binom_y)
        
        for k in range(self.nz + 1):
            B_z[:, k] = self._bernstein_1d(tz, k, self.nz, self.binom_z)
        
        # Compute tensor product
        phi = np.zeros((N, self.n_features))
        
        idx = 0
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    # Φ_{i,j,k} = B_i(x) * B_j(y) * B_k(z)
                    phi[:, idx] = B_x[:, i] * B_y[:, j] * B_z[:, k]
                    idx += 1
        
        return phi
    
    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇Φ = [∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z].
        
        Product rule:
            ∂Φ_{i,j,k}/∂x = B'_i(x) * B_j(y) * B_k(z) * scale_x
        
        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Physical coordinates
            
        Returns
        -------
        grad_phi : np.ndarray, shape (N, n_features, 3)
            Gradient of each basis function
        """
        N = X.shape[0]
        
        # Normalize coordinates
        t = self._normalize_coords(X)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        
        # Chain rule scaling factors
        scale_x = 1.0 / (self.x_max - self.x_min)
        scale_y = 1.0 / (self.y_max - self.y_min)
        scale_z = 1.0 / (self.z_max - self.z_min)
        
        # Evaluate basis and first derivatives
        B_x = np.zeros((N, self.nx + 1))
        B_y = np.zeros((N, self.ny + 1))
        B_z = np.zeros((N, self.nz + 1))
        
        dB_x = np.zeros((N, self.nx + 1))
        dB_y = np.zeros((N, self.ny + 1))
        dB_z = np.zeros((N, self.nz + 1))
        
        for i in range(self.nx + 1):
            B_x[:, i] = self._bernstein_1d(tx, i, self.nx, self.binom_x)
            dB_x[:, i] = self._bernstein_1d_deriv(tx, i, self.nx, self.binom_x, order=1) * scale_x
        
        for j in range(self.ny + 1):
            B_y[:, j] = self._bernstein_1d(ty, j, self.ny, self.binom_y)
            dB_y[:, j] = self._bernstein_1d_deriv(ty, j, self.ny, self.binom_y, order=1) * scale_y
        
        for k in range(self.nz + 1):
            B_z[:, k] = self._bernstein_1d(tz, k, self.nz, self.binom_z)
            dB_z[:, k] = self._bernstein_1d_deriv(tz, k, self.nz, self.binom_z, order=1) * scale_z
        
        # Compute gradients via product rule
        grad_phi = np.zeros((N, self.n_features, 3))
        
        idx = 0
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    # ∂Φ/∂x = B'_i * B_j * B_k
                    grad_phi[:, idx, 0] = dB_x[:, i] * B_y[:, j] * B_z[:, k]
                    # ∂Φ/∂y = B_i * B'_j * B_k
                    grad_phi[:, idx, 1] = B_x[:, i] * dB_y[:, j] * B_z[:, k]
                    # ∂Φ/∂z = B_i * B_j * B'_k
                    grad_phi[:, idx, 2] = B_x[:, i] * B_y[:, j] * dB_z[:, k]
                    idx += 1
        
        return grad_phi
    
    def laplacian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²Φ = ∂²Φ/∂x² + ∂²Φ/∂y² + ∂²Φ/∂z².
        
        ∇²Φ_{i,j,k} = B''_i*B_j*B_k + B_i*B''_j*B_k + B_i*B_j*B''_k
        
        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Physical coordinates
            
        Returns
        -------
        lap_phi : np.ndarray, shape (N, n_features)
            Laplacian of each basis function
        """
        N = X.shape[0]
        
        # Normalize coordinates
        t = self._normalize_coords(X)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        
        # Chain rule scaling for second derivatives
        scale_xx = 1.0 / (self.x_max - self.x_min) ** 2
        scale_yy = 1.0 / (self.y_max - self.y_min) ** 2
        scale_zz = 1.0 / (self.z_max - self.z_min) ** 2
        
        # Evaluate basis and second derivatives
        B_x = np.zeros((N, self.nx + 1))
        B_y = np.zeros((N, self.ny + 1))
        B_z = np.zeros((N, self.nz + 1))
        
        d2B_x = np.zeros((N, self.nx + 1))
        d2B_y = np.zeros((N, self.ny + 1))
        d2B_z = np.zeros((N, self.nz + 1))
        
        for i in range(self.nx + 1):
            B_x[:, i] = self._bernstein_1d(tx, i, self.nx, self.binom_x)
            d2B_x[:, i] = self._bernstein_1d_deriv(tx, i, self.nx, self.binom_x, order=2) * scale_xx
        
        for j in range(self.ny + 1):
            B_y[:, j] = self._bernstein_1d(ty, j, self.ny, self.binom_y)
            d2B_y[:, j] = self._bernstein_1d_deriv(ty, j, self.ny, self.binom_y, order=2) * scale_yy
        
        for k in range(self.nz + 1):
            B_z[:, k] = self._bernstein_1d(tz, k, self.nz, self.binom_z)
            d2B_z[:, k] = self._bernstein_1d_deriv(tz, k, self.nz, self.binom_z, order=2) * scale_zz
        
        # Compute Laplacian
        lap_phi = np.zeros((N, self.n_features))
        
        idx = 0
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    # ∇²Φ = B''_i*B_j*B_k + B_i*B''_j*B_k + B_i*B_j*B''_k
                    lap_phi[:, idx] = (d2B_x[:, i] * B_y[:, j] * B_z[:, k] +
                                       B_x[:, i] * d2B_y[:, j] * B_z[:, k] +
                                       B_x[:, i] * B_y[:, j] * d2B_z[:, k])
                    idx += 1
        
        return lap_phi


def verify_bernstein_derivatives(degrees=(5, 5, 4), 
                                 domain=((0, 1), (0, 1), (0, 1)),
                                 h=1e-5) -> dict:
    """
    Verify Bernstein derivatives against finite differences.
    
    Returns
    -------
    results : dict
        Gradient and Laplacian errors
    """
    basis = BernsteinBasis3D(degrees, domain)
    
    # Test points (avoid boundaries)
    np.random.seed(42)
    X = np.random.rand(10, 3) * 0.6 + 0.2
    X[:, 0] *= (domain[0][1] - domain[0][0])
    X[:, 1] *= (domain[1][1] - domain[1][0])
    X[:, 2] *= (domain[2][1] - domain[2][0])
    
    # Analytical
    grad_analytical = basis.gradient(X)
    lap_analytical = basis.laplacian(X)
    
    # Finite difference for gradient (x-direction)
    X_plus = X.copy()
    X_plus[:, 0] += h
    phi_plus = basis.forward(X_plus)
    
    X_minus = X.copy()
    X_minus[:, 0] -= h
    phi_minus = basis.forward(X_minus)
    
    grad_fd_x = (phi_plus - phi_minus) / (2 * h)
    grad_error = np.max(np.abs(grad_analytical[:, :, 0] - grad_fd_x))
    
    # Finite difference for Laplacian (x-direction component)
    phi_center = basis.forward(X)
    lap_fd_x = (phi_plus - 2 * phi_center + phi_minus) / h ** 2
    
    # Approximate Laplacian error (just x-component)
    lap_error_approx = np.max(np.abs(lap_fd_x - lap_analytical))
    
    return {
        'gradient_error': grad_error,
        'laplacian_error': lap_error_approx
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Bernstein Basis - Derivative Verification")
    print("="*70)
    
    results = verify_bernstein_derivatives()
    
    print(f"\nGradient error: {results['gradient_error']:.6e}")
    print(f"Laplacian error: {results['laplacian_error']:.6e}")
    
    if results['gradient_error'] < 1e-4 and results['laplacian_error'] < 1e-3:
        print("\n✓ Derivative verification PASSED")
    else:
        print("\n✗ Derivative verification FAILED")
    
    print("="*70)
