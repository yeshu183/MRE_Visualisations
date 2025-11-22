"""
MRE Trainer
===========

Training loop for MRE stiffness reconstruction with:
- Data fitting loss
- Total variation regularization
- Learning rate scheduling
- Early stopping
- Progress visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List

try:
    from .forward_model import ForwardMREModel
except ImportError:
    from forward_model import ForwardMREModel


class MRETrainer:
    """Trainer for MRE inverse problem."""

    def __init__(
        self,
        model: ForwardMREModel,
        device: torch.device,
        output_dir: str = "./outputs",
        loss_type: str = 'correlation'
    ):
        """Initialize trainer.

        Args:
            model: ForwardMREModel instance
            device: Torch device
            output_dir: Directory for saving results
            loss_type: Loss function type
                - 'correlation': Cosine similarity (phase/shape)
                - 'relative_l2': Normalized L2 loss
                - 'sobolev': Gradient-enhanced (most sensitive)
                - 'mse': Standard mean squared error
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_type = loss_type

        # Training history
        self.history = {
            'data_loss': [],
            'pde_loss': [],
            'tv_loss': [],
            'total_loss': [],
            'grad_norm': [],
            'mu_mse': [],
            'mu_min': [],
            'mu_max': [],
        }

    def _compute_pde_residual(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mu: torch.Tensor,
        rho_omega2: float
    ) -> torch.Tensor:
        """Compute NORMALIZED PDE residual: ∇·(μ∇u) + ρω²u using finite differences.

        Physics equation: ∇·(μ∇u) + ρω²u = 0

        NORMALIZATION: Divide by rho_omega2 to prevent scale mismatch.
        Normalized residual: (μ/ρω²)∇²u + u

        Args:
            x: (N, dim) coordinates
            u: (N, 1) displacement
            mu: (N, 1) stiffness
            rho_omega2: Physics parameter

        Returns:
            Normalized PDE residual loss
        """
        N = x.shape[0]
        if N <= 10:
            return torch.tensor(0.0, device=u.device, requires_grad=True)

        # Central differences for Laplacian
        u_f = u[2:]    # forward
        u_c = u[1:-1]  # center
        u_b = u[:-2]   # backward

        # Spatial spacing
        dx = torch.norm(x[2:] - x[:-2], dim=1, keepdim=True) / 2.0 + 1e-8

        # Laplacian approximation
        laplacian_u = (u_f - 2 * u_c + u_b) / (dx ** 2)

        # Get corresponding mu values
        mu_c = mu[1:-1]

        # NORMALIZED PDE residual
        residual_normalized = (mu_c / rho_omega2) * laplacian_u + u_c
        loss_pde = torch.mean(residual_normalized ** 2)

        return loss_pde

    def _compute_sobolev_loss(
        self,
        u_pred: torch.Tensor,
        u_meas: torch.Tensor,
        x: torch.Tensor,
        alpha: float = 0.1,
        beta: float = 0.9
    ) -> torch.Tensor:
        """Compute Sobolev loss with gradient terms for backpropagation.
        
        L_Sobolev = α||u_pred - u_meas||² + β||∇u_pred - ∇u_meas||²
        
        MATHEMATICAL DERIVATION (see approach/docs/SOBOLEV_LOSS_DERIVATION.md):
        
        1. Forward Pass:
           u_pred = Φ·C  (basis expansion)
           ∇u_pred = Ψ·C  (basis derivatives)
        
        2. Loss Gradient w.r.t. Coefficients:
           ∂L/∂C = α·Φ^T(u_pred - u_meas) + β·Ψ^T(∇u_pred - ∇u_meas)
        
        3. Backprop Through Solver (unchanged from MSE!):
           ∂L/∂H = -(H·v·C^T + r·v^T)
           where v = (H^T·H)^{-1}·∂L/∂C
        
        4. Full Gradient Chain:
           ∂L/∂μ = ∂L/∂u·∂u/∂C·∂C/∂H·∂H/∂μ
                 + ∂L/∂∇u·∂∇u/∂C·∂C/∂H·∂H/∂μ
        
        KEY INSIGHT: Gradient term (β=0.9) provides 90% of discrimination power
        because wave scattering at stiffness boundaries changes slopes more than
        amplitudes. From forward solve analysis, α=0.1, β=0.9 is optimal.
        
        Args:
            u_pred: Predicted displacement (N, 1) - requires_grad=True for backprop
            u_meas: Measured displacement (N, 1)
            x: Spatial coordinates (N, 3)
            alpha: Weight for L2 term (default 0.1, optimal from analysis)
            beta: Weight for gradient term (default 0.9, optimal from analysis)
        
        Returns:
            Sobolev loss value (scalar tensor with grad_fn for backprop)
        """
        # L2 term: α·||u_pred - u_meas||²
        loss_l2 = torch.mean((u_pred - u_meas) ** 2)
        
        # Gradient term: β·||∇u_pred - ∇u_meas||²
        # Using finite differences (fully differentiable via autograd)
        N = x.shape[0]
        
        if N > 10:
            # Forward finite differences: ∇u ≈ Δu / Δx
            # PyTorch autograd will backprop through these operations
            du_pred = u_pred[1:] - u_pred[:-1]  # Differentiable w.r.t. u_pred
            du_meas = u_meas[1:] - u_meas[:-1]
            
            # Spatial spacing (detach to avoid propagating through coordinates)
            dx = torch.norm(x[1:] - x[:-1], dim=1, keepdim=True) + 1e-8
            
            # Normalized gradients
            grad_pred = du_pred / dx  # Gradient flows through du_pred
            grad_meas = du_meas / dx
            
            # Gradient difference loss
            loss_grad = torch.mean((grad_pred - grad_meas) ** 2)
        else:
            loss_grad = torch.tensor(0.0, device=u_pred.device, requires_grad=True)
        
        # Combined Sobolev loss (fully differentiable)
        # α=0.1 (10% L2) + β=0.9 (90% gradient) = optimal discrimination
        return alpha * loss_l2 + beta * loss_grad

    def train(
        self,
        x: torch.Tensor,
        u_meas: torch.Tensor,
        mu_true: torch.Tensor,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        rho_omega2: float,
        iterations: int = 3000,
        lr: float = 0.005,
        bc_weight: float = 200.0,
        data_weight: float = 0.0,
        tv_weight: float = 0.001,
        pde_weight: float = 0.0,
        prior_weight: float = 0.0,
        barrier_weight: float = 0.0,
        grad_clip: float = 1.0,
        early_stopping: int = 1500,
        log_interval: int = 500,
        save_interval: int = 1000
    ) -> Dict:
        """
        Train stiffness reconstruction.

        Args:
            x: (N, dim) coordinates
            u_meas: (N, 1) measured displacement
            mu_true: (N, 1) ground truth stiffness (for monitoring)
            bc_indices: (K,) boundary indices
            u_bc_vals: (K, 1) boundary values
            rho_omega2: Physics parameter
            iterations: Max training iterations
            lr: Learning rate (Adam handles adaptive LR)
            bc_weight: BC enforcement weight (200 recommended)
            data_weight: Data constraint weight (0 for BC-only)
            tv_weight: Total variation regularization
            pde_weight: PDE residual loss weight
            grad_clip: Gradient clipping threshold
            early_stopping: Patience for early stopping
            log_interval: Iterations between log prints
            save_interval: Iterations between saving plots

        Returns:
            Dictionary with final results
        """
        print("=" * 60)
        print("MRE Training")
        print("=" * 60)
        print(f"  Points: {len(x)}")
        print(f"  BC points: {len(bc_indices)}")
        print(f"  Iterations: {iterations}")
        print(f"  LR: {lr} (Adam optimizer, no decay)")
        print(f"  Weights: bc={bc_weight}, data={data_weight}, tv={tv_weight}, pde={pde_weight}")
        if prior_weight > 0 or barrier_weight > 0:
            print(f"  Regularization: prior={prior_weight}, barrier={barrier_weight}")
        print("=" * 60)

        # Optimizer (no scheduler - Adam handles adaptive LR)
        optimizer = torch.optim.Adam(self.model.mu_net.parameters(), lr=lr)

        # Early stopping
        best_loss = float('inf')
        patience_counter = 0

        # Training loop
        for i in range(iterations):
            optimizer.zero_grad()

            # Forward pass
            u_pred, mu_pred = self.model(
                x, bc_indices, u_bc_vals, rho_omega2,
                bc_weight=bc_weight,
                u_data=u_meas,
                data_weight=data_weight,
                verbose=(i == 0)
            )

            # --- COMPUTE DATA LOSS (Multiple Options) ---
            u_p_flat = u_pred.view(-1)
            u_m_flat = u_meas.view(-1)
            
            # Always compute all metrics for comparison
            loss_mse = torch.mean((u_pred - u_meas) ** 2)
            
            # Correlation (cosine similarity)
            u_p_norm = u_p_flat / (torch.norm(u_p_flat) + 1e-8)
            u_m_norm = u_m_flat / (torch.norm(u_m_flat) + 1e-8)
            correlation = torch.dot(u_p_norm, u_m_norm)
            loss_correlation = 1.0 - correlation
            
            # Relative L2 (normalized MSE)
            loss_relative_l2 = torch.sqrt(loss_mse) / (torch.norm(u_m_flat) + 1e-8)
            
            # Sobolev loss (gradient-enhanced for better sensitivity)
            loss_sobolev = self._compute_sobolev_loss(u_pred, u_meas, x)

            # PDE Residual Loss
            loss_pde = self._compute_pde_residual(x, u_pred, mu_pred, rho_omega2)

            # Select loss based on configuration
            if self.loss_type == 'correlation':
                loss_data = loss_correlation
            elif self.loss_type == 'relative_l2':
                loss_data = loss_relative_l2
            elif self.loss_type == 'sobolev':
                loss_data = loss_sobolev
            elif self.loss_type == 'mse':
                loss_data = loss_mse
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # TV regularization (promotes smooth/piecewise-constant)
            # Sort by first coordinate for meaningful TV
            if self.model.input_dim == 1:
                loss_tv = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))
            else:
                # For 3D, use neighbor differences (simplified)
                loss_tv = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))

            # Prior regularization (pull predictions toward reasonable mean)
            if prior_weight > 0:
                mu_mean = (self.model.mu_net.mu_min + self.model.mu_net.mu_max) / 2
                mu_std_target = (self.model.mu_net.mu_max - self.model.mu_net.mu_min) / 6
                loss_prior = torch.mean(((mu_pred - mu_mean) / mu_std_target) ** 2)
            else:
                loss_prior = torch.tensor(0.0, device=mu_pred.device)

            # Barrier loss (penalize hitting bounds to prevent saturation)
            if barrier_weight > 0:
                mu_normalized = (mu_pred - self.model.mu_net.mu_min) / (self.model.mu_net.mu_max - self.model.mu_net.mu_min)
                epsilon = 0.05  # Stay away from [0, 1] boundaries
                # Clamp to valid range to avoid log(negative)
                mu_normalized = torch.clamp(mu_normalized, epsilon, 1.0 - epsilon)
                loss_barrier = -torch.mean(torch.log(mu_normalized - epsilon + 1e-8) +
                                          torch.log(1.0 - epsilon - mu_normalized + 1e-8))
            else:
                loss_barrier = torch.tensor(0.0, device=mu_pred.device)

            # Total loss with all regularization terms
            loss_total = (loss_data + tv_weight * loss_tv + pde_weight * loss_pde +
                         prior_weight * loss_prior + barrier_weight * loss_barrier)

            # Backward pass
            loss_total.backward()

            # Gradient clipping (optional)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.mu_net.parameters(), max_norm=grad_clip
                )

            optimizer.step()

            # Compute metrics
            with torch.no_grad():
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.model.mu_net.parameters()
                    if p.grad is not None
                ) ** 0.5
                mu_mse = torch.mean((mu_pred - mu_true) ** 2).item()

            # Record history - track all metrics
            self.history['data_loss'].append(loss_data.item())
            self.history['pde_loss'].append(loss_pde.item())
            self.history['mse_loss'] = self.history.get('mse_loss', [])
            self.history['mse_loss'].append(loss_mse.item())
            self.history['correlation'] = self.history.get('correlation', [])
            self.history['correlation'].append(correlation.item())
            self.history['relative_l2'] = self.history.get('relative_l2', [])
            self.history['relative_l2'].append(loss_relative_l2.item())
            self.history['sobolev'] = self.history.get('sobolev', [])
            self.history['sobolev'].append(loss_sobolev.item())
            self.history['tv_loss'].append(loss_tv.item())
            self.history['prior_loss'] = self.history.get('prior_loss', [])
            self.history['prior_loss'].append(loss_prior.item() if isinstance(loss_prior, torch.Tensor) else loss_prior)
            self.history['barrier_loss'] = self.history.get('barrier_loss', [])
            self.history['barrier_loss'].append(loss_barrier.item() if isinstance(loss_barrier, torch.Tensor) else loss_barrier)
            self.history['total_loss'].append(loss_total.item())
            self.history['grad_norm'].append(grad_norm)
            self.history['mu_mse'].append(mu_mse)
            self.history['mu_min'].append(mu_pred.min().item())
            self.history['mu_max'].append(mu_pred.max().item())

            # Logging - adaptive interval (every 10 iters for first 100, then every 100)
            if i < 100:
                current_log_interval = 10  # Frequent logging to monitor speed
            elif i < 500:
                current_log_interval = 100
            else:
                current_log_interval = log_interval

            if i % current_log_interval == 0 or i == iterations - 1:
                print(f"Iter {i:5d}: {self.loss_type.upper()}={loss_data.item():.4e}, "
                      f"PDE={loss_pde.item():.4e}, MuMSE={mu_mse:.4e}")
                print(f"           mu=[{mu_pred.min().item():.0f}, "
                      f"{mu_pred.max().item():.0f}] Pa, "
                      f"true=[{mu_true.min().item():.0f}, {mu_true.max().item():.0f}] Pa")

            # Save progress plots
            if save_interval > 0 and i > 0 and i % save_interval == 0:
                self._save_progress_plot(i, x, mu_pred, mu_true, u_pred, u_meas)

            # Early stopping
            if loss_data.item() < best_loss:
                best_loss = loss_data.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"\nEarly stopping at iteration {i}")
                    break

        # Final evaluation
        with torch.no_grad():
            u_pred, mu_pred = self.model(
                x, bc_indices, u_bc_vals, rho_omega2,
                bc_weight=bc_weight
            )

        results = {
            'u_pred': u_pred,
            'mu_pred': mu_pred,
            'final_loss': self.history['data_loss'][-1],
            'final_mu_mse': self.history['mu_mse'][-1],
            'history': self.history
        }

        # Save final results
        self._save_final_results(x, mu_pred, mu_true, u_pred, u_meas)

        return results

    def _save_progress_plot(
        self,
        iteration: int,
        x: torch.Tensor,
        mu_pred: torch.Tensor,
        mu_true: torch.Tensor,
        u_pred: torch.Tensor,
        u_meas: torch.Tensor
    ):
        """Save progress visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Loss curves
        ax = axes[0, 0]
        ax.semilogy(self.history['data_loss'], label=f'{self.loss_type.upper()}', linewidth=2)
        if 'mse_loss' in self.history:
            ax.semilogy(self.history['mse_loss'], label='MSE', linewidth=2, alpha=0.7)
        ax.semilogy(self.history['mu_mse'], label='Mu MSE', linewidth=2, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training Loss')
        ax.grid(True)

        # Correlation over time
        ax = axes[0, 1]
        if 'correlation' in self.history:
            ax.plot(self.history['correlation'], linewidth=2)
            ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect (1.0)')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Correlation')
            ax.set_title('Shape Match (Correlation)')
            ax.set_ylim([-0.1, 1.1])
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Correlation\nNot Available', ha='center', va='center')
            ax.axis('off')

        # Gradient norm
        ax = axes[0, 2]
        ax.plot(self.history['grad_norm'], linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True)

        # Mu comparison (scatter) - FIXED: Show in Pa units
        ax = axes[1, 0]
        mu_p = mu_pred.detach().cpu().numpy().flatten()
        mu_t = mu_true.detach().cpu().numpy().flatten()
        ax.scatter(mu_t, mu_p, alpha=0.3, s=1)
        lims = [min(mu_t.min(), mu_p.min()), max(mu_t.max(), mu_p.max())]
        ax.plot(lims, lims, 'r--', label='Perfect')
        ax.set_xlabel('True μ (Pa)')
        ax.set_ylabel('Predicted μ (Pa)')
        ax.set_title(f'Stiffness (iter {iteration})\nRange: [{mu_p.min():.0f}, {mu_p.max():.0f}]')
        ax.legend()
        ax.grid(True)

        # Mu histogram
        ax = axes[1, 1]
        ax.hist(mu_t, bins=30, alpha=0.5, label='True', density=True, color='blue')
        ax.hist(mu_p, bins=30, alpha=0.5, label='Predicted', density=True, color='orange')
        ax.set_xlabel('μ (Pa)')
        ax.set_ylabel('Density')
        ax.set_title('Stiffness Distribution')
        ax.legend()
        ax.grid(True)

        # Displacement comparison
        ax = axes[1, 2]
        u_p = u_pred.detach().cpu().numpy().flatten()
        u_m = u_meas.detach().cpu().numpy().flatten()
        ax.scatter(u_m, u_p, alpha=0.3, s=1)
        ax.plot([u_m.min(), u_m.max()], [u_m.min(), u_m.max()], 'r--')
        ax.set_xlabel('Measured u (m)')
        ax.set_ylabel('Predicted u (m)')
        ax.set_title('Displacement Fit')
        ax.grid(True)

        # Row 3: Spatial μ visualizations (Ground Truth, Predicted, Error)
        x_np = x.detach().cpu().numpy()
        if x.shape[1] == 3:  # 3D data
            # Find middle z-slice for 2D visualization
            z_coords = x_np[:, 2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            z_tolerance = 0.01  # 1cm tolerance for mid-slice
            mid_slice_mask = np.abs(z_coords - z_mid) < z_tolerance

            if mid_slice_mask.sum() > 10:  # Ensure enough points
                x_slice = x_np[mid_slice_mask, 0]
                y_slice = x_np[mid_slice_mask, 1]
                mu_t_slice = mu_t[mid_slice_mask]
                mu_p_slice = mu_p[mid_slice_mask]
                mu_e_slice = mu_p_slice - mu_t_slice

                # Shared colorbar limits for true and predicted
                vmin_mu = mu_t.min()
                vmax_mu = mu_t.max()

                # Plot 1: Ground Truth μ(x,y)
                ax = axes[2, 0]
                sc = ax.scatter(x_slice, y_slice, c=mu_t_slice, s=25, cmap='jet',
                               vmin=vmin_mu, vmax=vmax_mu, edgecolors='none')
                plt.colorbar(sc, ax=ax, label='μ (Pa)', fraction=0.046)
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                ax.set_title(f'Ground Truth μ(x,y) [Z≈{z_mid:.3f}m]', fontsize=10, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # Plot 2: Predicted μ(x,y)
                ax = axes[2, 1]
                sc = ax.scatter(x_slice, y_slice, c=mu_p_slice, s=25, cmap='jet',
                               vmin=vmin_mu, vmax=vmax_mu, edgecolors='none')
                plt.colorbar(sc, ax=ax, label='μ (Pa)', fraction=0.046)
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                ax.set_title(f'Predicted μ(x,y) [Iter {iteration}]', fontsize=10, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # Plot 3: Error μ(x,y)
                ax = axes[2, 2]
                error_max = max(abs(mu_e_slice.min()), abs(mu_e_slice.max()))
                if error_max > 1e-6:  # Avoid division by zero
                    sc = ax.scatter(x_slice, y_slice, c=mu_e_slice, s=25, cmap='RdBu_r',
                                   vmin=-error_max, vmax=error_max, edgecolors='none')
                    plt.colorbar(sc, ax=ax, label='Error (Pa)', fraction=0.046)
                    rmse_slice = np.sqrt(np.mean(mu_e_slice**2))
                    ax.set_title(f'Error = Pred - True [RMSE: {rmse_slice:.0f} Pa]',
                                fontsize=10, fontweight='bold')
                else:
                    ax.scatter(x_slice, y_slice, c=mu_e_slice, s=25, cmap='RdBu_r', edgecolors='none')
                    ax.set_title('Error = Pred - True [Zero Error]', fontsize=10, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
            else:
                # Not enough points in slice, show placeholder
                for col_idx in range(3):
                    axes[2, col_idx].text(0.5, 0.5, 'Spatial Plot\nNot Available\n(Insufficient points in Z-slice)',
                                         ha='center', va='center', transform=axes[2, col_idx].transAxes)
                    axes[2, col_idx].axis('off')
        else:
            # 1D or 2D data - show placeholder
            for col_idx in range(3):
                axes[2, col_idx].text(0.5, 0.5, 'Spatial μ(x,y)\nOnly for 3D data',
                                     ha='center', va='center', transform=axes[2, col_idx].transAxes)
                axes[2, col_idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'progress_iter_{iteration:05d}.png', dpi=150)
        plt.close()

    def _save_final_results(
        self,
        x: torch.Tensor,
        mu_pred: torch.Tensor,
        mu_true: torch.Tensor,
        u_pred: torch.Tensor,
        u_meas: torch.Tensor
    ):
        """Save final results visualization."""
        # Create comprehensive visualization with mu distributions
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Row 1: Training curves
        ax = fig.add_subplot(gs[0, 0])
        ax.semilogy(self.history['data_loss'], label='Data Loss')
        ax.semilogy(self.history['total_loss'], label='Total Loss', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training Loss')
        ax.grid(True)

        ax = fig.add_subplot(gs[0, 1])
        ax.semilogy(self.history['mu_mse'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title('Stiffness MSE')
        ax.grid(True)

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(self.history['grad_norm'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True)
        
        # Stiffness range evolution
        ax = fig.add_subplot(gs[0, 3])
        mu_p_np = mu_pred.detach().cpu().numpy().flatten()
        mu_t_np = mu_true.detach().cpu().numpy().flatten()
        ax.fill_between(
            range(len(self.history['mu_min'])),
            self.history['mu_min'],
            self.history['mu_max'],
            alpha=0.3, label='Predicted range'
        )
        ax.axhline(mu_t_np.min(), color='r', linestyle='--', label=f'True min: {mu_t_np.min():.0f}')
        ax.axhline(mu_t_np.max(), color='r', linestyle='--', label=f'True max: {mu_t_np.max():.0f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('μ (Pa)')
        ax.set_title('Stiffness Range Evolution')
        ax.legend(fontsize=8)
        ax.grid(True)

        # Row 2: Mu distributions and histograms
        ax = fig.add_subplot(gs[1, 0])
        ax.scatter(mu_t_np, mu_p_np, alpha=0.3, s=1)
        lims = [min(mu_t_np.min(), mu_p_np.min()), max(mu_t_np.max(), mu_p_np.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_xlabel('True μ (Pa)')
        ax.set_ylabel('Predicted μ (Pa)')
        ax.set_title(f'Stiffness Reconstruction\nMSE={self.history["mu_mse"][-1]:.2e}')
        ax.legend()
        ax.grid(True)

        # Mu histogram comparison
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(mu_t_np, bins=50, alpha=0.5, label='True', density=True, color='blue')
        ax.hist(mu_p_np, bins=50, alpha=0.5, label='Predicted', density=True, color='orange')
        ax.set_xlabel('μ (Pa)')
        ax.set_ylabel('Density')
        ax.set_title('Stiffness Distribution')
        ax.legend()
        ax.grid(True)

        # Mu error distribution
        ax = fig.add_subplot(gs[1, 2])
        mu_error = mu_p_np - mu_t_np
        ax.hist(mu_error, bins=50, alpha=0.7, color='red')
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('μ Error (Pa)')
        ax.set_ylabel('Count')
        ax.set_title(f'Error Distribution\nMean: {mu_error.mean():.1f} Pa, Std: {mu_error.std():.1f} Pa')
        ax.grid(True)

        # Mu spatial distribution (if 3D, show mid-slice)
        ax = fig.add_subplot(gs[1, 3])
        x_np = x.detach().cpu().numpy()
        if x.shape[1] == 3:  # 3D data
            # Find middle z-slice
            z_coords = x_np[:, 2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            mid_slice_mask = np.abs(z_coords - z_mid) < 0.01
            
            if mid_slice_mask.sum() > 0:
                sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1], 
                               c=mu_p_np[mid_slice_mask], s=20, cmap='jet', 
                               vmin=mu_t_np.min(), vmax=mu_t_np.max())
                plt.colorbar(sc, ax=ax, label='μ (Pa)')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('Predicted μ (Mid Z-slice)')
                ax.set_aspect('equal')
        else:
            ax.plot(x_np[:, 0], mu_p_np, 'o-', markersize=2, label='Predicted')
            ax.plot(x_np[:, 0], mu_t_np, 'r--', linewidth=2, label='True')
            ax.set_xlabel('Position')
            ax.set_ylabel('μ (Pa)')
            ax.set_title('Stiffness Profile')
            ax.legend()
        ax.grid(True)

        # Row 3: Displacement comparisons
        ax = fig.add_subplot(gs[2, 0])
        u_p = u_pred.detach().cpu().numpy().flatten()
        u_m = u_meas.detach().cpu().numpy().flatten()
        ax.scatter(u_m, u_p, alpha=0.3, s=1)
        lims = [min(u_m.min(), u_p.min()), max(u_m.max(), u_p.max())]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlabel('Measured u (m)')
        ax.set_ylabel('Predicted u (m)')
        ax.set_title('Displacement Fit')
        ax.grid(True)

        # Displacement error
        ax = fig.add_subplot(gs[2, 1])
        u_error = u_p - u_m
        ax.hist(u_error, bins=50, alpha=0.7, color='purple')
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('u Error (m)')
        ax.set_ylabel('Count')
        mse = np.mean(u_error**2)
        ax.set_title(f'Displacement Error\nMSE: {mse:.2e} m²')
        ax.grid(True)

        # All metrics comparison
        ax = fig.add_subplot(gs[2, 2])
        metrics_names = []
        metrics_values = []
        if 'correlation' in self.history:
            metrics_names.append('Correlation')
            metrics_values.append(self.history['correlation'][-1])
        if 'relative_l2' in self.history:
            metrics_names.append('Rel-L2')
            metrics_values.append(self.history['relative_l2'][-1])
        if 'mse_loss' in self.history:
            metrics_names.append('MSE')
            metrics_values.append(self.history['mse_loss'][-1])
        if 'sobolev' in self.history:
            metrics_names.append('Sobolev')
            metrics_values.append(self.history['sobolev'][-1])
        
        if metrics_names:
            ax.bar(metrics_names, metrics_values, alpha=0.7)
            ax.set_ylabel('Final Value')
            ax.set_title('Loss Metrics Comparison')
            ax.grid(True, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Relative error by spatial location
        ax = fig.add_subplot(gs[2, 3])
        if x.shape[1] == 3:
            rel_error = np.abs(mu_error) / (mu_t_np + 1e-8)
            z_coords = x_np[:, 2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            mid_slice_mask = np.abs(z_coords - z_mid) < 0.01

            if mid_slice_mask.sum() > 0:
                sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                               c=rel_error[mid_slice_mask], s=20, cmap='hot',
                               vmin=0, vmax=np.percentile(rel_error, 95))
                plt.colorbar(sc, ax=ax, label='Relative Error')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('μ Relative Error (Mid Z-slice)')
                ax.set_aspect('equal')
        ax.grid(True)

        # Row 4: Dedicated spatial μ visualizations (Ground Truth, Predicted, Error)
        if x.shape[1] == 3:  # 3D data
            # Find middle z-slice for 2D visualization
            z_coords = x_np[:, 2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            z_tolerance = 0.01  # 1cm tolerance for mid-slice
            mid_slice_mask = np.abs(z_coords - z_mid) < z_tolerance

            if mid_slice_mask.sum() > 10:  # Ensure enough points
                x_slice = x_np[mid_slice_mask, 0]
                y_slice = x_np[mid_slice_mask, 1]
                mu_t_slice = mu_t_np[mid_slice_mask]
                mu_p_slice = mu_p_np[mid_slice_mask]
                mu_e_slice = mu_p_slice - mu_t_slice

                # Shared colorbar limits for true and predicted
                vmin_mu = mu_t_np.min()
                vmax_mu = mu_t_np.max()

                # Plot 1: Ground Truth μ(x,y)
                ax = fig.add_subplot(gs[3, 0])
                sc = ax.scatter(x_slice, y_slice, c=mu_t_slice, s=30, cmap='jet',
                               vmin=vmin_mu, vmax=vmax_mu, edgecolors='none')
                cbar = plt.colorbar(sc, ax=ax, label='μ (Pa)', fraction=0.046)
                ax.set_xlabel('X (m)', fontsize=10)
                ax.set_ylabel('Y (m)', fontsize=10)
                ax.set_title(f'Ground Truth μ(x,y)\nZ ≈ {z_mid:.3f} m', fontsize=11, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # Plot 2: Predicted μ(x,y)
                ax = fig.add_subplot(gs[3, 1])
                sc = ax.scatter(x_slice, y_slice, c=mu_p_slice, s=30, cmap='jet',
                               vmin=vmin_mu, vmax=vmax_mu, edgecolors='none')
                cbar = plt.colorbar(sc, ax=ax, label='μ (Pa)', fraction=0.046)
                ax.set_xlabel('X (m)', fontsize=10)
                ax.set_ylabel('Y (m)', fontsize=10)
                ax.set_title(f'Predicted μ(x,y)\nZ ≈ {z_mid:.3f} m', fontsize=11, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # Plot 3: Error μ(x,y)
                ax = fig.add_subplot(gs[3, 2])
                error_max = max(abs(mu_e_slice.min()), abs(mu_e_slice.max()))
                sc = ax.scatter(x_slice, y_slice, c=mu_e_slice, s=30, cmap='RdBu_r',
                               vmin=-error_max, vmax=error_max, edgecolors='none')
                cbar = plt.colorbar(sc, ax=ax, label='Error (Pa)', fraction=0.046)
                ax.set_xlabel('X (m)', fontsize=10)
                ax.set_ylabel('Y (m)', fontsize=10)
                rmse_slice = np.sqrt(np.mean(mu_e_slice**2))
                ax.set_title(f'Error μ(x,y) = Pred - True\nRMSE: {rmse_slice:.1f} Pa',
                            fontsize=11, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # Plot 4: Statistics summary
                ax = fig.add_subplot(gs[3, 3])
                ax.axis('off')
                stats_text = f"""
SPATIAL STATISTICS (Mid Z-slice)

Ground Truth μ:
  Min:  {mu_t_slice.min():.1f} Pa
  Max:  {mu_t_slice.max():.1f} Pa
  Mean: {mu_t_slice.mean():.1f} Pa
  Std:  {mu_t_slice.std():.1f} Pa

Predicted μ:
  Min:  {mu_p_slice.min():.1f} Pa
  Max:  {mu_p_slice.max():.1f} Pa
  Mean: {mu_p_slice.mean():.1f} Pa
  Std:  {mu_p_slice.std():.1f} Pa

Error Metrics:
  RMSE:     {rmse_slice:.1f} Pa
  Mean Err: {mu_e_slice.mean():.1f} Pa
  Max |Err|: {error_max:.1f} Pa
  Rel RMSE: {rmse_slice/mu_t_slice.mean()*100:.1f}%

Points in slice: {mid_slice_mask.sum()}
"""
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_results.png', dpi=150)
        plt.close()

        # Save history (both npy and JSON for easy loading)
        np.save(self.output_dir / 'training_history.npy', self.history)
        
        # Save as JSON for comparison script
        import json
        history_json = {k: [float(v) if isinstance(v, (np.number, torch.Tensor)) else v 
                           for v in vals] 
                       for k, vals in self.history.items()}
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")
