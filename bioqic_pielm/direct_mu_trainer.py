"""
Direct Mu Trainer
=================

Training loop for direct μ parameterization (gradient-based inverse solver).

Key difference from MRETrainer:
- Optimizes μ values directly, not neural network weights
- Classical gradient descent: μ^(k+1) = μ^k - α ∇_μ J
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

try:
    from .direct_mu_model import DirectMuModel
except ImportError:
    from direct_mu_model import DirectMuModel


class DirectMuTrainer:
    """Trainer for direct μ parameterization (no neural network)."""

    def __init__(
        self,
        model: DirectMuModel,
        device: torch.device,
        output_dir: str = "./outputs",
        loss_type: str = 'mse'
    ):
        """Initialize trainer.

        Args:
            model: DirectMuModel instance
            device: Torch device
            output_dir: Directory for saving results
            loss_type: Loss function type ('mse', 'correlation', 'relative_l2', 'sobolev')
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
            'mu_mse': [],
            'grad_norm': [],
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

        We use finite differences to compute the Laplacian:
        ∇²u ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²

        NORMALIZATION: The residual is normalized by dividing by rho_omega2 before squaring.
        This brings the PDE loss to O(1) scale, preventing it from overwhelming other losses.

        Normalized residual: (μ∇²u + ρω²u) / ρω² = (μ/ρω²)∇²u + u

        Backprop chain:
        ∂L_PDE/∂μ = ∂/∂μ [||(μ/ρω²)∇²u + u||²]
                  = 2 * [(μ/ρω²)∇²u + u] * (∇²u/ρω²)

        Args:
            x: (N, dim) coordinates
            u: (N, 1) displacement - from forward solver
            mu: (N, 1) stiffness - learnable parameter
            rho_omega2: Physics parameter (scalar)

        Returns:
            Normalized PDE residual loss (scalar with grad_fn for backprop)
        """
        N = x.shape[0]
        if N <= 10:
            # Not enough points for finite differences
            return torch.tensor(0.0, device=u.device, requires_grad=True)

        # Central differences for Laplacian
        # ∇²u ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²
        u_f = u[2:]    # forward (i+1)
        u_c = u[1:-1]  # center (i)
        u_b = u[:-2]   # backward (i-1)

        # Spatial spacing (use L2 norm for 3D)
        dx = torch.norm(x[2:] - x[:-2], dim=1, keepdim=True) / 2.0 + 1e-8

        # Laplacian approximation
        laplacian_u = (u_f - 2 * u_c + u_b) / (dx ** 2)

        # Get corresponding mu values at center points
        mu_c = mu[1:-1]

        # NORMALIZED PDE residual: divide by rho_omega2 to scale to O(1)
        # Original: μ∇²u + ρω²u
        # Normalized: (μ/ρω²)∇²u + u
        residual_normalized = (mu_c / rho_omega2) * laplacian_u + u_c

        # Mean squared residual (now scaled to ~O(u²))
        # Gradients will flow:
        # ∂L_PDE/∂μ = 2 * residual_normalized * (laplacian_u / rho_omega2)
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
        """Compute Sobolev loss: α||u - u_meas||² + β||∇u - ∇u_meas||²"""
        # L2 term
        loss_l2 = torch.mean((u_pred - u_meas) ** 2)

        # Gradient term (finite differences)
        N = x.shape[0]
        if N > 10:
            du_pred = u_pred[1:] - u_pred[:-1]
            du_meas = u_meas[1:] - u_meas[:-1]
            dx = torch.norm(x[1:] - x[:-1], dim=1, keepdim=True) + 1e-8
            grad_pred = du_pred / dx
            grad_meas = du_meas / dx
            loss_grad = torch.mean((grad_pred - grad_meas) ** 2)
        else:
            loss_grad = torch.tensor(0.0, device=u_pred.device, requires_grad=True)

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
        lr: float = 10.0,  # Higher LR for direct optimization
        bc_weight: float = 200.0,
        data_weight: float = 0.0,
        tv_weight: float = 0.001,
        pde_weight: float = 0.0,  # NEW: PDE residual weight
        lr_decay_step: int = 1000,
        lr_decay_gamma: float = 0.9,
        grad_clip: float = 0.0,  # Disabled by default
        early_stopping: int = 1500,
        log_interval: int = 100,
        save_interval: int = 500
    ) -> Dict:
        """
        Train via direct μ optimization.

        Args:
            x: (N, dim) coordinates
            u_meas: (N, 1) measured displacement
            mu_true: (N, 1) ground truth stiffness
            bc_indices: (K,) boundary indices
            u_bc_vals: (K, 1) boundary values
            rho_omega2: Physics parameter
            iterations: Max iterations
            lr: Learning rate (typically higher than NN-based)
            bc_weight: BC enforcement weight
            data_weight: Data constraint weight
            tv_weight: Total variation regularization
            lr_decay_step: LR decay schedule step
            lr_decay_gamma: LR decay factor
            grad_clip: Gradient clipping (0 = disabled)
            early_stopping: Patience for early stopping
            log_interval: Iterations between logs
            save_interval: Iterations between saving plots

        Returns:
            Dictionary with final results
        """
        print("=" * 70)
        print("Direct Mu Optimization (Gradient-Based Inverse)")
        print("=" * 70)
        print(f"  Points: {len(x)}")
        print(f"  BC points: {len(bc_indices)}")
        print(f"  Iterations: {iterations}")
        print(f"  LR: {lr}, decay every {lr_decay_step} by {lr_decay_gamma}")
        print(f"  Weights: bc={bc_weight}, data={data_weight}, tv={tv_weight}")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Optimizer: Adam on μ_field directly")
        print("=" * 70)

        # Optimizer - directly on μ_field parameter
        optimizer = torch.optim.Adam([self.model.mu_field], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )

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

            # Compute loss
            u_p_flat = u_pred.view(-1)
            u_m_flat = u_meas.view(-1)

            # MSE
            loss_mse = torch.mean((u_pred - u_meas) ** 2)

            # Correlation
            u_p_norm = u_p_flat / (torch.norm(u_p_flat) + 1e-8)
            u_m_norm = u_m_flat / (torch.norm(u_m_flat) + 1e-8)
            correlation = torch.dot(u_p_norm, u_m_norm)
            loss_correlation = 1.0 - correlation

            # Relative L2
            loss_relative_l2 = torch.sqrt(loss_mse) / (torch.norm(u_m_flat) + 1e-8)

            # Sobolev
            loss_sobolev = self._compute_sobolev_loss(u_pred, u_meas, x)

            # PDE Residual Loss (NEW!)
            # Compute physics residual: ||∇·(μ∇u) + ρω²u||²
            # This provides DIRECT gradients through the physics equation
            loss_pde = self._compute_pde_residual(x, u_pred, mu_pred, rho_omega2)

            # Select data loss
            if self.loss_type == 'mse':
                loss_data = loss_mse
            elif self.loss_type == 'correlation':
                loss_data = loss_correlation
            elif self.loss_type == 'relative_l2':
                loss_data = loss_relative_l2
            elif self.loss_type == 'sobolev':
                loss_data = loss_sobolev
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # TV regularization
            if self.model.input_dim == 1:
                loss_tv = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))
            else:
                loss_tv = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))

            # CRITICAL FIX: Add prior regularization to prevent divergence
            # Penalize deviation from mean stiffness (encourages smoothness)
            mu_mean = (self.model.mu_min + self.model.mu_max) / 2
            loss_prior = torch.mean((mu_pred - mu_mean) ** 2) / (mu_mean ** 2)  # Normalized

            # Total loss WITH PDE term
            # PDE loss provides direct physics-based gradients on μ
            loss_total = loss_data + tv_weight * loss_tv + pde_weight * loss_pde + 1e-6 * loss_prior

            # Backward pass
            loss_total.backward()

            # Gradient clipping (optional)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([self.model.mu_field], max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

            # Metrics
            with torch.no_grad():
                grad_norm = self.model.mu_field.grad.norm().item()
                mu_mse = torch.mean((mu_pred - mu_true) ** 2).item()

            # Record history
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
            self.history['total_loss'].append(loss_total.item())
            self.history['grad_norm'].append(grad_norm)
            self.history['mu_mse'].append(mu_mse)
            self.history['mu_min'].append(mu_pred.min().item())
            self.history['mu_max'].append(mu_pred.max().item())

            # Logging
            if i % log_interval == 0 or i == iterations - 1:
                print(f"Iter {i:5d}: {self.loss_type.upper()}={loss_data.item():.4e}, "
                      f"PDE={loss_pde.item():.4e}, MuMSE={mu_mse:.4e}, Grad={grad_norm:.4e}")
                print(f"           mu=[{mu_pred.min().item():.0f}, {mu_pred.max().item():.0f}] Pa, "
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
        """Save progress visualization with spatial maps."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        mu_p_np = mu_pred.detach().cpu().numpy().flatten()
        mu_t_np = mu_true.detach().cpu().numpy().flatten()
        u_p_np = u_pred.detach().cpu().numpy().flatten()
        u_m_np = u_meas.detach().cpu().numpy().flatten()
        x_np = x.detach().cpu().numpy()

        # === ROW 1: SPATIAL MU MAPS ===

        # True Mu Map
        ax = fig.add_subplot(gs[0, 0])
        if x.shape[1] == 3:
            z_coords = x_np[:, 2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            z_range = z_coords.max() - z_coords.min()
            mid_slice_mask = np.abs(z_coords - z_mid) < (z_range * 0.15)

            if mid_slice_mask.sum() > 0:
                sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                               c=mu_t_np[mid_slice_mask], s=30, cmap='jet',
                               vmin=mu_t_np.min(), vmax=mu_t_np.max(), alpha=0.7)
                plt.colorbar(sc, ax=ax, label='μ (Pa)')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('True μ Map (XY)')
                ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Predicted Mu Map
        ax = fig.add_subplot(gs[0, 1])
        if x.shape[1] == 3 and mid_slice_mask.sum() > 0:
            sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                           c=mu_p_np[mid_slice_mask], s=30, cmap='jet',
                           vmin=mu_t_np.min(), vmax=mu_t_np.max(), alpha=0.7)
            plt.colorbar(sc, ax=ax, label='μ (Pa)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Predicted μ Map (iter {iteration})')
            ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Mu Histogram
        ax = fig.add_subplot(gs[0, 2])
        bins = np.linspace(min(mu_t_np.min(), mu_p_np.min()),
                          max(mu_t_np.max(), mu_p_np.max()), 40)
        ax.hist(mu_t_np, bins=bins, alpha=0.6, label='True', color='blue', density=True)
        ax.hist(mu_p_np, bins=bins, alpha=0.6, label='Pred', color='orange', density=True)
        ax.set_xlabel('μ (Pa)')
        ax.set_ylabel('Density')
        ax.set_title('μ Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # === ROW 2: SCATTER PLOTS ===

        # Predicted vs True Mu
        ax = fig.add_subplot(gs[1, 0])
        ax.scatter(mu_t_np, mu_p_np, alpha=0.4, s=5, c='blue')
        lims = [min(mu_t_np.min(), mu_p_np.min()), max(mu_t_np.max(), mu_p_np.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_xlabel('True μ (Pa)')
        ax.set_ylabel('Predicted μ (Pa)')
        ax.set_title(f'Stiffness (iter {iteration})')
        ax.legend()
        ax.grid(True)

        # Displacement Fit
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(u_m_np, u_p_np, alpha=0.4, s=5, c='purple')
        lims = [min(u_m_np.min(), u_p_np.min()), max(u_m_np.max(), u_p_np.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_xlabel('Measured u (m)')
        ax.set_ylabel('Predicted u (m)')
        ax.set_title('Displacement Fit')
        ax.legend()
        ax.grid(True)

        # Gradient Norm
        ax = fig.add_subplot(gs[1, 2])
        ax.semilogy(self.history['grad_norm'], linewidth=2, color='darkgreen')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm (∇_μ J)')
        ax.grid(True)

        # === ROW 3: LOSS HISTORIES ===

        # Data Loss
        ax = fig.add_subplot(gs[2, 0])
        ax.semilogy(self.history['data_loss'], label=f'{self.loss_type.upper()}', linewidth=2, color='blue')
        if 'pde_loss' in self.history and np.any(np.array(self.history['pde_loss']) > 0):
            ax.semilogy(self.history['pde_loss'], label='PDE', linewidth=2, alpha=0.7, color='magenta')
        if 'mse_loss' in self.history and self.loss_type != 'mse':
            ax.semilogy(self.history['mse_loss'], label='MSE', linewidth=2, alpha=0.5, color='cyan')
        if np.any(np.array(self.history['tv_loss']) > 0):
            ax.semilogy(self.history['tv_loss'], label='TV', linewidth=2, alpha=0.5, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Data Loss, PDE & TV')
        ax.grid(True)

        # Mu MSE (SEPARATE - CRITICAL!)
        ax = fig.add_subplot(gs[2, 1])
        ax.semilogy(self.history['mu_mse'], linewidth=2, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mu MSE')
        ax.set_title('Stiffness MSE (SHOULD DECREASE!)')
        ax.grid(True)

        # Mu Range Evolution
        ax = fig.add_subplot(gs[2, 2])
        ax.fill_between(range(len(self.history['mu_min'])),
                        self.history['mu_min'],
                        self.history['mu_max'],
                        alpha=0.3, label='Predicted range')
        ax.axhline(mu_t_np.min(), color='r', linestyle='--', label=f'True min: {mu_t_np.min():.0f}')
        ax.axhline(mu_t_np.max(), color='r', linestyle='--', label=f'True max: {mu_t_np.max():.0f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('μ (Pa)')
        ax.set_title('μ Range Evolution')
        ax.legend(fontsize=8)
        ax.grid(True)

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
        """Save final results with spatial maps and histograms."""
        # Create larger figure with spatial maps
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        mu_p_np = mu_pred.detach().cpu().numpy().flatten()
        mu_t_np = mu_true.detach().cpu().numpy().flatten()
        u_p_np = u_pred.detach().cpu().numpy().flatten()
        u_m_np = u_meas.detach().cpu().numpy().flatten()
        x_np = x.detach().cpu().numpy()

        # === ROW 1: SPATIAL MU MAPS ===

        # 1. True Mu Map (X-Y plane, mid Z-slice)
        ax = fig.add_subplot(gs[0, 0])
        if x.shape[1] == 3:  # 3D data
            z_coords = x_np[:, 2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            z_range = z_coords.max() - z_coords.min()
            mid_slice_mask = np.abs(z_coords - z_mid) < (z_range * 0.15)  # Take middle 15% for better coverage

            if mid_slice_mask.sum() > 0:
                sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                               c=mu_t_np[mid_slice_mask], s=50, cmap='jet',
                               vmin=mu_t_np.min(), vmax=mu_t_np.max(), alpha=0.8)
                plt.colorbar(sc, ax=ax, label='μ (Pa)')
                ax.set_xlabel('X (m)', fontsize=11)
                ax.set_ylabel('Y (m)', fontsize=11)
                ax.set_title('True μ Map (XY, Mid Z-slice)', fontsize=13, fontweight='bold')
                ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 2. Predicted Mu Map (X-Y plane, mid Z-slice)
        ax = fig.add_subplot(gs[0, 1])
        if x.shape[1] == 3:  # 3D data
            if mid_slice_mask.sum() > 0:
                sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                               c=mu_p_np[mid_slice_mask], s=50, cmap='jet',
                               vmin=mu_t_np.min(), vmax=mu_t_np.max(), alpha=0.8)  # Same colorbar scale
                plt.colorbar(sc, ax=ax, label='μ (Pa)')
                ax.set_xlabel('X (m)', fontsize=11)
                ax.set_ylabel('Y (m)', fontsize=11)
                ax.set_title('Predicted μ Map (XY, Mid Z-slice)', fontsize=13, fontweight='bold')
                ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 3. Mu Error Map (X-Y plane, mid Z-slice)
        ax = fig.add_subplot(gs[0, 2])
        if x.shape[1] == 3:  # 3D data
            mu_error = mu_p_np - mu_t_np
            if mid_slice_mask.sum() > 0:
                error_max = np.abs(mu_error).max()
                sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                               c=mu_error[mid_slice_mask], s=50, cmap='RdBu_r',
                               vmin=-error_max, vmax=error_max, alpha=0.8)
                plt.colorbar(sc, ax=ax, label='Error (Pa)')
                ax.set_xlabel('X (m)', fontsize=11)
                ax.set_ylabel('Y (m)', fontsize=11)
                ax.set_title('μ Error Map (Pred - True)', fontsize=13, fontweight='bold')
                ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # === ROW 2: HISTOGRAMS & DISTRIBUTIONS ===

        # 4. Mu Histogram (Overlaid)
        ax = fig.add_subplot(gs[1, 0])
        bins = np.linspace(min(mu_t_np.min(), mu_p_np.min()),
                          max(mu_t_np.max(), mu_p_np.max()), 50)
        ax.hist(mu_t_np, bins=bins, alpha=0.6, label='True', color='blue', density=True, edgecolor='black')
        ax.hist(mu_p_np, bins=bins, alpha=0.6, label='Predicted', color='orange', density=True, edgecolor='black')
        ax.set_xlabel('μ (Pa)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Stiffness Distribution (Histogram)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 5. Predicted vs True Mu (Scatter)
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(mu_t_np, mu_p_np, alpha=0.4, s=10, c='blue')
        lims = [min(mu_t_np.min(), mu_p_np.min()), max(mu_t_np.max(), mu_p_np.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_xlabel('True μ (Pa)', fontsize=12)
        ax.set_ylabel('Predicted μ (Pa)', fontsize=12)
        ax.set_title('Predicted vs True Stiffness', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)

        # 6. Mu Error Histogram
        ax = fig.add_subplot(gs[1, 2])
        mu_error = mu_p_np - mu_t_np
        ax.hist(mu_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(mu_error.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {mu_error.mean():.0f} Pa')
        ax.set_xlabel('μ Error (Pa)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Error Distribution (Std: {mu_error.std():.0f} Pa)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # === ROW 3: DISPLACEMENT FIT ===

        # 7. Displacement Scatter
        ax = fig.add_subplot(gs[2, 0])
        ax.scatter(u_m_np, u_p_np, alpha=0.4, s=10, c='purple')
        lims = [min(u_m_np.min(), u_p_np.min()), max(u_m_np.max(), u_p_np.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect')
        ax.set_xlabel('Measured u (m)', fontsize=12)
        ax.set_ylabel('Predicted u (m)', fontsize=12)
        ax.set_title('Displacement Fit', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)

        # 8. Displacement Error Histogram
        ax = fig.add_subplot(gs[2, 1])
        u_error = u_p_np - u_m_np
        ax.hist(u_error, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        mse = np.mean(u_error**2)
        ax.set_xlabel('u Error (m)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Displacement Error (MSE: {mse:.2e} m²)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 9. Displacement Map (if 3D)
        ax = fig.add_subplot(gs[2, 2])
        if x.shape[1] == 3 and mid_slice_mask.sum() > 0:
            sc = ax.scatter(x_np[mid_slice_mask, 0], x_np[mid_slice_mask, 1],
                           c=u_p_np[mid_slice_mask], s=50, cmap='viridis', alpha=0.8)
            plt.colorbar(sc, ax=ax, label='u (m)')
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_title('Predicted Displacement Map', fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # === ROW 4: LOSS HISTORIES ===

        # 10. Data Loss, PDE & TV
        ax = fig.add_subplot(gs[3, 0])
        ax.semilogy(self.history['data_loss'], label=f'{self.loss_type.upper()}', linewidth=2, color='blue')
        if 'pde_loss' in self.history and np.any(np.array(self.history['pde_loss']) > 0):
            ax.semilogy(self.history['pde_loss'], label='PDE Residual', linewidth=2, alpha=0.7, color='magenta')
        if 'mse_loss' in self.history and self.loss_type != 'mse':
            ax.semilogy(self.history['mse_loss'], label='MSE', linewidth=2, alpha=0.5, color='cyan')
        if np.any(np.array(self.history['tv_loss']) > 0):
            ax.semilogy(self.history['tv_loss'], label='TV', linewidth=2, alpha=0.5, color='green')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Data Loss, PDE & TV vs Iteration', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)

        # 11. Mu MSE Propagation
        ax = fig.add_subplot(gs[3, 1])
        ax.semilogy(self.history['mu_mse'], linewidth=2, color='red')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mu MSE', fontsize=12)
        ax.set_title('Stiffness MSE vs Iteration', fontsize=13, fontweight='bold')
        ax.grid(True)

        # 12. Gradient Norm
        ax = fig.add_subplot(gs[3, 2])
        ax.semilogy(self.history['grad_norm'], linewidth=2, color='darkgreen')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm (∇_μ J)', fontsize=13, fontweight='bold')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_results.png', dpi=200)
        plt.close()

        # Save history
        np.save(self.output_dir / 'training_history.npy', self.history)

        import json
        history_json = {k: [float(v) if isinstance(v, (np.number, torch.Tensor)) else v
                           for v in vals]
                       for k, vals in self.history.items()}
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history_json, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")
